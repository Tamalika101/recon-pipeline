"""Model providers (against fake local servers), model stages, configs and the benchmark runner."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from recon import run_matching  # noqa: E402
from recon.ai import (AnthropicChat, Decision, DecisionItem, OllamaChat, OllamaDecider,  # noqa: E402
                      ProviderError, Usage, parse_llm_reply, parse_probabilities)
from recon.funnel import load_config  # noqa: E402
from recon.model_stages import ModelStage, StageConfig, build_items  # noqa: E402
from test_engine import txn  # noqa: E402

# --- a fake Ollama / Anthropic server ---------------------------------------------------


class FakeHandler(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):  # keep test output quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeHandler.requests.append((self.path, body, {k.lower(): v for k, v in self.headers.items()}))
        if self.path == "/v1/systemone":
            keys = list(body["questions"]["match"]["criteria"])
            probs = {k: (0.93 if k == "A" else 0.07 / (len(keys) - 1)) for k in keys}
            out = {"answers": {"match": {"choice": "A", "probabilities": probs}}}
        elif self.path == "/api/chat":
            out = {"message": {"role": "assistant",
                               "content": '{"choice": "A", "confidence": "high", "reason": "same vendor"}'},
                   "prompt_eval_count": 210, "eval_count": 18}
        elif self.path == "/v1/messages":
            out = {"content": [{"type": "text", "text": '{"choice":"none","confidence":"medium","reason":"x"}'}],
                   "usage": {"input_tokens": 300, "output_tokens": 20}}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), FakeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def item(n_options: int = 2) -> DecisionItem:
    opts = {chr(65 + i): f"option {i}" for i in range(n_options)}
    opts["none"] = "none of these"
    return DecisionItem("bank:1", "Bank line: ...", "Which entry?", opts,
                        {chr(65 + i): [f"ledger:{i + 1}"] for i in range(n_options)}, 100)


def twins():
    bank = [txn("bank", 1, 2, -500.00, "ACH DEBIT UL"), txn("bank", 2, 3, -500.00, "ACH DEBIT UL")]
    ledger = [txn("ledger", 1, 0, -500.00, "Bill payment - Uline"),
              txn("ledger", 2, 1, -500.00, "Bill payment - Uline")]
    return bank, ledger


class FixedProvider:
    """Always picks the same option with a set confidence."""

    def __init__(self, choice="A", confidence=0.95, fail=False):
        self.choice, self.confidence, self.fail, self.calls = choice, confidence, fail, 0

    def decide(self, it):
        self.calls += 1
        if self.fail:
            raise ProviderError("server down")
        return Decision(self.choice, self.confidence, "fixed", Usage(1000, 10, 0.2))


# --- providers ------------------------------------------------------------------------------

def test_ollama_decider_request_and_response(fake_server):
    FakeHandler.requests.clear()
    d = OllamaDecider(base_url=fake_server).decide(item())
    path, body, _ = FakeHandler.requests[-1]
    assert path == "/v1/systemone" and body["model"] == "strands-decider"
    q = body["questions"]["match"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"A", "B", "none"}
    assert d.choice == "A" and d.confidence == pytest.approx(0.93)
    assert d.usage.estimated_tokens and d.usage.input_tokens > 0


def test_ollama_chat_uses_json_mode_and_reports_tokens(fake_server):
    FakeHandler.requests.clear()
    d = OllamaChat(base_url=fake_server).decide(item())
    path, body, _ = FakeHandler.requests[-1]
    assert path == "/api/chat" and body["format"] == "json" and body["think"] is False
    assert body["stream"] is False and body["messages"][0]["role"] == "system"
    assert (d.choice, d.confidence) == ("A", 0.9)
    assert (d.usage.input_tokens, d.usage.output_tokens, d.usage.estimated_tokens) == (210, 18, False)


def test_anthropic_chat(fake_server):
    FakeHandler.requests.clear()
    d = AnthropicChat(base_url=fake_server, api_key="test-key").decide(item())
    path, body, headers = FakeHandler.requests[-1]
    assert path == "/v1/messages" and headers["x-api-key"] == "test-key"
    assert body["model"] == "claude-haiku-5-5"
    assert d.choice == "none" and d.usage.input_tokens == 300


def test_unreachable_server_raises_clear_error():
    with pytest.raises(ProviderError, match="running"):
        OllamaDecider(base_url="http://127.0.0.1:9", timeout_s=2).decide(item())


def test_parsers_tolerate_shapes():
    keys = ["A", "B", "none"]
    assert parse_probabilities({"answers": {"match": {"probabilities": [0.7, 0.2, 0.1]}}}, keys)["A"] == 0.7
    lst = {"answers": {"match": {"probabilities": [{"name": "B", "probability": 0.8},
                                                  {"name": "none", "probability": 0.2}]}}}
    assert parse_probabilities(lst, keys)["B"] == pytest.approx(0.8)
    assert parse_probabilities({"answers": {"match": {"choice": "B"}}}, keys)["B"] == 1.0
    assert parse_llm_reply('Sure! {"choice":"b","confidence":"HIGH"}', keys)[:2] == ("B", 0.9)
    assert parse_llm_reply("no json here", keys)[0] == "none"
    assert parse_llm_reply('{"choice":"Z","confidence":"high"}', keys)[0] == "none"


# --- model stages ---------------------------------------------------------------------------

def test_items_only_for_lines_with_agreeing_amounts():
    bank = [txn("bank", 1, 2, -1243.00, "ACH DEBIT GRAINGER"),      # transposed: no candidate
            txn("bank", 2, 1, -15.00, "MONTHLY SVC FEE")]           # fee: no candidate
    ledger = [txn("ledger", 1, 0, -1234.00, "Bill payment - Grainger")]
    r_state = run_matching(bank, ledger)  # rules leave all open
    assert len(r_state.exceptions) == 3
    from recon.layers import MatchState
    from recon.models import MatchConfig
    assert build_items(MatchState(bank, ledger, MatchConfig()), 4) == []


def test_decider_threshold_controls_auto_match():
    bank, ledger = twins()
    sure = ModelStage(StageConfig("decider", threshold=0.9), FixedProvider("A", 0.95))
    r = run_matching(bank, ledger, extra_layers=[sure])
    assert r.summary["matches_by_layer"].get("decider") == 2
    unsure = ModelStage(StageConfig("decider", threshold=0.9), FixedProvider("A", 0.6))
    r = run_matching(bank, ledger, extra_layers=[unsure])
    assert "decider" not in r.summary["matches_by_layer"]
    assert {u["outcome"] for u in r.usage} == {"escalated"}


def test_llm_accepts_only_configured_confidence():
    bank, ledger = twins()
    med = ModelStage(StageConfig("llm", accept_confidence=["high"]), FixedProvider("A", 0.7))
    assert "llm" not in run_matching(bank, ledger, extra_layers=[med]).summary["matches_by_layer"]
    ok = ModelStage(StageConfig("llm", accept_confidence=["high", "medium"]), FixedProvider("A", 0.7))
    assert run_matching(bank, ledger, extra_layers=[ok]).summary["matches_by_layer"]["llm"] == 2


def test_budget_cap_and_cost_accounting():
    bank, ledger = twins()
    prov = FixedProvider("none", 0.9)
    stage = ModelStage(StageConfig("llm", max_calls=1, price_in_per_mtok=2.0, price_out_per_mtok=10.0), prov)
    r = run_matching(bank, ledger, extra_layers=[stage])
    outcomes = [u["outcome"] for u in r.usage]
    assert outcomes == ["said_none", "over_budget"] and prov.calls == 1
    assert r.usage[0]["cost_usd"] == pytest.approx((1000 * 2.0 + 10 * 10.0) / 1e6)


def test_three_stage_escalation_and_error_stop():
    bank, ledger = twins()
    decider = ModelStage(StageConfig("decider", threshold=0.99), FixedProvider("A", 0.9))
    llm = ModelStage(StageConfig("llm"), FixedProvider("A", 0.9))
    r = run_matching(bank, ledger, extra_layers=[decider, llm])
    assert r.summary["matches_by_layer"] == {"llm": 2}
    assert r.remaining_after["decider"] == 4 and r.remaining_after["llm"] == 0
    broken = ModelStage(StageConfig("llm", max_consecutive_errors=2), FixedProvider(fail=True))
    with pytest.raises(ProviderError, match="2 errors"):
        run_matching(bank + [txn("bank", 3, 4, -500.00, "X")], ledger, extra_layers=[broken])


def test_model_stage_against_fake_ollama(fake_server):
    bank, ledger = twins()
    spec = StageConfig("decider", threshold=0.9, extra={"base_url": fake_server})
    r = run_matching(bank, ledger, extra_layers=[ModelStage(spec, OllamaDecider(base_url=fake_server))])
    assert r.summary["matches_by_layer"]["decider"] == 2


# --- configs and benchmark runner ---------------------------------------------------------

@pytest.mark.parametrize("path", sorted((ROOT / "benchmarks/configs").glob("*.json")), ids=lambda p: p.stem)
def test_every_shipped_config_loads(path):
    fc = load_config(path)
    assert fc.name and all(s.kind in ("decider", "llm") for s in fc.stages)


def test_benchmark_runner_end_to_end(tmp_path):
    out = tmp_path / "res"
    subprocess.run([sys.executable, str(ROOT / "benchmarks/run_benchmark.py"),
                    "--configs", str(ROOT / "benchmarks/configs/rules_only.json"),
                    str(ROOT / "benchmarks/configs/sim_rules_decider_llm.json"),
                    "--events", "300", "--seeds", "1", "--data-dir", str(tmp_path / "data"),
                    "--out", str(out)], check=True, capture_output=True)
    import csv
    rows = list(csv.DictReader(open(out / "results.csv")))
    assert len(rows) == 2
    sim = next(r for r in rows if r["simulated"] == "True")
    assert float(sim["precision"]) > 0.98 and "decider_calls" in sim and "llm_cost_usd" in sim
    base = next(r for r in rows if r["simulated"] == "False")
    assert int(sim["items_for_person"]) <= int(base["items_for_person"])
    assert "*sim*" in (out / "summary.md").read_text()
