"""Model providers for the decision-model and LLM stages.

Every provider answers one DecisionItem: a bank line plus a few candidate book
entries, labelled A, B, C... and "none". Providers return which option they pick,
how sure they are, and a Usage record (tokens and latency) so cost can be priced.

    OllamaDecider   Strands Decider via Ollama's /v1/systemone endpoint (local)
    OllamaChat      any chat model in Ollama, e.g. a local Qwen (local)
    AnthropicChat   Claude via the Anthropic Messages API (hosted, needs ANTHROPIC_API_KEY)
    Simulated*      answer from the ground truth at a set accuracy, ONLY for testing the
                    pipeline without a model; their numbers are not model results

Standard library only.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

NONE = "none"
CONFIDENCE_LEVELS = {"high": 0.9, "medium": 0.7, "low": 0.4}


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    latency_s: float
    estimated_tokens: bool = False   # True when the provider doesn't report counts


@dataclass
class DecisionItem:
    bank_id: str
    state: str                        # the evidence, as text
    instructions: str                 # the question
    options: dict[str, str]           # option key -> description, includes "none"
    option_ids: dict[str, list[str]] = field(default_factory=dict)  # key -> ledger ids
    value_cents: int = 0              # used to prioritise items under a budget


@dataclass
class Decision:
    choice: str                       # an option key, or "none"
    confidence: float                 # 0..1
    reason: str
    usage: Usage
    raw: str = ""


class ProviderError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """About 4 characters per token for English text. Marked as an estimate wherever used."""
    return max(1, round(len(text) / 4))


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise ProviderError(f"{url} returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise ProviderError(f"Can't reach {url} ({e.reason}). Is the model server running?") from e


# ---------------------------------------------------------------------------
# Decision model (system one)
# ---------------------------------------------------------------------------


class OllamaDecider:
    """Strands Decider served by Ollama (0.40+ on Apple silicon): POST /v1/systemone."""

    def __init__(self, model: str = "strands-decider", base_url: str = "http://localhost:11434",
                 timeout_s: float = 60):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout_s

    def decide(self, item: DecisionItem) -> Decision:
        body = {"model": self.model, "state": item.state,
                "questions": {"match": {"type": "choice", "instructions": item.instructions,
                                        "criteria": item.options}}}
        t0 = time.perf_counter()
        resp = _post(f"{self.base_url}/v1/systemone", body, timeout=self.timeout)
        latency = time.perf_counter() - t0
        probs = parse_probabilities(resp, list(item.options))
        choice = max(probs, key=probs.get)
        prompt = item.state + item.instructions + json.dumps(item.options)
        return Decision(choice, probs[choice], f"decision model probability {probs[choice]:.0%}",
                        Usage(estimate_tokens(prompt), 1, latency, estimated_tokens=True),
                        raw=json.dumps(resp)[:500])


def parse_probabilities(resp: dict, keys: list[str]) -> dict[str, float]:
    """Read answers.match from a /v1/systemone response, tolerating a few shapes."""
    try:
        ans = resp["answers"]["match"]
    except (KeyError, TypeError) as e:
        raise ProviderError(f"Unexpected decider response: {json.dumps(resp)[:300]}") from e
    probs = ans.get("probabilities")
    if isinstance(probs, dict):
        out = {k: float(probs.get(k, 0.0)) for k in keys}
    elif isinstance(probs, list) and len(probs) == len(keys) and all(isinstance(p, (int, float)) for p in probs):
        out = {k: float(p) for k, p in zip(keys, probs)}
    elif isinstance(probs, list):  # [{"name"/"option": k, "probability"/"p": x}, ...]
        out = {k: 0.0 for k in keys}
        for entry in probs:
            k = entry.get("name") or entry.get("option") or entry.get("choice")
            if k in out:
                out[k] = float(entry.get("probability", entry.get("p", 0.0)))
    else:
        out = {k: (1.0 if k == ans.get("choice") else 0.0) for k in keys}
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else out


# ---------------------------------------------------------------------------
# Chat models (LLM stage)
# ---------------------------------------------------------------------------

LLM_SYSTEM = (
    "You reconcile bank statements against a company's accounting records. "
    "You will see one bank line and a few candidate book entries whose amounts already agree. "
    "Decide which candidate records the same real-world transaction. Bank descriptions are often "
    "abbreviated or garbled, and banks usually post 0 to 5 days after the book date, sometimes later. "
    "Never guess: if two options fit equally well, or none clearly fits, answer \"none\". "
    'Reply with JSON only: {"choice": "<option key or none>", "confidence": "high|medium|low", '
    '"reason": "<one short sentence>"}'
)


def llm_user_prompt(item: DecisionItem) -> str:
    opts = "\n".join(f"{k}: {v}" for k, v in item.options.items())
    return f"{item.state}\n\n{item.instructions}\n\nOptions:\n{opts}"


def parse_llm_reply(text: str, keys: list[str]) -> tuple[str, float, str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return NONE, 0.0, f"unparseable reply: {text[:80]!r}"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return NONE, 0.0, f"unparseable reply: {text[:80]!r}"
    choice = str(data.get("choice", NONE)).strip()
    if choice not in keys:
        choice = next((k for k in keys if k.lower() == choice.lower()), NONE)
    level = str(data.get("confidence", "low")).lower()
    return choice, CONFIDENCE_LEVELS.get(level, 0.0), str(data.get("reason", ""))[:200]


class OllamaChat:
    """A local chat model in Ollama (e.g. qwen3.5:9b): POST /api/chat with JSON output."""

    def __init__(self, model: str = "qwen3.5:9b", base_url: str = "http://localhost:11434",
                 timeout_s: float = 300, think: bool = False, temperature: float = 0.0):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout_s
        self.think, self.temperature = think, temperature

    def decide(self, item: DecisionItem) -> Decision:
        user = llm_user_prompt(item)
        body = {"model": self.model, "stream": False, "format": "json", "think": self.think,
                "options": {"temperature": self.temperature},
                "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": user}]}
        t0 = time.perf_counter()
        resp = _post(f"{self.base_url}/api/chat", body, timeout=self.timeout)
        latency = time.perf_counter() - t0
        text = (resp.get("message") or {}).get("content", "")
        choice, conf, reason = parse_llm_reply(text, list(item.options))
        tin, tout = resp.get("prompt_eval_count"), resp.get("eval_count")
        usage = Usage(tin or estimate_tokens(LLM_SYSTEM + user), tout or estimate_tokens(text),
                      latency, estimated_tokens=not (tin and tout))
        return Decision(choice, conf, reason, usage, raw=text[:500])


class AnthropicChat:
    """Claude via the Messages API. Reads the key from ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-haiku-5-5", base_url: str = "https://api.anthropic.com",
                 timeout_s: float = 120, max_tokens: int = 200, api_key: str | None = None):
        self.model, self.base_url, self.timeout, self.max_tokens = model, base_url.rstrip("/"), timeout_s, max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def decide(self, item: DecisionItem) -> Decision:
        if not self.api_key:
            raise ProviderError("Set ANTHROPIC_API_KEY to use the Claude stage.")
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": LLM_SYSTEM,
                "messages": [{"role": "user", "content": llm_user_prompt(item)}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        t0 = time.perf_counter()
        resp = _post(f"{self.base_url}/v1/messages", body, headers, timeout=self.timeout)
        latency = time.perf_counter() - t0
        text = "".join(c.get("text", "") for c in resp.get("content", []) if c.get("type") == "text")
        choice, conf, reason = parse_llm_reply(text, list(item.options))
        u = resp.get("usage", {})
        return Decision(choice, conf, reason,
                        Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)), latency),
                        raw=text[:500])


# ---------------------------------------------------------------------------
# Simulated providers (testing only)
# ---------------------------------------------------------------------------


class _Simulated:
    """Answers from the ground truth with a configurable accuracy, to exercise the pipeline,
    budgets and cost accounting without running a model. Results are NOT model results."""

    def __init__(self, oracle: Callable[[DecisionItem], str], accuracy: float = 0.9,
                 latency_s: float = 0.05, seed: int = 0, tokens_out: int = 30):
        self.oracle, self.accuracy, self.latency, self.tokens_out = oracle, accuracy, latency_s, tokens_out
        self.rng = random.Random(seed)

    def _pick(self, item: DecisionItem) -> tuple[str, float]:
        truth = self.oracle(item)
        if self.rng.random() < self.accuracy:
            return truth, self.rng.uniform(0.80, 0.99)
        wrong = [k for k in item.options if k != truth] or [truth]
        return self.rng.choice(wrong), self.rng.uniform(0.40, 0.95)


class SimulatedDecider(_Simulated):
    def decide(self, item: DecisionItem) -> Decision:
        choice, conf = self._pick(item)
        prompt = item.state + item.instructions + json.dumps(item.options)
        return Decision(choice, conf, "simulated", Usage(estimate_tokens(prompt), 1, self.latency, True))


class SimulatedChat(_Simulated):
    def decide(self, item: DecisionItem) -> Decision:
        choice, conf = self._pick(item)
        level = "high" if conf >= 0.85 else "medium" if conf >= 0.6 else "low"
        return Decision(choice, CONFIDENCE_LEVELS[level], "simulated",
                        Usage(estimate_tokens(LLM_SYSTEM + llm_user_prompt(item)), self.tokens_out,
                              self.latency, True))


def make_provider(spec: dict, oracle: Callable[[DecisionItem], str] | None = None):
    kind, provider = spec["kind"], spec.get("provider", "ollama")
    common = {k: spec[k] for k in ("base_url", "timeout_s") if k in spec}
    if provider == "simulated":
        if oracle is None:
            raise ValueError("Simulated providers need the dataset's truth.csv")
        cls = SimulatedDecider if kind == "decider" else SimulatedChat
        return cls(oracle, accuracy=spec.get("simulated_accuracy", 0.9),
                   latency_s=spec.get("simulated_latency_s", 0.05), seed=spec.get("seed", 0))
    if kind == "decider" and provider == "ollama":
        return OllamaDecider(spec.get("model", "strands-decider"), **common)
    if kind == "llm" and provider == "ollama":
        return OllamaChat(spec.get("model", "qwen3.5:9b"), think=spec.get("think", False), **common)
    if kind == "llm" and provider == "anthropic":
        return AnthropicChat(spec.get("model", "claude-haiku-5-5"), **common)
    raise ValueError(f"No provider {provider!r} for a {kind!r} stage")
