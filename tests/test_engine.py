"""Unit tests per layer plus one end-to-end check on generated data.

Run: python -m pytest -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon import Layer, MatchConfig, register_layer, run_matching  # noqa: E402
from recon.engine import classify_exceptions  # noqa: E402
from recon.evaluate import evaluate  # noqa: E402
from recon.layers import MatchState, find_subsets, is_transposition  # noqa: E402
from recon.models import Transaction  # noqa: E402
from recon.normalize import load_mapping, normalize, normalize_file  # noqa: E402
from recon.text import clean, similarity  # noqa: E402

D0 = date(2026, 11, 2)


def txn(source: str, row: int, day: int, dollars: float, desc: str, ref: str = "") -> Transaction:
    return Transaction(id=f"{source}:{row}", source=source, source_row=row,
                       txn_date=D0 + timedelta(days=day), amount_cents=round(dollars * 100),
                       description=desc, clean_description=clean(desc), reference_id=ref)


def matched_pairs(result) -> set[frozenset]:
    return {frozenset(m.transaction_ids) for m in result.matches}


# --- normalization -----------------------------------------------------------

BANK_MAP = {"source": "bank", "date": {"column": "Posting Date", "format": "%m/%d/%Y"},
            "amount": {"debit": "Debit", "credit": "Credit"}, "description": "Description",
            "reference": "Bank Ref"}


def test_normalize_debit_credit_and_formats():
    csv_text = ("Posting Date,Description,Debit,Credit,Bank Ref\n"
                "11/03/2026,ACH DEBIT SYSCO,\"1,234.56\",,ACH-1\n"
                "11/04/2026,DEPOSIT,,$500.00,\n"
                "11/05/2026,REVERSAL,(12.00),,\n")
    res = normalize(csv_text, BANK_MAP)
    assert res.ok
    assert [t.amount_cents for t in res.transactions] == [-123456, 50000, -1200]
    assert res.transactions[0].reference_id == "ACH-1"
    assert res.transactions[0].txn_date == date(2026, 11, 3)
    assert res.reference_coverage == pytest.approx(1 / 3)


def test_normalize_reports_every_problem_with_row_numbers():
    csv_text = ("Posting Date,Description,Debit,Credit,Bank Ref\n"
                "2026-11-03,BAD DATE,10.00,,\n"
                "11/04/2026,BOTH,5.00,5.00,\n"
                "11/05/2026,OK,1.00,,\n")
    res = normalize(csv_text, BANK_MAP)
    assert [i.row for i in res.issues] == [1, 2]
    assert len(res.transactions) == 1


def test_normalize_missing_column_is_file_level_issue():
    res = normalize("Date,Description,Amount\n2026-11-01,X,1\n", BANK_MAP)
    assert res.issues and res.issues[0].row is None
    assert "Posting Date" in res.issues[0].message


# --- text similarity -----------------------------------------------------------

def test_similarity_handles_garbled_and_truncated_bank_text():
    assert similarity("ACHDBTSYSC*X7Q2", "Bill payment - Sysco BILL-561") >= 0.9
    assert similarity("SQR INC DE*Q349", "Square settlement") >= 0.9
    assert similarity("POSDBTSLCK*S*Y4LY", "Slack subscription") >= 0.9


def test_similarity_low_for_unrelated_names():
    assert similarity("ACH DEBIT SYSCO", "Bill payment - FedEx BILL-578") < 0.5
    assert similarity("MOBILE DEPOSIT", "Payment received - Acme Corp INV-1") == 0.0


def test_transposition_detection():
    assert is_transposition(-123456, -124356)
    assert not is_transposition(-123456, 124356)       # sign differs
    assert not is_transposition(2217, 20217)            # digit count differs
    assert not is_transposition(1000, 1000)


# --- layers ----------------------------------------------------------------------

def test_reference_layer_matches_and_leaves_duplicate_flagged():
    bank = [txn("bank", 1, 1, -250.00, "ACH DEBIT SYSCO", "ACH-9")]
    ledger = [txn("ledger", 1, 0, -250.00, "Bill payment - Sysco", "ACH-9"),
              txn("ledger", 2, 0, -250.00, "Bill payment - Sysco", "ACH-9")]
    r = run_matching(bank, ledger)
    assert r.matches[0].layer == "reference_id"
    exc = {e.transaction_id: e for e in r.exceptions}
    assert exc["ledger:2"].suggested_bucket == "error"
    assert "duplicate" in exc["ledger:2"].reason.lower()


def test_exact_layer_refuses_to_guess_between_twins():
    bank = [txn("bank", 1, 2, -500.00, "ACH DEBIT UL"), txn("bank", 2, 3, -500.00, "ACH DEBIT UL")]
    ledger = [txn("ledger", 1, 0, -500.00, "Bill payment - Uline"),
              txn("ledger", 2, 1, -500.00, "Bill payment - Uline")]
    r = run_matching(bank, ledger)
    assert r.matches == []
    exc = {e.transaction_id: e for e in r.exceptions}
    assert exc["bank:1"].suggested_bucket is None
    assert {s.ledger_ids[0] for s in exc["bank:1"].suggestions} == {"ledger:1", "ledger:2"}


def test_exact_layer_matches_unique_candidate():
    bank = [txn("bank", 1, 3, -77.31, "POS DEBIT ADOBE*11")]
    ledger = [txn("ledger", 1, 0, -77.31, "Adobe subscription")]
    r = run_matching(bank, ledger)
    assert r.matches[0].layer == "exact"
    assert "3 days later" in r.matches[0].reason


def test_fuzzy_layer_auto_matches_late_garbled_line():
    bank = [txn("bank", 1, 7, 1820.40, "WRINDLTLGSTCS*R5RD")]
    ledger = [txn("ledger", 1, 0, 1820.40, "Payment received - Delta Logistics INV-1")]
    r = run_matching(bank, ledger)
    assert r.matches and r.matches[0].layer == "fuzzy"
    assert r.matches[0].confidence >= 90


def test_fuzzy_layer_only_suggests_when_no_name_evidence():
    bank = [txn("bank", 1, 9, 950.00, "MBL DPST*64CI")]
    ledger = [txn("ledger", 1, 0, 950.00, "Payment received - Evergreen Dental INV-2")]
    r = run_matching(bank, ledger)
    assert r.matches == []
    assert r.exceptions[0].suggestions[0].ledger_ids == ["ledger:1"]


def test_transposed_amount_becomes_error_on_both_sides():
    bank = [txn("bank", 1, 2, -1243.00, "ACH DEBIT GRAINGER")]
    ledger = [txn("ledger", 1, 0, -1234.00, "Bill payment - Grainger")]
    r = run_matching(bank, ledger)
    assert r.matches == []
    assert {e.suggested_bucket for e in r.exceptions} == {"error"}


def test_combination_layer_matches_lump_deposit():
    bank = [txn("bank", 1, 4, 1000.00 + 2345.67 + 432.10, "DEPOSIT 3 ITEMS")]
    ledger = [txn("ledger", 1, 0, 1000.00, "Payment received - Acme"),
              txn("ledger", 2, 0, 2345.67, "Payment received - Juniper"),
              txn("ledger", 3, 1, 432.10, "Payment received - Harbor"),
              txn("ledger", 4, 1, 999.99, "Payment received - Other")]
    r = run_matching(bank, ledger, period_end=D0 + timedelta(days=30))
    m = [m for m in r.matches if m.layer == "combination"]
    assert m and set(m[0].transaction_ids) == {"bank:1", "ledger:1", "ledger:2", "ledger:3"}
    assert m[0].match_type == "one_to_many"


def test_find_subsets_reports_ambiguity():
    pool = [txn("ledger", i, 0, v, "x") for i, v in enumerate([1.0, 2.0, 3.0, 4.0], start=1)]
    sols = find_subsets(500, pool, max_size=3, tol=0, limit=5)   # $5 = 1+4 = 2+3
    assert len(sols) == 2


def test_exception_buckets_fee_unknown_timing():
    bank = [txn("bank", 1, 5, -15.00, "MONTHLY SVC FEE"),
            txn("bank", 2, 6, -412.77, "POS DEBIT XJ*9921 LAGOS NG")]
    ledger = [txn("ledger", 1, 27, -800.00, "Check 1101 - FedEx", "CHK-1101")]
    r = run_matching(bank, ledger, period_end=D0 + timedelta(days=28))
    buckets = {e.transaction_id: e.suggested_bucket for e in r.exceptions}
    assert buckets == {"bank:1": "missing_from_books", "bank:2": "real_problem", "ledger:1": "timing"}


def test_summary_counts_and_unreconciled_amount():
    bank = [txn("bank", 1, 1, -100.00, "A", "R1"), txn("bank", 2, 1, -15.00, "MONTHLY SVC FEE")]
    ledger = [txn("ledger", 1, 0, -100.00, "A", "R1")]
    s = run_matching(bank, ledger).summary
    assert s["matched"] == {"bank": 1, "ledger": 1}
    assert s["matches_by_layer"] == {"reference_id": 1}
    assert s["unreconciled_amount"] == {"bank": 15.0, "ledger": 0.0}


# --- config and plugins -------------------------------------------------------------

def test_config_round_trip_and_layer_switches():
    cfg = MatchConfig(auto_accept=85, layers=["reference_id", "exact"])
    again = MatchConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    bank = [txn("bank", 1, 7, 1820.40, "WRINDLTLGSTCS*R5RD")]
    ledger = [txn("ledger", 1, 0, 1820.40, "Payment received - Delta Logistics")]
    assert run_matching(bank, ledger, again).matches == []   # fuzzy switched off


def test_custom_layer_plugs_into_the_funnel():
    @register_layer
    class MemoNumberLayer(Layer):
        """Example plugin: match when the bank text contains the ledger's invoice number."""
        name = "memo_number_test"

        def run(self, state: MatchState) -> None:
            for b in state.open_bank():
                for l in state.open_ledger():
                    if l.reference_id and l.reference_id in b.description:
                        state.add_match([b], [l], self.name, 99, "Invoice number in bank text")
                        break

    bank = [txn("bank", 1, 20, 300.00, "WIRE IN INV-77 THANKS")]
    ledger = [txn("ledger", 1, 0, 300.00, "Payment received", "INV-77")]
    cfg = MatchConfig(layers=["memo_number_test", "reference_id", "exact", "fuzzy"])
    r = run_matching(bank, ledger, cfg)
    assert r.matches[0].layer == "memo_number_test"


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError):
        run_matching([], [], MatchConfig(layers=["nope"]))


# --- end to end ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7])
def test_end_to_end_on_generated_data(tmp_path, seed):
    out = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "synth/generate.py"), "--seed", str(seed),
                    "--out", str(out)], check=True, capture_output=True)
    meta = json.loads((out / "meta.json").read_text())
    bank = normalize_file(out / "bank.csv", load_mapping(ROOT / "mappings/bank_default.json"))
    ledger = normalize_file(out / "ledger.csv", load_mapping(ROOT / "mappings/ledger_default.json"))
    assert bank.ok and ledger.ok
    r = run_matching(bank.transactions, ledger.transactions,
                     period_end=date.fromisoformat(meta["period"]["end"]))
    ev = evaluate(r, out / "truth.csv")
    assert ev["pairs"]["precision"] == 1.0          # never a wrong automatic match
    assert ev["pairs"]["recall"] >= 0.95
    assert ev["exception_bucket_accuracy"] == 1.0
    for name in ("shared_id", "clean_no_id", "ambiguous_twins", "lump_deposit"):
        sc = ev["scenarios"][name]
        assert sc["correct"] == sc["total"], name
    g = ev["scenarios"]["garbled"]
    assert g["correct"] + g["top_suggestion"] == g["total"]   # auto, or one click away
