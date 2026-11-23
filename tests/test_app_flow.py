"""Review decisions, the reconciliation report, column guessing, and the Streamlit app."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon import run_matching  # noqa: E402
from recon.autodetect import guess_date_format, guess_mapping  # noqa: E402
from recon.normalize import normalize  # noqa: E402
from recon.report import build_reconciliation, journal_rows, render_html  # noqa: E402
from recon.review import accept_suggestion, reopen, resolve  # noqa: E402
from test_engine import D0, txn  # noqa: E402


def sample_result():
    bank_text = (ROOT / "data/bank.csv").read_text()
    led_text = (ROOT / "data/ledger.csv").read_text()
    b = normalize(bank_text, guess_mapping(bank_text, "bank"))
    l = normalize(led_text, guess_mapping(led_text, "ledger"))
    assert b.ok and l.ok
    meta = json.loads((ROOT / "data/meta.json").read_text())
    r = run_matching(b.transactions, l.transactions, period_end=date.fromisoformat(meta["period"]["end"]))
    cents = lambda k: round(float(meta["balances"][k]) * 100)  # noqa: E731
    return r, cents("bank_closing"), cents("ledger_closing")


# --- column guessing -------------------------------------------------------------

def test_guess_mapping_for_both_sample_formats():
    bm = guess_mapping((ROOT / "data/bank.csv").read_text(), "bank")
    lm = guess_mapping((ROOT / "data/ledger.csv").read_text(), "ledger")
    assert bm["date"] == {"column": "Posting Date", "format": "%m/%d/%Y"}
    assert bm["amount"] == {"debit": "Debit", "credit": "Credit", "debit_is_inflow": False}
    assert bm["reference"] == "Bank Ref" and bm["balance"] == "Running Balance"
    assert lm["amount"] == {"column": "Amount", "sign": 1}
    assert lm["reference"] == "Reference" and lm["external_id"] == "Entry ID"


def test_guess_date_format_prefers_a_format_that_fits_every_value():
    assert guess_date_format(["03/11/2026", "25/11/2026"]) == "%d/%m/%Y"
    assert guess_date_format(["11/03/2026", "11/25/2026"]) == "%m/%d/%Y"
    assert guess_date_format(["not a date"]) is None


def test_ledger_debit_credit_columns_use_book_convention():
    text = "Date,Memo,Debit,Credit\n2026-11-01,Deposit,100.00,\n2026-11-02,Rent,,50.00\n"
    m = guess_mapping(text, "ledger")
    assert [t.amount_cents for t in normalize(text, m).transactions] == [10000, -5000]


# --- review decisions -------------------------------------------------------------

def test_accept_suggestion_creates_manual_match_and_updates_summary():
    bank = [txn("bank", 1, 2, -500.00, "ACH DEBIT UL"), txn("bank", 2, 3, -500.00, "ACH DEBIT UL")]
    ledger = [txn("ledger", 1, 0, -500.00, "Bill payment - Uline"),
              txn("ledger", 2, 1, -500.00, "Bill payment - Uline")]
    r = run_matching(bank, ledger)
    assert r.summary["exceptions"]["open"] == 4
    m = accept_suggestion(r, "bank:1", 0, note="checked the invoice")
    assert m.layer == "manual" and "checked the invoice" in m.reason
    assert r.summary["matches_by_layer"] == {"manual": 1}
    assert r.summary["exceptions"]["open"] == 2
    # suggestions pointing at lines that are now matched are gone
    for e in r.exceptions:
        for s in e.suggestions:
            assert not set(m.transaction_ids) & set(s.bank_ids + s.ledger_ids)


def test_accept_refuses_transpositions_and_taken_lines():
    r = run_matching([txn("bank", 1, 2, -1243.00, "ACH DEBIT GRAINGER")],
                     [txn("ledger", 1, 0, -1234.00, "Bill payment - Grainger")])
    with pytest.raises(ValueError, match="digits"):
        accept_suggestion(r, "bank:1")


def test_resolve_and_reopen():
    r = run_matching([txn("bank", 1, 1, -15.00, "MONTHLY SVC FEE")], [])
    resolve(r, "bank:1", "missing_from_books", "monthly fee")
    assert r.exceptions[0].resolved and r.summary["exceptions"]["resolved"] == 1
    with pytest.raises(ValueError):
        resolve(r, "bank:1", "not_a_bucket")
    reopen(r, "bank:1")
    assert not r.exceptions[0].resolved


# --- reconciliation report -------------------------------------------------------------

def test_sample_reconciles_to_zero_and_open_items_offset():
    r, bank_bal, book_bal = sample_result()
    rec = build_reconciliation(r, bank_bal, book_bal)
    assert rec.difference == 0
    assert rec.status == "balanced_open_items"
    labels = {i.label for i in rec.bank_items}
    assert labels == {"Outstanding payment"}


def test_wrong_starting_balance_is_unexplained():
    r, bank_bal, book_bal = sample_result()
    rec = build_reconciliation(r, bank_bal + 10000, book_bal)
    assert rec.status == "unexplained" and rec.difference == 10000


def test_resolving_everything_gives_reconciled():
    r, bank_bal, book_bal = sample_result()
    for e in list(r.exceptions):
        if e.transaction_id not in {x.transaction_id for x in r.exceptions}:
            continue
        if e.bucket is None and e.suggestions and not e.suggestions[0].transposition:
            try:
                accept_suggestion(r, e.transaction_id)
            except ValueError:
                pass
    for e in r.exceptions:
        if e.bucket is None:
            resolve(r, e.transaction_id, "real_problem")
        elif not e.resolved:
            resolve(r, e.transaction_id, e.bucket)
    rec = build_reconciliation(r, bank_bal, book_bal)
    assert rec.status == "reconciled" and rec.difference == 0


def test_journal_entries_balance_and_html_renders():
    r, bank_bal, book_bal = sample_result()
    rec = build_reconciliation(r, bank_bal, book_bal)
    rows = journal_rows(rec)
    assert rows and len(rows) == 2 * len(rec.book_items)
    debits = sum(float(x["debit"] or 0) for x in rows)
    credits = sum(float(x["credit"] or 0) for x in rows)
    assert round(debits, 2) == round(credits, 2)
    fee = next(x for x in rows if "Bank fee" in x["memo"] and x["debit"])
    assert fee["account"] == "Bank fees"
    page = render_html(r, rec, date(2026, 11, 30))
    assert page.startswith("<!DOCTYPE html>") and "Adjusted bank balance" in page


def test_timing_item_sits_on_bank_side():
    ledger = [txn("ledger", 1, 27, -800.00, "Check 1101 - FedEx", "CHK-1101")]
    r = run_matching([], ledger, period_end=D0 + timedelta(days=28))
    rec = build_reconciliation(r, 100000, 20000)   # books already show the $800 check
    assert rec.bank_items[0].label == "Outstanding payment"
    assert rec.adjusted_bank == rec.adjusted_book == 20000


# --- Streamlit app ---------------------------------------------------------------------

def test_app_runs_end_to_end_on_sample_data():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    next(b for b in at.sidebar.button if b.label.startswith("Try it")).click().run()
    next(b for b in at.button if b.label == "Reconcile").click().run()
    assert not at.exception
    assert [t.label for t in at.tabs][0] == "Summary"
    acc = [b for b in at.button if b.label == "Accept" and not b.disabled]
    acc[0].click().run()
    assert not at.exception
    assert any("Confirmed" in t.label or "Matched" in t.label for t in at.tabs)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Difference"] == "$0.00"
    assert metrics["Resolved by you"] == "1"
