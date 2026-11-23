"""Decisions a person makes in the exceptions queue.

- accept a suggested match  -> becomes a "manual" match, its lines leave the queue
- resolve an item           -> records the person's bucket and note
- reopen an item            -> clears the decision

Every change refreshes the run summary, so the summary view stays current.
"""
from __future__ import annotations

from .engine import RunResult, summarize_parts
from .models import BUCKETS, LAYER_MANUAL, Match


def _split(result: RunResult):
    bank = [t for t in result.transactions if t.source == "bank"]
    ledger = [t for t in result.transactions if t.source == "ledger"]
    return bank, ledger


def refresh_summary(result: RunResult) -> None:
    bank, ledger = _split(result)
    result.summary = summarize_parts(bank, ledger, result.matches, result.exceptions)


def _find(result: RunResult, txn_id: str):
    for e in result.exceptions:
        if e.transaction_id == txn_id:
            return e
    raise KeyError(f"{txn_id} is not an open exception")


def accept_suggestion(result: RunResult, txn_id: str, index: int = 0, note: str = "") -> Match:
    exc = _find(result, txn_id)
    if index >= len(exc.suggestions):
        raise ValueError("That item has no such suggestion")
    s = exc.suggestions[index]
    ids = s.bank_ids + s.ledger_ids
    open_ids = {e.transaction_id for e in result.exceptions}
    taken = [i for i in ids if i not in open_ids]
    if taken:
        raise ValueError(f"Already matched elsewhere: {', '.join(taken)}")
    if s.transposition:
        raise ValueError("The amounts differ (digits look swapped). Correct the entry and mark it "
                         "resolved as an error instead of matching it.")

    nb, nl = len(s.bank_ids), len(s.ledger_ids)
    match_type = "one_to_one" if nb == 1 and nl == 1 else "one_to_many" if nb == 1 else "many_to_one"
    reason = f"Confirmed by user. System's evidence: {s.reason}"
    if note:
        reason += f". Note: {note}"
    m = Match(match_id=f"M{len(result.matches) + 1:05d}", layer=LAYER_MANUAL,
              match_type=match_type, confidence=100.0, reason=reason, transaction_ids=ids)
    result.matches.append(m)

    gone = set(ids)
    result.exceptions = [e for e in result.exceptions if e.transaction_id not in gone]
    for e in result.exceptions:
        e.suggestions = [x for x in e.suggestions if not gone & set(x.bank_ids + x.ledger_ids)]
    refresh_summary(result)
    return m


def resolve(result: RunResult, txn_id: str, bucket: str, note: str = "") -> None:
    if bucket not in BUCKETS:
        raise ValueError(f"Unknown reason {bucket!r}")
    exc = _find(result, txn_id)
    exc.decided_bucket = bucket
    exc.note = note
    refresh_summary(result)


def reopen(result: RunResult, txn_id: str) -> None:
    exc = _find(result, txn_id)
    exc.decided_bucket = None
    refresh_summary(result)
