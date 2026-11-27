"""Run the funnel, classify what's left over, and summarize the run."""
from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from . import layers as _layers  # noqa: F401  (registers the built-in layers)
from .layers import LAYER_REGISTRY, MatchState
from .models import (BUCKET_ERROR, BUCKET_MISSING, BUCKET_PROBLEM, BUCKET_TIMING,
                     ExceptionItem, Match, MatchConfig, Suggestion, Transaction)

FEE_PATTERN = re.compile(r"\b(FEE|FEES|SVC|SERVICE CHARGE|INTEREST|OVERDRAFT)\b")


@dataclass
class RunResult:
    config: MatchConfig
    transactions: list[Transaction]
    matches: list[Match]
    suggestions: list[Suggestion]
    exceptions: list[ExceptionItem]
    summary: dict
    usage: list[dict] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    remaining_after: dict[str, int] = field(default_factory=dict)


def run_matching(bank: list[Transaction], ledger: list[Transaction],
                 cfg: MatchConfig | None = None, period_end: date | None = None,
                 extra_layers: list | None = None) -> RunResult:
    """Run the rule layers named in cfg.layers, then any extra stage objects (e.g. model stages)."""
    cfg = cfg or MatchConfig()
    unknown = [name for name in cfg.layers if name not in LAYER_REGISTRY]
    if unknown:
        raise ValueError(f"unknown matching layer(s): {unknown}")

    state = MatchState(bank, ledger, cfg)
    stages = [LAYER_REGISTRY[name]() for name in cfg.layers] + list(extra_layers or [])
    for layer in stages:
        t0 = time.perf_counter()
        layer.run(state)
        state.stage_seconds[layer.name] = round(time.perf_counter() - t0, 4)
        state.remaining_after[layer.name] = len(state.unmatched)

    period_end = period_end or max(t.txn_date for t in state.by_id.values())
    exceptions = classify_exceptions(state, period_end)
    summary = summarize(state, exceptions)
    return RunResult(cfg, state.bank + state.ledger, state.matches, state.suggestions,
                     exceptions, summary, state.usage, state.stage_seconds, state.remaining_after)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def classify_exceptions(state: MatchState, period_end: date) -> list[ExceptionItem]:
    """Every unmatched transaction becomes an exception with a *suggested* bucket.

    The bucket is a starting point for the preparer, never a final decision.
    None means the system can't tell and a person has to choose.
    """
    cfg = state.cfg
    by_txn: dict[str, list[Suggestion]] = defaultdict(list)
    for s in state.suggestions:
        if all(i in state.unmatched for i in s.bank_ids + s.ledger_ids):
            for i in s.bank_ids + s.ledger_ids:
                by_txn[i].append(s)
    for lst in by_txn.values():
        lst.sort(key=lambda s: -s.confidence)

    matched_ledger = [state.by_id[i] for m in state.matches for i in m.transaction_ids
                      if i.startswith("ledger:")]
    seen_ref_amt = {(t.reference_id, t.amount_cents) for t in matched_ledger if t.reference_id}
    seen_doc = {(t.txn_date, t.amount_cents, t.description) for t in matched_ledger}

    out: list[ExceptionItem] = []
    for t in state.bank + state.ledger:
        if t.id not in state.unmatched:
            continue
        sugg = by_txn.get(t.id, [])
        transposed = next((s for s in sugg if s.transposition), None)

        combo = bool(sugg) and sugg[0].layer == "combination"
        if t.source == "bank":
            if FEE_PATTERN.search(t.description.upper()):
                bucket, why = BUCKET_MISSING, "Bank fee or interest not recorded in the books; post an entry"
            elif transposed:
                bucket, why = BUCKET_ERROR, ("Amount matches a ledger line with two digits swapped "
                                             "(difference is a multiple of 9); correct the ledger entry")
            elif t.id in state.ambiguous and combo:
                bucket, why = None, "More than one group of ledger lines adds up to this amount; pick the right one"
            elif t.id in state.ambiguous:
                bucket, why = None, "Several ledger lines fit equally well; pick the right one"
            elif sugg:
                bucket, why = None, "Possible match found; confirm or reject it"
            elif t.amount_cents > 0:
                bucket, why = BUCKET_MISSING, "Deposit with no ledger entry; record it or find its source"
            else:
                bucket, why = BUCKET_PROBLEM, "Unrecognized bank debit with no ledger entry; investigate"
        else:
            if (t.reference_id and (t.reference_id, t.amount_cents) in seen_ref_amt) or \
                    (t.txn_date, t.amount_cents, t.description) in seen_doc:
                bucket, why = BUCKET_ERROR, "Likely duplicate of an entry that already matched; reverse it"
            elif transposed:
                bucket, why = BUCKET_ERROR, ("Amount matches a bank line with two digits swapped "
                                             "(difference is a multiple of 9); correct this entry")
            elif t.id in state.ambiguous:
                bucket, why = None, "Several bank lines fit equally well; pick the right one"
            elif (period_end - t.txn_date).days <= cfg.timing_days:
                bucket, why = BUCKET_TIMING, "Recorded near period end and not yet at the bank; carry forward"
            elif sugg:
                bucket, why = None, "Possible match found; confirm or reject it"
            else:
                bucket, why = None, "No bank line found; investigate"

        out.append(ExceptionItem(t.id, bucket, why, sugg[:3]))
    return out


# ---------------------------------------------------------------------------
# Summary (feeds the per-run summary view)
# ---------------------------------------------------------------------------


def summarize(state: MatchState, exceptions: list[ExceptionItem]) -> dict:
    return summarize_parts(state.bank, state.ledger, state.matches, exceptions)


def summarize_parts(bank: list[Transaction], ledger: list[Transaction], matches: list[Match],
                    exceptions: list[ExceptionItem]) -> dict:
    """Counts behind the per-run summary view. Recomputed after every review decision."""
    by_id = {t.id: t for t in bank + ledger}
    nb, nl = len(bank), len(ledger)
    matched_bank = {i for m in matches for i in m.transaction_ids if i.startswith("bank:")}
    matched_ledger = {i for m in matches for i in m.transaction_ids if i.startswith("ledger:")}

    bank_by_layer: Counter = Counter()
    for m in matches:
        bank_by_layer[m.layer] += sum(1 for i in m.transaction_ids if i.startswith("bank:"))

    def unreconciled(source: str) -> float:
        return round(sum(abs(by_id[e.transaction_id].amount_cents) for e in exceptions
                         if e.transaction_id.startswith(source + ":")) / 100, 2)

    def coverage(txns: list[Transaction]) -> float:
        return round(sum(1 for t in txns if t.reference_id) / len(txns), 3) if txns else 0.0

    total_matched = len(matched_bank) + len(matched_ledger)
    return {
        "transactions": {"bank": nb, "ledger": nl},
        "matched": {"bank": len(matched_bank), "ledger": len(matched_ledger)},
        "match_rate": round(total_matched / (nb + nl), 3) if nb + nl else 0.0,
        "matches_by_layer": dict(Counter(m.layer for m in matches)),
        "matched_bank_share_by_layer": {k: round(v / len(matched_bank), 3)
                                        for k, v in bank_by_layer.items()} if matched_bank else {},
        "exceptions": {
            "bank": sum(1 for e in exceptions if e.transaction_id.startswith("bank:")),
            "ledger": sum(1 for e in exceptions if e.transaction_id.startswith("ledger:")),
            "by_suggested_bucket": dict(Counter(e.suggested_bucket or "needs_decision"
                                                for e in exceptions)),
            "by_bucket": dict(Counter(e.bucket or "needs_decision" for e in exceptions)),
            "with_suggestions": sum(1 for e in exceptions if e.suggestions),
            "resolved": sum(1 for e in exceptions if e.resolved),
            "open": sum(1 for e in exceptions if not e.resolved),
        },
        "unreconciled_amount": {"bank": unreconciled("bank"), "ledger": unreconciled("ledger")},
        "reference_coverage": {"bank": coverage(bank), "ledger": coverage(ledger)},
    }


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------


def _rows(ids: list[str], source: str) -> str:
    return ";".join(i.split(":")[1] for i in ids if i.startswith(source + ":"))


def write_results(result: RunResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_id = {t.id: t for t in result.transactions}

    with open(out / "transactions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "source_row", "txn_date", "amount", "description",
                    "clean_description", "reference_id", "external_id"])
        for t in result.transactions:
            w.writerow([t.id, t.source, t.source_row, t.txn_date.isoformat(), f"{t.amount:.2f}",
                        t.description, t.clean_description, t.reference_id, t.external_id])

    with open(out / "matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match_id", "layer", "match_type", "confidence", "bank_rows", "ledger_rows", "reason"])
        for m in result.matches:
            w.writerow([m.match_id, m.layer, m.match_type, m.confidence,
                        _rows(m.transaction_ids, "bank"), _rows(m.transaction_ids, "ledger"), m.reason])

    with open(out / "exceptions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transaction_id", "date", "amount", "description", "suggested_bucket",
                    "reason", "top_suggestion", "suggestion_confidence"])
        for e in result.exceptions:
            t = by_id[e.transaction_id]
            top = e.suggestions[0] if e.suggestions else None
            w.writerow([e.transaction_id, t.txn_date.isoformat(), f"{t.amount:.2f}", t.description,
                        e.suggested_bucket or "", e.reason,
                        "+".join(top.bank_ids + top.ledger_ids) if top else "",
                        top.confidence if top else ""])

    (out / "summary.json").write_text(json.dumps(result.summary, indent=2))
    (out / "config.json").write_text(json.dumps(result.config.to_dict(), indent=2))
    (out / "suggestions.json").write_text(json.dumps([asdict(s) for s in result.suggestions], indent=2))
