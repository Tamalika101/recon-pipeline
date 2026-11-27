"""Score a run against a ground-truth file (truth.csv from the synthetic generator).

What "correct" means per scenario:
- matched scenarios: one automatic match holding exactly the true bank and ledger
  lines. Whether the expected layer made it is reported separately.
- ambiguous twins: NOT auto-matched, and the true pair is among the suggestions.
- exception scenarios: every line involved left unmatched, with the right suggested bucket.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .engine import RunResult

AUTO_LAYERS = {"reference_id", "exact", "fuzzy", "combination"}


def load_truth(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "bank": f"bank:{r['bank_row']}" if r["bank_row"] else None,
                "ledger": [f"ledger:{x}" for x in r["ledger_rows"].split(";") if x],
                "scenario": r["scenario"],
                "layer": r["expected_layer"],
                "outcome": r["expected_outcome"],
            })
    return rows


def evaluate(result: RunResult, truth_path: str | Path) -> dict:
    truth = load_truth(truth_path)
    match_of = {i: m for m in result.matches for i in m.transaction_ids}
    exc_of = {e.transaction_id: e for e in result.exceptions}

    pred_pairs = {(b, l) for m in result.matches
                  for b in m.transaction_ids if b.startswith("bank:")
                  for l in m.transaction_ids if l.startswith("ledger:")}
    true_pairs_all = {(t["bank"], l) for t in truth if t["outcome"] == "matched" for l in t["ledger"]}
    true_pairs_auto = {(t["bank"], l) for t in truth
                       if t["outcome"] == "matched" and t["layer"] in AUTO_LAYERS for l in t["ledger"]}

    per = defaultdict(lambda: {"total": 0, "correct": 0, "right_layer": 0, "top_suggestion": 0,
                               "misses": []})
    for t in truth:
        s = per[t["scenario"]]
        s["total"] += 1
        ids = ([t["bank"]] if t["bank"] else []) + t["ledger"]

        if t["outcome"] == "matched" and t["layer"] in AUTO_LAYERS:
            m = match_of.get(t["bank"])
            ok = m is not None and set(m.transaction_ids) == set(ids)
            if ok and m.layer == t["layer"]:
                s["right_layer"] += 1
            exc = exc_of.get(t["bank"])
            if not ok and exc and exc.suggestions and \
                    set(exc.suggestions[0].bank_ids + exc.suggestions[0].ledger_ids) == set(ids):
                s["top_suggestion"] += 1   # not automatic, but one click for the preparer
            detail = (f"matched by {m.layer} to {m.transaction_ids}" if m else "left unmatched")
        elif t["layer"] == "manual":
            exc = exc_of.get(t["bank"])
            ok = exc is not None and any(t["ledger"][0] in sg.ledger_ids for sg in exc.suggestions)
            detail = (f"auto-matched by {match_of[t['bank']].layer}" if t["bank"] in match_of
                      else "no suggestion with the true pair")
        else:
            bucket = t["outcome"].split(":", 1)[1]
            ok = all(i in exc_of and exc_of[i].suggested_bucket == bucket for i in ids)
            got = [(i, match_of[i].layer if i in match_of else exc_of[i].suggested_bucket) for i in ids]
            detail = f"expected {bucket}, got {got}"
        if ok:
            s["correct"] += 1
        elif len(s["misses"]) < 5:
            s["misses"].append({"truth": ids, "detail": detail})

    correct_pred = len(pred_pairs & true_pairs_all)
    exc_rows = [t for t in truth if t["outcome"].startswith("exception")]
    return {
        "pairs": {
            "predicted": len(pred_pairs),
            "precision": round(correct_pred / len(pred_pairs), 4) if pred_pairs else 1.0,
            "recall": round(len(pred_pairs & true_pairs_auto) / len(true_pairs_auto), 4)
            if true_pairs_auto else 1.0,
            # Share of ALL real matches made automatically, including look-alike pairs that no rule
            # can separate. Rules alone can't reach 100% here; model stages may.
            "recall_all": round(len(pred_pairs & true_pairs_all) / len(true_pairs_all), 4)
            if true_pairs_all else 1.0,
            "wrong": len(pred_pairs - true_pairs_all),
            "wrong_pairs": sorted(pred_pairs - true_pairs_all)[:10],
        },
        "exception_bucket_accuracy": round(
            sum(per[s]["correct"] for s in {t["scenario"] for t in exc_rows})
            / len(exc_rows), 4) if exc_rows else 1.0,
        "scenarios": {k: dict(v) for k, v in sorted(per.items())},
    }


def format_report(ev: dict) -> str:
    p = ev["pairs"]
    lines = [
        f"Match pairs  precision {p['precision']:.1%}  recall {p['recall']:.1%}  "
        f"({p['predicted']} predicted)",
        f"Exception buckets correct: {ev['exception_bucket_accuracy']:.1%}",
        "",
        f"{'scenario':<20}{'total':>6}{'correct':>9}{'layer ok':>10}{'top sugg.':>11}",
    ]
    for name, s in ev["scenarios"].items():
        layer = str(s["right_layer"]) if s["right_layer"] else "-"
        top = str(s["top_suggestion"]) if s["top_suggestion"] else "-"
        lines.append(f"{name:<20}{s['total']:>6}{s['correct']:>9}{layer:>10}{top:>11}")
    lines.append("(top sugg. = not auto-matched, but the right answer is the first suggestion)")
    if p["wrong_pairs"]:
        lines += ["", f"Wrong pairs (first 10): {p['wrong_pairs']}"]
    for name, s in ev["scenarios"].items():
        for miss in s["misses"]:
            lines.append(f"  miss [{name}] {miss['truth']}: {miss['detail']}")
    return "\n".join(lines)
