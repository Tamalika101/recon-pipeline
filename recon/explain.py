"""Plain-language description of the pipeline, for the app's "How it works" view.

Everything here is generated from the actual MatchConfig and run results, so
the explanation can't drift from what the code really did.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine import RunResult
from .models import MatchConfig

REPO = "https://github.com/Tamalika101/recon-pipeline"
SRC = f"{REPO}/blob/main/recon"


@dataclass
class Stage:
    key: str
    title: str
    does: str
    rule: str
    never: str
    source: str


def stages(cfg: MatchConfig) -> list[Stage]:
    w = cfg.weights
    return [
        Stage("normalize", "1. Read and normalize",
              "Turns both CSV files into one common format: a date, a signed amount (+ money in, "
              "- money out), a cleaned description and a reference ID if there is one.",
              "Uses the column mapping you confirmed. Debit/credit columns become signed amounts; "
              "rows that can't be read are reported with their row number instead of being skipped.",
              "Never changes your numbers. The original text of every line is kept alongside the cleaned version.",
              f"{SRC}/normalize.py"),
        Stage("reference_id", "2. Match by reference ID",
              "Pairs a bank line and a book line that carry the same reference (check number, "
              "transfer ID) and the same amount.",
              "Same reference + same amount = match, confidence 100. If two book lines share the "
              "reference, the closest date wins and the other is flagged as a likely duplicate.",
              "Never matches on a shared reference when the amounts differ; that becomes a suggestion instead.",
              f"{SRC}/layers.py"),
        Stage("exact", "3. Match by exact amount and date",
              "For lines without a shared reference, looks for the same amount posted close together.",
              f"Same amount, bank date 0 to {cfg.exact_window_days} days after the book date, and "
              f"exactly one candidate on each side = match, confidence 95.",
              "Never guesses. If two lines fit equally well (say two $500 payments a day apart), "
              "none is matched and both are shown to you.",
              f"{SRC}/layers.py"),
        Stage("fuzzy", "4. Close match (scored)",
              "Scores remaining pairs on how close the amounts, dates and descriptions are, so "
              "late or garbled bank lines can still be paired.",
              f"Score out of 100 = amount {w['amount']:.0%} + date {w['date']:.0%} + description "
              f"{w['description']:.0%}, searching up to {cfg.fuzzy_window_days} days. "
              f"{cfg.auto_accept:.0f}+ and at least {cfg.ambiguity_margin:.0f} points ahead of the "
              f"next best = match. {cfg.suggest:.0f}+ = suggested to you.",
              "Never auto-matches amounts that look like swapped digits, or anything an earlier "
              "step marked as ambiguous.",
              f"{SRC}/layers.py"),
        Stage("combination", "5. Match grouped amounts",
              "Finds one line that equals the sum of several lines on the other side, such as one "
              "deposit covering several invoices.",
              f"Groups of 2 to {cfg.max_group_size} lines within {cfg.combo_window_days} days that add "
              f"up to the cent = match, confidence 90, but only if no other group also adds up.",
              "Never picks between two groups that both add up; both are shown to you, smallest first.",
              f"{SRC}/layers.py"),
        Stage("exceptions", "6. Sort what's left",
              "Every unmatched line goes to Needs attention with a suggested reason and the best "
              "candidate matches.",
              f"Bank fees go to 'Missing from books'; amounts with swapped digits and duplicate "
              f"entries to 'Error in books'; book payments in the last {cfg.timing_days} days of the "
              f"period to 'Timing'; unknown bank charges to 'Needs investigation'.",
              "The reason is only a suggestion. When the system can't tell, it says so and leaves the decision to you.",
              f"{SRC}/engine.py"),
        Stage("review", "7. Your decisions",
              "You accept suggested matches or mark items resolved with a reason and a note.",
              "Accepted suggestions become matches labelled 'Confirmed by you', so it's always "
              "clear which matches a person made.",
              "Nothing you decide is saved anywhere once you close the tab.",
              f"{SRC}/review.py"),
        Stage("report", "8. Reconciliation report",
              "Shows the bank balance and book balance, each adjusted by the reconciling items, "
              "plus suggested journal entries.",
              "Payments not yet cleared and deposits in transit adjust the bank side; fees, "
              "unrecorded items and corrections adjust the book side. The two should agree.",
              "Never posts anything to your books. Journal entries are suggestions to review and post yourself.",
              f"{SRC}/report.py"),
    ]


def funnel(result: RunResult) -> list[dict]:
    """Lines entering, matched at, and left after each automatic layer."""
    total = len(result.transactions)
    matched_by_layer: dict[str, int] = {}
    for m in result.matches:
        matched_by_layer[m.layer] = matched_by_layer.get(m.layer, 0) + len(m.transaction_ids)
    rows, remaining = [], total
    for key in result.config.layers:
        got = matched_by_layer.get(key, 0)
        rows.append({"stage": key, "in": remaining, "matched": got, "left": remaining - got})
        remaining -= got
    manual = matched_by_layer.get("manual", 0)
    rows.append({"stage": "manual", "in": remaining, "matched": manual, "left": remaining - manual})
    return rows


def diagram_html(result: RunResult | None, cfg: MatchConfig) -> str:
    """The pipeline as a row of step cards that wraps on narrow screens, with counts after a run."""
    from html import escape
    counts = {r["stage"]: r for r in funnel(result)} if result else {}
    names = {"reference_id": "Reference ID", "exact": "Exact amount and date",
             "fuzzy": "Close match (scored)", "combination": "Grouped amounts"}

    steps = [("Your 2 CSV files", f"{len(result.transactions)} lines" if result else "bank + books", "file"),
             ("Normalize", "one common format", "auto")]
    for key in cfg.layers:
        steps.append((names.get(key, key),
                      f"{counts[key]['matched']} matched" if key in counts else "automatic", "auto"))
    open_items = sum(1 for e in result.exceptions if not e.resolved) if result else None
    manual = counts.get("manual", {}).get("matched", 0) if result else 0
    steps += [("Needs attention", f"{open_items} open" if result else "with suggestions", "person"),
              ("Your decisions", f"{manual} lines confirmed" if manual else "accept or resolve", "person"),
              ("Report", "balances + journal entries", "file")]

    colors = {"file": ("#ffffff", "#9aa5a1"), "auto": ("#eef5f2", "#7fae9f"), "person": ("#fdf3e6", "#d9a868")}
    cards = []
    for i, (title, sub, kind) in enumerate(steps):
        bg, border = colors[kind]
        cards.append(
            f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:8px 12px;'
            f'min-width:120px;text-align:center">'
            f'<div style="font-weight:600;font-size:14px">{escape(title)}</div>'
            f'<div style="font-size:12.5px;color:#555">{escape(sub)}</div></div>')
        if i < len(steps) - 1:
            cards.append('<div style="color:#7fae9f;font-size:18px">&rarr;</div>')
    legend = ('<div style="font-size:12px;color:#666;margin-top:8px">'
              '<span style="color:#7fae9f">&#9632;</span> automatic &nbsp; '
              '<span style="color:#d9a868">&#9632;</span> needs a person</div>')
    return ('<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;'
            'font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif">' + "".join(cards) + "</div>" + legend)
