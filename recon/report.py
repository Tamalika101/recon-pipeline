"""The reconciliation report: bank balance and book balance, each adjusted by the
reconciling items, plus suggested journal entries and CSV/HTML exports.

Standard bank reconciliation layout:

    Bank statement balance            Book (ledger) balance
    + deposits in transit             + / - items on the bank but not in the books
    - outstanding payments            + / - corrections of errors in the books
    = adjusted bank balance           = adjusted book balance

The two adjusted balances should be equal. Any gap left over is explained by
items nobody has decided on yet, or by a wrong starting balance.
"""
from __future__ import annotations

import csv
import html
import io
from dataclasses import dataclass, field
from datetime import date

from .engine import RunResult
from .models import BUCKET_ERROR, BUCKET_MISSING, BUCKET_PROBLEM, BUCKET_TIMING, Transaction

CASH = "Cash - checking"

LAYER_LABELS = {
    "reference_id": "Reference ID",
    "exact": "Exact amount and date",
    "fuzzy": "Close match (scored)",
    "combination": "Grouped amounts",
    "manual": "Confirmed by you",
}
BUCKET_LABELS = {
    BUCKET_TIMING: "Timing (will clear)",
    BUCKET_MISSING: "Missing from books",
    BUCKET_ERROR: "Error in books",
    BUCKET_PROBLEM: "Needs investigation",
    None: "Needs your decision",
}


@dataclass
class RecItem:
    side: str                 # "bank" | "book" | "unresolved"
    label: str
    amount_cents: int         # effect on that side's balance
    txn_ids: list[str]
    txn_date: date
    description: str
    offset_account: str = ""
    note: str = ""


@dataclass
class Reconciliation:
    bank_balance_cents: int
    book_balance_cents: int
    bank_items: list[RecItem] = field(default_factory=list)
    book_items: list[RecItem] = field(default_factory=list)
    unresolved: list[RecItem] = field(default_factory=list)

    @property
    def adjusted_bank(self) -> int:
        return self.bank_balance_cents + sum(i.amount_cents for i in self.bank_items)

    @property
    def adjusted_book(self) -> int:
        return self.book_balance_cents + sum(i.amount_cents for i in self.book_items)

    @property
    def difference(self) -> int:
        return self.adjusted_bank - self.adjusted_book

    @property
    def unresolved_net(self) -> int:
        """What the open items would contribute to the difference."""
        return sum(i.amount_cents for i in self.unresolved)

    @property
    def status(self) -> str:
        if self.difference == 0 and not self.unresolved:
            return "reconciled"
        if self.difference == 0:
            return "balanced_open_items"
        if self.unresolved and self.difference == self.unresolved_net:
            return "unresolved_items"
        return "unexplained"


def build_reconciliation(result: RunResult, bank_balance_cents: int,
                         book_balance_cents: int) -> Reconciliation:
    by_id: dict[str, Transaction] = {t.id: t for t in result.transactions}
    rec = Reconciliation(bank_balance_cents, book_balance_cents)
    open_ids = {e.transaction_id for e in result.exceptions}
    used: set[str] = set()

    # Transposition pairs first: one correction covers both the bank and the ledger line.
    for e in result.exceptions:
        t = by_id[e.transaction_id]
        if t.source != "bank" or e.bucket != BUCKET_ERROR or e.transaction_id in used:
            continue
        pair = next((s for s in e.suggestions if s.transposition
                     and all(i in open_ids and i not in used for i in s.ledger_ids)), None)
        if pair and len(pair.ledger_ids) == 1:
            led = by_id[pair.ledger_ids[0]]
            rec.book_items.append(RecItem(
                "book", "Correct amount entered with swapped digits",
                t.amount_cents - led.amount_cents, [t.id, led.id], led.txn_date,
                f"{led.description} (books {led.amount:.2f}, bank {t.amount:.2f})",
                offset_account=f"Account on entry {led.external_id or led.id}", note=e.note))
            used.update({t.id, led.id})

    for e in result.exceptions:
        if e.transaction_id in used:
            continue
        t = by_id[e.transaction_id]
        b = e.bucket
        item = None
        if t.source == "ledger" and b == BUCKET_TIMING:
            label = "Deposit in transit" if t.amount_cents > 0 else "Outstanding payment"
            item = RecItem("bank", label, t.amount_cents, [t.id], t.txn_date, t.description, note=e.note)
            rec.bank_items.append(item)
        elif t.source == "ledger" and b == BUCKET_ERROR:
            item = RecItem("book", "Reverse duplicate or wrong entry", -t.amount_cents, [t.id],
                           t.txn_date, t.description,
                           offset_account=f"Account on entry {t.external_id or t.id}", note=e.note)
            rec.book_items.append(item)
        elif t.source == "bank" and b == BUCKET_MISSING:
            if t.amount_cents < 0:
                label, acct = "Bank fee or charge not yet booked", "Bank fees"
            else:
                label, acct = "Deposit not yet booked", "Uncategorized income (confirm source)"
            item = RecItem("book", label, t.amount_cents, [t.id], t.txn_date, t.description,
                           offset_account=acct, note=e.note)
            rec.book_items.append(item)
        elif t.source == "bank" and b == BUCKET_PROBLEM:
            item = RecItem("book", "Unrecognized bank item, held in suspense", t.amount_cents, [t.id],
                           t.txn_date, t.description, offset_account="Suspense - under investigation",
                           note=e.note)
            rec.book_items.append(item)

        if item is None:
            # Undecided, or a combination that doesn't fit the layout. Its effect on the
            # difference: bank lines push it up, ledger lines push it down.
            effect = t.amount_cents if t.source == "bank" else -t.amount_cents
            rec.unresolved.append(RecItem("unresolved", BUCKET_LABELS.get(b, str(b)), effect,
                                          [t.id], t.txn_date, t.description, note=e.note))
    for lst in (rec.bank_items, rec.book_items, rec.unresolved):
        lst.sort(key=lambda i: (i.txn_date, i.txn_ids[0]))
    return rec


def journal_rows(rec: Reconciliation) -> list[dict]:
    """Suggested entries to post in the books, one per book-side item."""
    rows = []
    for n, item in enumerate(rec.book_items, start=1):
        amt = abs(item.amount_cents) / 100
        dr, cr = (CASH, item.offset_account) if item.amount_cents > 0 else (item.offset_account, CASH)
        memo = f"{item.label}: {item.description}"
        rows.append({"entry": n, "date": item.txn_date.isoformat(), "account": dr,
                     "debit": f"{amt:.2f}", "credit": "", "memo": memo, "source": " ".join(item.txn_ids)})
        rows.append({"entry": n, "date": item.txn_date.isoformat(), "account": cr,
                     "debit": "", "credit": f"{amt:.2f}", "memo": memo, "source": " ".join(item.txn_ids)})
    return rows


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def _csv(rows: list[dict], header: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def journal_csv(rec: Reconciliation) -> str:
    return _csv(journal_rows(rec), ["entry", "date", "account", "debit", "credit", "memo", "source"])


def matches_csv(result: RunResult) -> str:
    by_id = {t.id: t for t in result.transactions}
    rows = []
    for m in result.matches:
        for i in m.transaction_ids:
            t = by_id[i]
            rows.append({"match_id": m.match_id, "how": LAYER_LABELS.get(m.layer, m.layer),
                         "confidence": m.confidence, "source": t.source, "row": t.source_row,
                         "date": t.txn_date.isoformat(), "amount": f"{t.amount:.2f}",
                         "description": t.description, "reason": m.reason})
    return _csv(rows, ["match_id", "how", "confidence", "source", "row", "date", "amount",
                       "description", "reason"])


def exceptions_csv(result: RunResult) -> str:
    by_id = {t.id: t for t in result.transactions}
    rows = []
    for e in result.exceptions:
        t = by_id[e.transaction_id]
        rows.append({"source": t.source, "row": t.source_row, "date": t.txn_date.isoformat(),
                     "amount": f"{t.amount:.2f}", "description": t.description,
                     "reason": BUCKET_LABELS.get(e.bucket, e.bucket),
                     "status": "resolved" if e.resolved else "open",
                     "system_note": e.reason, "your_note": e.note})
    return _csv(rows, ["source", "row", "date", "amount", "description", "reason", "status",
                       "system_note", "your_note"])


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:,.2f}"


def render_html(result: RunResult, rec: Reconciliation, period_end: date, title: str = "") -> str:
    s = result.summary
    esc = html.escape

    def items_table(items: list[RecItem], empty: str) -> str:
        if not items:
            return f"<p class='muted'>{esc(empty)}</p>"
        body = "".join(
            f"<tr><td>{i.txn_date.isoformat()}</td><td>{esc(i.label)}</td><td>{esc(i.description)}</td>"
            f"<td class='num'>{money(i.amount_cents)}</td></tr>" for i in items)
        return ("<table><tr><th>Date</th><th>Item</th><th>Description</th><th class='num'>Amount</th>"
                f"</tr>{body}</table>")

    status_text = {
        "reconciled": "Reconciled. The adjusted balances agree and every item has a decision.",
        "balanced_open_items": (f"The adjusted balances agree, but {len(rec.unresolved)} item(s) still "
                                f"need a decision. They currently offset each other."),
        "unresolved_items": (f"Not yet reconciled. The difference of {money(rec.difference)} comes "
                             f"from {len(rec.unresolved)} item(s) still waiting for a decision."),
        "unexplained": (f"Not reconciled. A difference of {money(rec.difference)} is not explained "
                        f"by open items; check the starting balances or look for missing lines."),
    }[rec.status]

    jrows = journal_rows(rec)
    jbody = "".join(
        f"<tr><td>{r['entry']}</td><td>{r['date']}</td><td>{esc(r['account'])}</td>"
        f"<td class='num'>{r['debit']}</td><td class='num'>{r['credit']}</td><td>{esc(r['memo'])}</td></tr>"
        for r in jrows)
    journal = ("<table><tr><th>#</th><th>Date</th><th>Account</th><th class='num'>Debit</th>"
               f"<th class='num'>Credit</th><th>Memo</th></tr>{jbody}</table>") if jrows else \
        "<p class='muted'>No entries needed.</p>"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Bank reconciliation - {period_end.isoformat()}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:900px;margin:32px auto;color:#222;padding:0 16px}}
h1{{font-size:22px;margin-bottom:4px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #eee}}
.num{{text-align:right;white-space:nowrap}} .muted{{color:#777}} .status{{padding:10px 12px;border-radius:6px;background:#f3f3f3}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}} .total td{{font-weight:600;border-top:2px solid #333}}
</style></head><body>
<h1>Bank reconciliation{(' - ' + esc(title)) if title else ''}</h1>
<p class="muted">Period ending {period_end.strftime('%B %d, %Y')}</p>
<p class="status">{esc(status_text)}</p>
<div class="grid"><div>
<h2>Bank side</h2><table>
<tr><td>Bank statement balance</td><td class="num">{money(rec.bank_balance_cents)}</td></tr>
{''.join(f"<tr><td>{esc(i.label)}: {esc(i.description)}</td><td class='num'>{money(i.amount_cents)}</td></tr>" for i in rec.bank_items)}
<tr class="total"><td>Adjusted bank balance</td><td class="num">{money(rec.adjusted_bank)}</td></tr></table>
</div><div>
<h2>Book side</h2><table>
<tr><td>Book (ledger) balance</td><td class="num">{money(rec.book_balance_cents)}</td></tr>
{''.join(f"<tr><td>{esc(i.label)}: {esc(i.description)}</td><td class='num'>{money(i.amount_cents)}</td></tr>" for i in rec.book_items)}
<tr class="total"><td>Adjusted book balance</td><td class="num">{money(rec.adjusted_book)}</td></tr></table>
</div></div>
<h2>Items still open</h2>{items_table(rec.unresolved, 'None. Every item has a decision.')}
<h2>Suggested journal entries</h2>{journal}
<h2>Matching summary</h2>
<p>{s['matched']['bank']} of {s['transactions']['bank']} bank lines and {s['matched']['ledger']} of
{s['transactions']['ledger']} ledger lines matched ({s['match_rate']:.0%}).
By method: {esc(', '.join(f"{LAYER_LABELS.get(k, k)} {v}" for k, v in s['matches_by_layer'].items()))}.</p>
<p class="muted">Generated by the reconciliation prototype. Matches and suggestions should be reviewed before posting entries.</p>
</body></html>"""
