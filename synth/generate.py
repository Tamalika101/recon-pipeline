#!/usr/bin/env python3
"""
Synthetic bank + ledger data generator for the account reconciliation pipeline.

Creates one month of "true" business events for a fictional small business,
writes them to a bank export (bank.csv) and an accounting export (ledger.csv)
in deliberately different formats, injects realistic reconciliation problems,
and writes a ground-truth gold set (truth.csv) plus run metadata (meta.json).

Standard library only. Reproducible with --seed.

Usage:
    python generate.py --seed 42 --period 2026-11 --events 300 --out data/
    python generate.py --rate garbled=0.15 --rate bank_fee=0.03

Row numbers in truth.csv are 1-based data rows (the header is not counted),
so they line up with the pipeline's TRANSACTION.source_row.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
import random
import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RATES = {
    # Share of ordinary events assigned to each matching scenario.
    # Whatever is left over becomes "clean_no_id" (layer 2).
    "shared_id": 0.40,        # layer 1: same reference on both sides
    "garbled": 0.08,          # layer 3: mangled bank text, posted outside the date window
    "transposed": 0.02,       # exception/error: ledger amount has two digits swapped
    # Injected scenarios, as a fraction of the requested event count.
    "twins": 0.02,            # share of events used in ambiguous same-amount pairs
    "lump_deposit": 0.02,     # number of lump deposits (each covers 2-5 invoices)
    "duplicate": 0.015,       # ledger lines posted twice (integration retry)
    "outstanding": 0.02,      # ledger payments not yet cleared at the bank (timing)
    "bank_fee": 0.015,        # bank-only fees never booked (missing from books)
    "unknown_charge": 0.007,  # bank-only odd debits (real problem)
    # Data-quality knobs.
    "missing_ledger_ref": 0.15,  # non-shared ledger lines with a blank Reference
    "bank_trace_ref": 0.30,      # non-shared bank lines with a bank-internal trace ID (matches nothing)
}

DATE_WINDOW_DAYS = 5    # exact layer's default window; garbled lines post beyond it
FUZZY_WINDOW_DAYS = 10  # fuzzy layer searches wider, so it can still reach them

CUSTOMERS = ["Acme Corp", "Brightline Studio", "Cedar & Pine LLC", "Delta Logistics",
             "Evergreen Dental", "Foxglove Bakery", "Granite Peak Co", "Harbor Health",
             "Ironwood Fitness", "Juniper Events"]
VENDORS = ["Office Depot", "Uline", "Pacific Gas & Electric", "Comcast Business",
           "Sysco", "Grainger", "FedEx", "Staples"]
SUBSCRIPTIONS = ["Adobe", "Google Workspace", "Slack", "Zoom", "Dropbox",
                 "QuickBooks Online", "Mailchimp", "Canva"]
LANDLORD = "Bayview Properties"
PAYROLL = "Gusto"
PROCESSOR = "Square"

BANK_FEES = [("MONTHLY SVC FEE", 1500), ("INCOMING WIRE FEE", 1500),
             ("OVERDRAFT ITEM FEE", 3500), ("STOP PAYMENT FEE", 3000),
             ("PAPER STATEMENT FEE", 300)]
ODD_CHARGES = ["POS DEBIT XJ*9921 LAGOS NG", "ONLINE PMT ZZ-TRVL*AF2 INTL",
               "POS DEBIT GIFTCRD*7781 ONLINE", "RECURRING CHG MSFT*X0Q9 IE",
               "ACH DEBIT UNKNOWN ORIG 44102"]

BANK_DESC_MAX = 32

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LedgerLine:
    day: date
    cents: int              # signed, + = money into the bank account
    account: str
    memo: str
    reference: str
    entry_id: str = ""
    row: int = 0


@dataclass
class BankLine:
    day: date
    cents: int
    description: str
    bank_ref: str
    row: int = 0
    balance: int = 0


@dataclass
class TruthItem:
    bank: BankLine | None
    ledger: list[LedgerLine]
    scenario: str
    expected_layer: str     # reference_id | exact | fuzzy | combination | manual | none
    expected_outcome: str   # matched | exception:<bucket>


@dataclass
class World:
    rng: random.Random
    start: date
    end: date
    used_cents: set[int] = field(default_factory=set)
    used_refs: set[str] = field(default_factory=set)
    bank: list[BankLine] = field(default_factory=list)
    ledger: list[LedgerLine] = field(default_factory=list)
    truth: list[TruthItem] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    def unique_cents(self, lo: float, hi: float, sign: int) -> int:
        """Random amount in dollars [lo, hi], unique across the whole dataset.

        Uniqueness keeps accidental collisions from creating ambiguity that the
        truth file doesn't know about. Ambiguity is only ever injected on purpose.
        """
        while True:
            c = self.rng.randint(int(lo * 100), int(hi * 100))
            if c not in self.used_cents:
                self.used_cents.add(c)
                return sign * c

    def claim_cents(self, c: int) -> bool:
        if abs(c) in self.used_cents:
            return False
        self.used_cents.add(abs(c))
        return True

    def ref(self, prefix: str) -> str:
        while True:
            r = prefix + "".join(self.rng.choices(string.ascii_uppercase + string.digits, k=6))
            if r not in self.used_refs:
                self.used_refs.add(r)
                return r

    def seq(self, name: str, startval: int) -> int:
        self.counters[name] = self.counters.get(name, startval) + 1
        return self.counters[name]

    def day_in(self, first: int = 0, last_margin: int = 0) -> date:
        span = (self.end - self.start).days - last_margin
        return self.start + timedelta(days=self.rng.randint(first, max(first, span)))


# ---------------------------------------------------------------------------
# Event templates
# ---------------------------------------------------------------------------


def bank_text(s: str) -> str:
    s = s.upper().replace("&", "").replace(",", "").replace(".", "")
    return " ".join(s.split())[:BANK_DESC_MAX]


def garble(desc: str, rng: random.Random) -> str:
    """Compress/mangle bank text the way processors and banks often do."""
    words = desc.split()
    out = []
    for w in words:
        if len(w) <= 3:
            out.append(w)
        else:
            out.append(w[0] + "".join(ch for ch in w[1:] if ch not in "AEIOU"))
    joined = rng.choice([" ", ""]).join(out)
    code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
    return (joined[: rng.randint(10, 15)] + "*" + code)[:BANK_DESC_MAX]


def make_event(w: World, kind: str, day: date | None = None, cents: int | None = None,
               counterparty: str | None = None) -> dict:
    """Return the true facts of an event plus how each side describes it."""
    rng = w.rng
    if kind == "customer_payment":
        cp = counterparty or rng.choice(CUSTOMERS)
        cents = cents if cents is not None else w.unique_cents(300, 4000, +1)
        inv = w.seq("inv", 1040)
        return dict(kind=kind, cp=cp, cents=cents, day=day, account="Accounts Receivable",
                    memo=f"Payment received - {cp} INV-{inv}", ledger_ref=f"INV-{inv}",
                    bank_desc=bank_text(rng.choice([f"ACH CREDIT {cp} PAYMENT",
                                                    f"WIRE IN {cp}", "MOBILE DEPOSIT"])),
                    ref_prefix=rng.choice(["ACH-", "WIRE-"]))
    if kind == "vendor_bill":
        cp = counterparty or rng.choice(VENDORS)
        cents = cents if cents is not None else w.unique_cents(150, 5000, -1)
        bill = w.seq("bill", 550)
        return dict(kind=kind, cp=cp, cents=cents, day=day, account="Accounts Payable",
                    memo=f"Bill payment - {cp} BILL-{bill}", ledger_ref=f"BILL-{bill}",
                    bank_desc=bank_text(f"ACH DEBIT {cp}"), ref_prefix="ACH-")
    if kind == "subscription":
        cp = counterparty or rng.choice(SUBSCRIPTIONS)
        cents = cents if cents is not None else w.unique_cents(9, 400, -1)
        code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
        return dict(kind=kind, cp=cp, cents=cents, day=day, account="Software Subscriptions",
                    memo=f"{cp} subscription", ledger_ref="",
                    bank_desc=bank_text(f"POS DEBIT {cp}*{code} {cp.replace(' ', '')}COM"),
                    ref_prefix="POS-")
    if kind == "card_settlement":
        cents = cents if cents is not None else w.unique_cents(150, 1800, +1)
        return dict(kind=kind, cp=PROCESSOR, cents=cents, day=day, account="Merchant Settlements",
                    memo=f"{PROCESSOR} settlement", ledger_ref="",
                    bank_desc=bank_text("SQUARE INC DEP SETTLEMENT"), ref_prefix="STL-")
    if kind == "payroll":
        return dict(kind=kind, cp=PAYROLL, cents=cents, day=day, account="Payroll Expense",
                    memo=f"Payroll net pay {day.isoformat()}", ledger_ref="",
                    bank_desc=bank_text("GUSTO PAYROLL NET PAY"), ref_prefix="PR-")
    if kind == "rent":
        return dict(kind=kind, cp=LANDLORD, cents=cents, day=day, account="Rent Expense",
                    memo=f"Rent - {LANDLORD} {day.strftime('%b %Y')}", ledger_ref="",
                    bank_desc=bank_text(f"ACH DEBIT {LANDLORD} RENT"), ref_prefix="ACH-")
    if kind == "check_payment":
        cp = counterparty or rng.choice(VENDORS)
        cents = cents if cents is not None else w.unique_cents(150, 4000, -1)
        chk = w.seq("check", 1100)
        return dict(kind=kind, cp=cp, cents=cents, day=day, account="Accounts Payable",
                    memo=f"Check {chk} - {cp}", ledger_ref=f"CHK-{chk}",
                    bank_desc=bank_text(f"CHECK {chk}"), ref_prefix="CHK-")
    raise ValueError(kind)


KIND_WEIGHTS = [("customer_payment", 0.30), ("vendor_bill", 0.25),
                ("subscription", 0.20), ("card_settlement", 0.25)]


# ---------------------------------------------------------------------------
# Writers for each side
# ---------------------------------------------------------------------------


def add_ledger(w: World, ev: dict, day: date, cents: int, reference: str) -> LedgerLine:
    line = LedgerLine(day=day, cents=cents, account=ev["account"], memo=ev["memo"],
                      reference=reference)
    w.ledger.append(line)
    return line


def add_bank(w: World, day: date, cents: int, desc: str, bank_ref: str) -> BankLine:
    line = BankLine(day=day, cents=cents, description=desc, bank_ref=bank_ref)
    w.bank.append(line)
    return line


def non_shared_refs(w: World, ev: dict, rates: dict) -> tuple[str, str]:
    """Ledger keeps its own document number (or nothing); bank maybe has a trace ID."""
    ledger_ref = "" if w.rng.random() < rates["missing_ledger_ref"] else ev["ledger_ref"]
    bank_ref = (w.ref("TRC") if w.rng.random() < rates["bank_trace_ref"] else "")
    return ledger_ref, bank_ref


def transpose(cents: int, rng: random.Random) -> int | None:
    digits = list(str(abs(cents)))
    # Skip swaps that would move a zero to the front (that changes the digit count).
    spots = [i for i in range(len(digits) - 1)
             if digits[i] != digits[i + 1] and not (i == 0 and digits[1] == "0")]
    if not spots:
        return None
    i = rng.choice(spots)
    digits[i], digits[i + 1] = digits[i + 1], digits[i]
    swapped = int("".join(digits))
    return swapped if cents > 0 else -swapped


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def emit_ordinary(w: World, ev: dict, scenario: str, rates: dict) -> None:
    rng = w.rng
    if scenario == "shared_id":
        lag = rng.randint(0, 3)
        day = ev["day"] or w.day_in(last_margin=lag)
        lag = min(lag, (w.end - day).days)  # keep the bank posting inside the period
        shared = w.ref(ev["ref_prefix"])
        L = add_ledger(w, ev, day, ev["cents"], shared)
        B = add_bank(w, day + timedelta(days=lag), ev["cents"], ev["bank_desc"], shared)
        w.truth.append(TruthItem(B, [L], scenario, "reference_id", "matched"))
        return

    if scenario == "garbled":
        # Just past the exact layer's window, inside the fuzzy layer's wider one.
        lag = rng.randint(DATE_WINDOW_DAYS + 1, FUZZY_WINDOW_DAYS - 1)
        day = ev["day"] or w.day_in(last_margin=lag)
        lref, bref = non_shared_refs(w, ev, rates)
        L = add_ledger(w, ev, day, ev["cents"], lref)
        B = add_bank(w, day + timedelta(days=lag), ev["cents"], garble(ev["bank_desc"], rng), bref)
        w.truth.append(TruthItem(B, [L], scenario, "fuzzy", "matched"))
        return

    if scenario == "transposed":
        bad = transpose(ev["cents"], rng)
        if bad is None or not w.claim_cents(bad):
            return emit_ordinary(w, ev, "clean_no_id", rates)
        lag = rng.randint(1, DATE_WINDOW_DAYS)
        day = ev["day"] or w.day_in(last_margin=lag)
        lref, bref = non_shared_refs(w, ev, rates)
        L = add_ledger(w, ev, day, bad, lref)
        B = add_bank(w, day + timedelta(days=lag), ev["cents"], ev["bank_desc"], bref)
        w.truth.append(TruthItem(B, [L], scenario, "none", "exception:error"))
        return

    # clean_no_id
    lag = rng.randint(1, DATE_WINDOW_DAYS)
    day = ev["day"] or w.day_in(last_margin=lag)
    lref, bref = non_shared_refs(w, ev, rates)
    L = add_ledger(w, ev, day, ev["cents"], lref)
    B = add_bank(w, day + timedelta(days=lag), ev["cents"], ev["bank_desc"], bref)
    w.truth.append(TruthItem(B, [L], "clean_no_id", "exact", "matched"))


def emit_twins(w: World, rates: dict) -> int:
    """Two same-amount payments from one counterparty a day apart, no IDs anywhere.

    Both bank lines fall inside the date window of both ledger lines, so no
    automated layer can tell them apart. The right outcome is human review.
    """
    rng = w.rng
    kind = rng.choice(["vendor_bill", "customer_payment"])
    sign = -1 if kind == "vendor_bill" else +1
    cents = w.unique_cents(200, 3000, sign)
    cp = rng.choice(VENDORS if sign < 0 else CUSTOMERS)
    d0 = w.day_in(last_margin=5)
    for offset in (0, 1):
        ev = make_event(w, kind, day=d0 + timedelta(days=offset), cents=cents, counterparty=cp)
        L = add_ledger(w, ev, ev["day"], cents, "")
        B = add_bank(w, ev["day"] + timedelta(days=2), cents, ev["bank_desc"], "")
        w.truth.append(TruthItem(B, [L], "ambiguous_twins", "manual", "matched"))
    return 2


def emit_lump(w: World) -> int:
    """Several customer invoices deposited together as one bank credit."""
    rng = w.rng
    k = rng.randint(2, 5)
    while True:
        parts = [rng.randint(30000, 300000) for _ in range(k)]  # $300-$3,000 each
        total = sum(parts)
        if all(p not in w.used_cents for p in parts) and total not in w.used_cents \
                and len(set(parts)) == k:
            break
    for p in parts:
        w.used_cents.add(p)
    w.used_cents.add(total)
    d0 = w.day_in(last_margin=6)
    lines = []
    for p in parts:
        ev = make_event(w, "customer_payment", day=d0 + timedelta(days=rng.randint(0, 1)), cents=p)
        lines.append(add_ledger(w, ev, ev["day"], p, ev["ledger_ref"]))
    B = add_bank(w, d0 + timedelta(days=rng.randint(3, 5)), total,
                 bank_text(f"DEPOSIT {k} ITEMS"), "")
    w.truth.append(TruthItem(B, lines, "lump_deposit", "combination", "matched"))
    return k


def emit_duplicate(w: World) -> bool:
    """Re-post a shared-ID ledger line under a new entry (integration retry)."""
    candidates = [t for t in w.truth if t.scenario == "shared_id"
                  and not getattr(t, "_duplicated", False)]
    if not candidates:
        return False
    t = w.rng.choice(candidates)
    t._duplicated = True  # type: ignore[attr-defined]
    orig = t.ledger[0]
    dup = LedgerLine(day=orig.day, cents=orig.cents, account=orig.account,
                     memo=orig.memo, reference=orig.reference)
    w.ledger.append(dup)
    w.truth.append(TruthItem(None, [dup], "duplicate_entry", "none", "exception:error"))
    return True


def emit_outstanding(w: World) -> None:
    """Check written late in the month that hasn't cleared the bank yet."""
    day = w.end - timedelta(days=w.rng.randint(0, 4))
    ev = make_event(w, "check_payment", day=day)
    L = add_ledger(w, ev, day, ev["cents"], ev["ledger_ref"])
    w.truth.append(TruthItem(None, [L], "outstanding_check", "none", "exception:timing"))


def emit_bank_fee(w: World) -> None:
    desc, cents = w.rng.choice(BANK_FEES)
    # Fees reuse standard amounts, so they may repeat. That is fine: they never
    # have a ledger counterpart, and no ledger line ever uses these amounts.
    w.used_cents.add(cents)
    B = add_bank(w, w.day_in(), -cents, desc, "")
    w.truth.append(TruthItem(B, [], "bank_fee_missing", "none", "exception:missing_from_books"))


def emit_unknown(w: World) -> None:
    cents = w.unique_cents(40, 900, -1)
    B = add_bank(w, w.day_in(), cents, w.rng.choice(ODD_CHARGES)[:BANK_DESC_MAX], "")
    w.truth.append(TruthItem(B, [], "unknown_charge", "none", "exception:real_problem"))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def fixed_schedule(w: World) -> list[dict]:
    """Rent on the 1st, payroll mid-month and on the last business day."""
    events = []
    events.append(make_event(w, "rent", day=w.start, cents=w.unique_cents(3200, 4800, -1)))
    for d in (w.start.replace(day=15), last_business_day(w.end)):
        events.append(make_event(w, "payroll", day=d, cents=w.unique_cents(9000, 16000, -1)))
    return events


def last_business_day(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def generate(seed: int, period: str, n_events: int, rates: dict) -> World:
    year, month = (int(x) for x in period.split("-"))
    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    w = World(rng=random.Random(seed), start=start, end=end)
    rng = w.rng

    n = lambda key: max(1, round(n_events * rates[key])) if rates[key] > 0 else 0  # noqa: E731

    injected = 0
    for _ in range(max(1, round(n_events * rates["twins"] / 2)) if rates["twins"] > 0 else 0):
        injected += emit_twins(w, rates)
    for _ in range(n("lump_deposit")):
        injected += emit_lump(w)
    for _ in range(n("outstanding")):
        emit_outstanding(w); injected += 1
    for _ in range(n("bank_fee")):
        emit_bank_fee(w); injected += 1
    for _ in range(n("unknown_charge")):
        emit_unknown(w); injected += 1

    # Fixed events always reconcile by reference (payroll provider and landlord send IDs).
    fixed = fixed_schedule(w)
    for ev in fixed:
        emit_ordinary(w, ev, "shared_id", rates)

    n_ordinary = max(0, n_events - injected - len(fixed))
    kinds, weights = zip(*KIND_WEIGHTS)
    p_shared, p_garbled, p_trans = rates["shared_id"], rates["garbled"], rates["transposed"]
    for _ in range(n_ordinary):
        ev = make_event(w, rng.choices(kinds, weights)[0])
        r = rng.random()
        if r < p_shared:
            scenario = "shared_id"
        elif r < p_shared + p_garbled:
            scenario = "garbled"
        elif r < p_shared + p_garbled + p_trans:
            scenario = "transposed"
        else:
            scenario = "clean_no_id"
        emit_ordinary(w, ev, scenario, rates)

    # Duplicates last, so they can copy any shared-ID line.
    for _ in range(n("duplicate")):
        emit_duplicate(w)
    return w


def finalize(w: World, opening: int) -> dict:
    # Ledger: sort by date, keep insertion order on ties, then assign entry IDs and rows.
    for i, line in enumerate(sorted(w.ledger, key=lambda l: l.day)):  # stable on ties
        line.row = i + 1
        line.entry_id = f"JE-{10001 + i}"
    w.ledger.sort(key=lambda l: l.row)

    # Bank: sort by posting date (credits before debits on a day, as many banks do).
    w.bank.sort(key=lambda b: (b.day, b.cents < 0))
    bal = opening
    for i, b in enumerate(w.bank):
        b.row = i + 1
        bal += b.cents
        b.balance = bal

    bank_close = bal
    ledger_close = opening + sum(l.cents for l in w.ledger)

    # Reconciling identity: every difference between the two closing balances
    # must be explained by an injected exception. Prove it before writing files.
    bank_only = sum(t.bank.cents for t in w.truth if t.bank and not t.ledger)
    ledger_only = sum(l.cents for t in w.truth if t.bank is None for l in t.ledger)
    transposition = sum(t.bank.cents - t.ledger[0].cents for t in w.truth
                        if t.scenario == "transposed")
    assert bank_close - ledger_close == bank_only - ledger_only + transposition, \
        "reconciling identity failed"

    # Every line appears in exactly one truth row.
    seen_bank = [t.bank.row for t in w.truth if t.bank]
    seen_ledger = [l.row for t in w.truth for l in t.ledger]
    assert sorted(seen_bank) == list(range(1, len(w.bank) + 1)), "bank rows not covered once"
    assert sorted(seen_ledger) == list(range(1, len(w.ledger) + 1)), "ledger rows not covered once"

    return dict(bank_close=bank_close, ledger_close=ledger_close, bank_only=bank_only,
                ledger_only=ledger_only, transposition=transposition)


def money(c: int) -> str:
    return f"{c / 100:.2f}"


def write_outputs(w: World, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "bank.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Posting Date", "Description", "Debit", "Credit", "Running Balance", "Bank Ref"])
        for b in w.bank:
            wr.writerow([b.day.strftime("%m/%d/%Y"), b.description,
                         money(-b.cents) if b.cents < 0 else "",
                         money(b.cents) if b.cents > 0 else "",
                         money(b.balance), b.bank_ref])

    with open(out / "ledger.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Date", "Entry ID", "Account", "Memo", "Amount", "Reference"])
        for l in w.ledger:
            wr.writerow([l.day.isoformat(), l.entry_id, l.account, l.memo, money(l.cents), l.reference])

    truth_rows = sorted(w.truth, key=lambda t: (t.bank.row if t.bank else 10**9,
                                                min((l.row for l in t.ledger), default=0)))
    with open(out / "truth.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bank_row", "ledger_rows", "scenario", "expected_layer", "expected_outcome"])
        for t in truth_rows:
            wr.writerow([t.bank.row if t.bank else "",
                         ";".join(str(l.row) for l in sorted(t.ledger, key=lambda l: l.row)),
                         t.scenario, t.expected_layer, t.expected_outcome])

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--period", default="2026-11", help="YYYY-MM")
    ap.add_argument("--events", type=int, default=300, help="approximate true events")
    ap.add_argument("--out", default="data")
    ap.add_argument("--rate", action="append", default=[], metavar="NAME=VALUE",
                    help=f"override a rate; names: {', '.join(DEFAULT_RATES)}")
    args = ap.parse_args()

    rates = dict(DEFAULT_RATES)
    for item in args.rate:
        k, v = item.split("=", 1)
        if k not in rates:
            ap.error(f"unknown rate {k!r}")
        rates[k] = float(v)
    if rates["shared_id"] + rates["garbled"] + rates["transposed"] > 1:
        ap.error("shared_id + garbled + transposed must be <= 1")

    w = generate(args.seed, args.period, args.events, rates)
    opening = random.Random(args.seed + 1).randint(2_500_000, 6_000_000)
    bal = finalize(w, opening)

    scen_counts: dict[str, int] = {}
    for t in w.truth:
        scen_counts[t.scenario] = scen_counts.get(t.scenario, 0) + 1
    layer_counts: dict[str, int] = {}
    for t in w.truth:
        layer_counts[t.expected_layer] = layer_counts.get(t.expected_layer, 0) + 1

    meta = {
        "seed": args.seed,
        "period": {"start": w.start.isoformat(), "end": w.end.isoformat()},
        "events_requested": args.events,
        "rows": {"bank": len(w.bank), "ledger": len(w.ledger), "truth": len(w.truth)},
        "balances": {
            "opening": money(opening),
            "bank_closing": money(bal["bank_close"]),
            "ledger_closing": money(bal["ledger_close"]),
            "difference": money(bal["bank_close"] - bal["ledger_close"]),
            "explained_by": {
                "bank_only_items": money(bal["bank_only"]),
                "ledger_only_items_reversed": money(-bal["ledger_only"]),
                "transposition_errors": money(bal["transposition"]),
            },
        },
        "id_coverage": {
            "bank_ref_present": round(sum(1 for b in w.bank if b.bank_ref) / len(w.bank), 3),
            "ledger_ref_present": round(sum(1 for l in w.ledger if l.reference) / len(w.ledger), 3),
            "shared_ref_lines": scen_counts.get("shared_id", 0),
        },
        "date_window_days": {"exact": DATE_WINDOW_DAYS, "fuzzy": FUZZY_WINDOW_DAYS},
        "scenario_counts": dict(sorted(scen_counts.items())),
        "expected_layer_counts": dict(sorted(layer_counts.items())),
        "rates": rates,
    }
    write_outputs(w, Path(args.out), meta)

    print(f"Wrote {args.out}/: {len(w.bank)} bank rows, {len(w.ledger)} ledger rows, "
          f"{len(w.truth)} truth rows")
    print(f"Bank closing {money(bal['bank_close'])}  Ledger closing {money(bal['ledger_close'])}  "
          f"Difference {money(bal['bank_close'] - bal['ledger_close'])} (fully explained)")
    for k, v in meta["scenario_counts"].items():
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    main()
