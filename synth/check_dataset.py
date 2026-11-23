"""Independent check: does each scenario behave the way truth.csv claims?"""
import csv, sys
from datetime import datetime
from collections import Counter

d = sys.argv[1] if len(sys.argv) > 1 else "data"
bank = list(csv.DictReader(open(f"{d}/bank.csv")))
led = list(csv.DictReader(open(f"{d}/ledger.csv")))
truth = list(csv.DictReader(open(f"{d}/truth.csv")))
B = {i+1: dict(date=datetime.strptime(r["Posting Date"], "%m/%d/%Y").date(),
               amt=round(float(r["Credit"] or 0) - float(r["Debit"] or 0), 2), ref=r["Bank Ref"])
     for i, r in enumerate(bank)}
L = {i+1: dict(date=datetime.fromisoformat(r["Date"]).date(), amt=float(r["Amount"]), ref=r["Reference"])
     for i, r in enumerate(led)}

def exact_candidates(b, window=5):
    return [k for k, l in L.items() if l["amt"] == b["amt"] and 0 <= (b["date"] - l["date"]).days <= window]

problems = Counter(); ok = Counter()
for t in truth:
    s = t["scenario"]; br = int(t["bank_row"]) if t["bank_row"] else None
    lrs = [int(x) for x in t["ledger_rows"].split(";") if x]
    b = B.get(br)
    if s == "shared_id":
        hits = [k for k, l in L.items() if l["ref"] and l["ref"] == b["ref"]]
        good = lrs[0] in hits  # duplicates may add a second hit, flagged separately
    elif s == "clean_no_id":
        good = exact_candidates(b) == lrs and not any(l["ref"] == b["ref"] and b["ref"] for l in L.values())
    elif s == "ambiguous_twins":
        good = len(exact_candidates(b)) >= 2
    elif s == "garbled":
        lag = (b["date"] - L[lrs[0]]["date"]).days
        good = not exact_candidates(b) and L[lrs[0]]["amt"] == b["amt"] and 5 < lag < 10
    elif s == "lump_deposit":
        good = round(sum(L[k]["amt"] for k in lrs), 2) == b["amt"] and not exact_candidates(b)
    elif s == "transposed":
        diff = round(abs(b["amt"] - L[lrs[0]]["amt"]) * 100)
        good = diff > 0 and diff % 9 == 0 and not exact_candidates(b)
    elif s in ("bank_fee_missing", "unknown_charge"):
        good = not lrs and not exact_candidates(b)
    elif s == "duplicate_entry":
        same = [k for k, l in L.items() if l["ref"] == L[lrs[0]]["ref"] and l["amt"] == L[lrs[0]]["amt"]]
        good = len(same) == 2
    elif s == "outstanding_check":
        good = br is None and L[lrs[0]]["ref"].startswith("CHK-")
    else:
        good = False
    (ok if good else problems)[s] += 1

print("OK:", dict(ok)); print("PROBLEMS:", dict(problems) or "none")
sys.exit(1 if problems else 0)
