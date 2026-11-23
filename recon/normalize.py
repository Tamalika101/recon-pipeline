"""Turn an uploaded CSV plus a saved column mapping into normalized transactions.

A mapping is JSON saved per source (e.g. per bank), for example:

    {
      "source": "bank",
      "date": {"column": "Posting Date", "format": "%m/%d/%Y"},
      "amount": {"debit": "Debit", "credit": "Credit"},
      "description": "Description",
      "reference": "Bank Ref"
    }

Amounts come either as one signed column ({"column": "Amount", "sign": 1})
or as separate debit and credit columns. On a bank statement a debit is money
out; in the books' cash account a debit is money in, so ledger mappings with
debit/credit columns set "debit_is_inflow": true. Signs are always normalized so that
+ means money into the bank account.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import Transaction
from .text import clean


@dataclass
class Issue:
    row: int | None      # None = file-level problem
    message: str


@dataclass
class NormalizeResult:
    transactions: list[Transaction] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def reference_coverage(self) -> float:
        if not self.transactions:
            return 0.0
        return sum(1 for t in self.transactions if t.reference_id) / len(self.transactions)


class NormalizationError(Exception):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        head = "; ".join(f"row {i.row}: {i.message}" if i.row else i.message for i in issues[:5])
        more = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
        super().__init__(head + more)


def load_mapping(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_amount(text: str) -> int | None:
    """'1,234.56', '$1234.56', '(12.00)', '-12' -> cents. Blank -> None."""
    s = (text or "").strip().replace(",", "").replace("$", "")
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        value = Decimal(s)
    except InvalidOperation:
        raise ValueError(f"not a number: {text!r}")
    cents = int((value * 100).quantize(Decimal("1")))
    return -cents if negative else cents


def required_columns(mapping: dict) -> list[str]:
    cols = [mapping["date"]["column"], mapping["description"]]
    amt = mapping["amount"]
    cols += [amt["column"]] if "column" in amt else [amt["debit"], amt["credit"]]
    return cols


def normalize(csv_text: str, mapping: dict) -> NormalizeResult:
    """Normalize one file. Collects every problem instead of stopping at the first."""
    result = NormalizeResult()
    source = mapping["source"]
    reader = csv.DictReader(io.StringIO(csv_text))
    header = reader.fieldnames or []

    missing = [c for c in required_columns(mapping) if c not in header]
    for opt in ("reference", "external_id"):
        if mapping.get(opt) and mapping[opt] not in header:
            missing.append(mapping[opt])
    if missing:
        result.issues.append(Issue(None, f"{source} file is missing column(s): {', '.join(missing)}"))
        return result

    date_col, date_fmt = mapping["date"]["column"], mapping["date"]["format"]
    amt = mapping["amount"]
    for i, row in enumerate(reader, start=1):
        try:
            d = datetime.strptime(row[date_col].strip(), date_fmt).date()
        except ValueError:
            result.issues.append(Issue(i, f"date {row[date_col]!r} doesn't match format {date_fmt}"))
            continue
        try:
            if "column" in amt:
                cents = parse_amount(row[amt["column"]])
                if cents is None:
                    raise ValueError("amount is blank")
                cents *= int(amt.get("sign", 1))
            else:
                debit = parse_amount(row[amt["debit"]]) or 0
                credit = parse_amount(row[amt["credit"]]) or 0
                if debit and credit:
                    raise ValueError("both debit and credit are filled")
                if not debit and not credit:
                    raise ValueError("debit and credit are both blank")
                # Bank statements: debit = money out. Books (cash account): debit = money in.
                cents = (abs(debit) - credit) if amt.get("debit_is_inflow") else (credit - abs(debit))
        except ValueError as e:
            result.issues.append(Issue(i, str(e)))
            continue

        desc = row[mapping["description"]].strip()
        result.transactions.append(Transaction(
            id=f"{source}:{i}",
            source=source,
            source_row=i,
            txn_date=d,
            amount_cents=cents,
            description=desc,
            clean_description=clean(desc),
            reference_id=(row.get(mapping.get("reference") or "", "") or "").strip(),
            external_id=(row.get(mapping.get("external_id") or "", "") or "").strip(),
        ))
    if not result.transactions and not result.issues:
        result.issues.append(Issue(None, f"{source} file has no data rows"))
    return result


def normalize_file(path: str | Path, mapping: dict) -> NormalizeResult:
    return normalize(Path(path).read_text(), mapping)
