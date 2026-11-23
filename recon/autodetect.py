"""Guess a column mapping from a CSV's header and first rows, so users only
confirm or correct it instead of filling everything in."""
from __future__ import annotations

import csv
import io
from datetime import datetime

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%d/%m/%y",
                "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%m-%d-%Y"]


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_head(csv_text: str, n: int = 50) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for i, row in enumerate(reader):
        if i >= n:
            break
        rows.append(row)
    return list(reader.fieldnames or []), rows


def guess_date_format(values: list[str]) -> str | None:
    sample = [v.strip() for v in values if v and v.strip()][:30]
    if not sample:
        return None
    for fmt in DATE_FORMATS:
        try:
            for v in sample:
                datetime.strptime(v, fmt)
            return fmt
        except ValueError:
            continue
    return None


def _find(header: list[str], *keys: str, exclude: tuple[str, ...] = ()) -> str | None:
    for key in keys:
        for h in header:
            low = h.lower()
            if key in low and not any(x in low for x in exclude):
                return h
    return None


def guess_mapping(csv_text: str, source: str) -> dict:
    header, rows = read_head(csv_text)
    date_col = _find(header, "posting date", "post date", "transaction date", "date") or (header[0] if header else "")
    fmt = guess_date_format([r.get(date_col, "") for r in rows]) or DATE_FORMATS[0]

    debit = _find(header, "debit", "withdrawal", "money out", "paid out")
    credit = _find(header, "credit", "deposit", "money in", "paid in", exclude=("card",))
    amount = _find(header, "amount", "value", exclude=("balance",))
    if debit and credit and debit != credit:
        amt = {"debit": debit, "credit": credit, "debit_is_inflow": source == "ledger"}
    else:
        amt = {"column": amount or (header[1] if len(header) > 1 else ""), "sign": 1}

    desc = _find(header, "description", "memo", "narrative", "details", "payee", "name") or ""
    ref = _find(header, "ref", "transaction id", "check", "cheque", exclude=("entry",))
    ext = _find(header, "entry id", "entry no", "journal") if source == "ledger" else None
    bal = _find(header, "balance") if source == "bank" else None
    return {"source": source, "date": {"column": date_col, "format": fmt}, "amount": amt,
            "description": desc, "reference": ref, "external_id": ext, "balance": bal}
