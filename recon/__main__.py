"""Command line:

    python -m recon run --bank data/bank.csv --ledger data/ledger.csv --out results/
    python -m recon run ... --truth data/truth.csv      # also score against ground truth
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .engine import run_matching, write_results
from .evaluate import evaluate, format_report
from .models import MatchConfig
from .normalize import NormalizationError, load_mapping, normalize_file

HERE = Path(__file__).resolve().parent.parent


def cmd_run(args: argparse.Namespace) -> int:
    bank_res = normalize_file(args.bank, load_mapping(args.bank_mapping))
    ledger_res = normalize_file(args.ledger, load_mapping(args.ledger_mapping))
    issues = bank_res.issues + ledger_res.issues
    if issues:
        print(f"Validation failed: {NormalizationError(issues)}", file=sys.stderr)
        return 2

    cfg = MatchConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else MatchConfig()
    period_end = date.fromisoformat(args.period_end) if args.period_end else None
    result = run_matching(bank_res.transactions, ledger_res.transactions, cfg, period_end)
    write_results(result, Path(args.out))

    s = result.summary
    print(f"Matched {s['matched']['bank']}/{s['transactions']['bank']} bank and "
          f"{s['matched']['ledger']}/{s['transactions']['ledger']} ledger lines "
          f"({s['match_rate']:.1%}). By layer: {s['matches_by_layer']}")
    print(f"Exceptions: {s['exceptions']['by_suggested_bucket']}. "
          f"Unreconciled: bank {s['unreconciled_amount']['bank']:.2f}, "
          f"ledger {s['unreconciled_amount']['ledger']:.2f}")
    print(f"Results written to {args.out}/")

    if args.truth:
        ev = evaluate(result, args.truth)
        Path(args.out, "evaluation.json").write_text(json.dumps(ev, indent=2, default=str))
        print("\n" + format_report(ev))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="recon", description="Bank-to-ledger reconciliation engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="normalize, match and classify one reconciliation run")
    r.add_argument("--bank", required=True)
    r.add_argument("--ledger", required=True)
    r.add_argument("--bank-mapping", default=str(HERE / "mappings/bank_default.json"))
    r.add_argument("--ledger-mapping", default=str(HERE / "mappings/ledger_default.json"))
    r.add_argument("--config", help="matching config JSON (defaults otherwise)")
    r.add_argument("--period-end", help="YYYY-MM-DD; defaults to the latest transaction date")
    r.add_argument("--out", default="results")
    r.add_argument("--truth", help="truth.csv to score the run against")
    args = ap.parse_args()
    sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
