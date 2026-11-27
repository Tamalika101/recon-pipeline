#!/usr/bin/env python3
"""Benchmark pipeline configurations on synthetic datasets with known answers.

    python benchmarks/run_benchmark.py \\
        --configs benchmarks/configs/rules_only.json benchmarks/configs/rules_decider_llm_local.json \\
        --profiles realistic hard --events 10000 --seeds 1 2 3

For every config x profile x seed it generates (or reuses) a dataset, runs the funnel,
scores it against truth.csv and records accuracy, cost, latency and manual effort.

Outputs, in --out (default benchmarks/results/<timestamp>/):
    results.csv     one row per run, every metric
    summary.md      mean +/- standard deviation per config and profile, ready for the article
    runs/*.json     per-run detail including every model call's tokens, latency and cost
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon.evaluate import evaluate  # noqa: E402
from recon.funnel import load_config, run_funnel  # noqa: E402
from recon.normalize import load_mapping, normalize_file  # noqa: E402

RULE_LAYERS = {"reference_id", "exact", "fuzzy", "combination"}


def dataset(profile: str, events: int, seed: int, data_dir: Path) -> Path:
    d = data_dir / f"{profile}_{events}_s{seed}"
    if not (d / "truth.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "synth/generate.py"), "--seed", str(seed),
                        "--events", str(events), "--profile", profile, "--out", str(d)],
                       check=True, capture_output=True)
    return d


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def run_one(config_path: str, data: Path, minutes_per_item: float) -> tuple[dict, dict]:
    fc = load_config(config_path)
    meta = json.loads((data / "meta.json").read_text())
    bank = normalize_file(data / "bank.csv", load_mapping(ROOT / "mappings/bank_default.json"))
    ledger = normalize_file(data / "ledger.csv", load_mapping(ROOT / "mappings/ledger_default.json"))
    if bank.issues or ledger.issues:
        raise SystemExit(f"{data}: normalization failed: {(bank.issues + ledger.issues)[:3]}")

    t0 = time.perf_counter()
    result = run_funnel(bank.transactions, ledger.transactions, fc,
                        date.fromisoformat(meta["period"]["end"]), data / "truth.csv")
    wall = time.perf_counter() - t0
    ev = evaluate(result, data / "truth.csv")

    total = len(result.transactions)
    matched_lines = sum(len(m.transaction_ids) for m in result.matches)
    last_rule = [n for n in result.remaining_after if n in RULE_LAYERS]
    left_after_rules = result.remaining_after[last_rule[-1]] if last_rule else total
    rules_seconds = sum(v for k, v in result.stage_seconds.items() if k in RULE_LAYERS)

    row = {
        "config": fc.name, "profile": meta.get("profile", "realistic"), "events": meta["events_requested"],
        "seed": meta["seed"], "simulated": fc.uses_simulation, "lines": total,
        "precision": ev["pairs"]["precision"], "recall_all": ev["pairs"]["recall_all"],
        "recall_rule_matchable": ev["pairs"]["recall"], "wrong_matches": ev["pairs"]["wrong"],
        "auto_match_rate": round(matched_lines / total, 4),
        "left_after_rules": left_after_rules, "left_after_rules_share": round(left_after_rules / total, 4),
        "items_for_person": len(result.exceptions),
        "person_minutes": round(len(result.exceptions) * minutes_per_item, 1),
        "rules_seconds": round(rules_seconds, 3), "wall_seconds": round(wall, 3),
    }
    cost_total, model_seconds = 0.0, 0.0
    for stage in fc.stages:
        name = stage.layer_name
        calls = [u for u in result.usage if u["stage"] == name]
        made = [u for u in calls if u["outcome"] not in ("over_budget", "error")]
        outcomes = Counter(u["outcome"] for u in calls)
        lat = [u["latency_s"] for u in made]
        cost = sum(u["cost_usd"] for u in made)
        cost_total += cost
        model_seconds += sum(lat)
        row.update({
            f"{name}_calls": len(made), f"{name}_matched": outcomes["matched"],
            f"{name}_escalated": outcomes["escalated"], f"{name}_said_none": outcomes["said_none"],
            f"{name}_over_budget": outcomes["over_budget"], f"{name}_errors": outcomes["error"],
            f"{name}_tokens_in": sum(u["input_tokens"] for u in made),
            f"{name}_tokens_out": sum(u["output_tokens"] for u in made),
            f"{name}_tokens_estimated": any(u.get("estimated_tokens") for u in made),
            f"{name}_cost_usd": round(cost, 6),
            f"{name}_latency_p50_s": round(pct(lat, 0.5), 4), f"{name}_latency_p95_s": round(pct(lat, 0.95), 4),
            f"{name}_seconds": round(sum(lat), 3),
        })
    row.update({
        "cost_usd": round(cost_total, 6),
        "cost_per_1k_lines_usd": round(cost_total / total * 1000, 6),
        "model_seconds": round(model_seconds, 3),
        # Model calls run one after another, so total time is rules + every call's latency.
        "est_total_seconds": round(rules_seconds + model_seconds, 3),
    })
    detail = {"metrics": row, "config": json.loads(Path(config_path).read_text()),
              "dataset": str(data), "stage_seconds": result.stage_seconds,
              "remaining_after": result.remaining_after, "usage": result.usage,
              "scenarios": ev["scenarios"]}
    return row, detail


def summarize(rows: list[dict], minutes_per_item: float) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["profile"], r["events"])].append(r)

    def ms(key: str, rs: list[dict], fmt: str) -> str:
        vals = [r[key] for r in rs]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{fmt.format(mean)} ± {fmt.format(sd)}" if len(vals) > 1 else fmt.format(mean)

    lines = ["# Benchmark summary", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. Mean ± standard deviation across seeds. "
             f"Manual effort assumes {minutes_per_item:g} minutes per item left for a person.", ""]
    if any(r["simulated"] for r in rows):
        lines += ["> **Some rows use simulated model stages** (marked *sim*). They test the pipeline and "
                  "budget logic only; their accuracy, cost and latency are not model results.", ""]
    lines += ["| Config | Profile | Lines | Precision | Recall (all) | Left after rules | Items for a person "
              "| Person-hours | Cost / 1k lines | Est. time (s) | Seeds |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (cfg, profile, events), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        name = cfg + (" *sim*" if rs[0]["simulated"] else "")
        hours = [r["person_minutes"] / 60 for r in rs]
        hours_txt = (f"{statistics.mean(hours):.1f} ± {statistics.stdev(hours):.1f}" if len(hours) > 1
                     else f"{hours[0]:.1f}")
        lines.append(
            f"| {name} | {profile} | {rs[0]['lines']:,} | {ms('precision', rs, '{:.2%}')} "
            f"| {ms('recall_all', rs, '{:.2%}')} | {ms('left_after_rules_share', rs, '{:.1%}')} "
            f"| {ms('items_for_person', rs, '{:,.0f}')} | {hours_txt} "
            f"| {ms('cost_per_1k_lines_usd', rs, '${:.4f}')} | {ms('est_total_seconds', rs, '{:,.1f}')} | {len(rs)} |")
    lines += ["", "Precision: share of automatic matches that are right. Recall (all): share of all real "
              "matches made automatically. Left after rules: lines the rules couldn't settle, which is "
              "what model stages are billed on. Est. time: rule time plus every model call's latency, "
              "run one after another.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--profiles", nargs="+", default=["realistic"], choices=["easy", "realistic", "hard"])
    ap.add_argument("--events", type=int, nargs="+", default=[3000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--minutes-per-item", type=float, default=2.0)
    ap.add_argument("--data-dir", default=str(ROOT / "benchmarks/data"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or ROOT / "benchmarks/results" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    (out / "runs").mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    rows = []
    jobs = [(c, p, n, s) for p in args.profiles for n in args.events for s in args.seeds for c in args.configs]
    for i, (cfg, profile, events, seed) in enumerate(jobs, 1):
        data = dataset(profile, events, seed, data_dir)
        print(f"[{i}/{len(jobs)}] {Path(cfg).stem} on {data.name} ...", flush=True)
        row, detail = run_one(cfg, data, args.minutes_per_item)
        rows.append(row)
        (out / "runs" / f"{Path(cfg).stem}__{data.name}.json").write_text(json.dumps(detail, indent=1, default=str))
        print(f"      precision {row['precision']:.2%}  recall {row['recall_all']:.2%}  "
              f"for a person {row['items_for_person']:,}  cost ${row['cost_usd']:.4f}  "
              f"est. {row['est_total_seconds']:,.1f}s", flush=True)

    header = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text(summarize(rows, args.minutes_per_item))
    print(f"\nWrote {out}/results.csv and summary.md")


if __name__ == "__main__":
    main()
