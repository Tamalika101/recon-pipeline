"""Load a pipeline configuration and run the full funnel: rules, then model stages.

A configuration is a JSON file:

    {
      "name": "rules+decider+llm (local)",
      "rules": {"auto_accept": 90},                  # MatchConfig overrides
      "stages": [
        {"kind": "decider", "provider": "ollama", "model": "strands-decider",
         "threshold": 0.9, "price_in_per_mtok": 0.05, "price_out_per_mtok": 0.05},
        {"kind": "llm", "provider": "ollama", "model": "qwen3.5:9b",
         "accept_confidence": ["high"], "max_calls": 500,
         "price_in_per_mtok": 0.10, "price_out_per_mtok": 0.40}
      ]
    }

Keys starting with "_" are comments and ignored.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .ai import NONE, DecisionItem, make_provider
from .engine import RunResult, run_matching
from .model_stages import ModelStage, StageConfig
from .models import MatchConfig, Transaction


@dataclass
class FunnelConfig:
    name: str
    rules: MatchConfig
    stages: list[StageConfig] = field(default_factory=list)
    description: str = ""
    path: str = ""

    @property
    def uses_simulation(self) -> bool:
        return any(s.provider == "simulated" for s in self.stages)


def load_config(path: str | Path) -> FunnelConfig:
    raw = json.loads(Path(path).read_text())
    rules = MatchConfig.from_dict({k: v for k, v in raw.get("rules", {}).items() if not k.startswith("_")})
    stages = [StageConfig.from_dict(s) for s in raw.get("stages", [])]
    names = [s.layer_name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: stage names must be unique, got {names}")
    return FunnelConfig(raw.get("name", Path(path).stem), rules, stages, raw.get("description", ""), str(path))


def truth_oracle(truth_path: str | Path):
    """For simulated providers only: the correct option for an item, from truth.csv."""
    true_sets: dict[str, frozenset] = {}
    with open(truth_path) as f:
        for r in csv.DictReader(f):
            if r["bank_row"] and r["expected_outcome"] == "matched":
                true_sets[f"bank:{r['bank_row']}"] = frozenset(f"ledger:{x}" for x in r["ledger_rows"].split(";") if x)

    def oracle(item: DecisionItem) -> str:
        want = true_sets.get(item.bank_id)
        for key, ids in item.option_ids.items():
            if want and frozenset(ids) == want:
                return key
        return NONE
    return oracle


def run_funnel(bank: list[Transaction], ledger: list[Transaction], fc: FunnelConfig,
               period_end: date | None = None, truth_path: str | Path | None = None) -> RunResult:
    oracle = truth_oracle(truth_path) if fc.uses_simulation and truth_path else None
    stages = [ModelStage(s, make_provider(s.provider_spec(), oracle)) for s in fc.stages]
    return run_matching(bank, ledger, fc.rules, period_end, extra_layers=stages)
