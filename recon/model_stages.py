"""Decision-model and LLM stages, run after the rules.

What the models are allowed to do is deliberately narrow:

- They only see bank lines the rules left open that have candidates whose amounts
  already agree (same amount within the date window, or a group the rules found
  that sums exactly). Arithmetic is never up to a model: swapped digits, fees and
  unknown charges get no candidates and stay exceptions for a person.
- They pick one candidate or "none". The decision model's pick counts only above a
  probability threshold; the LLM's only at an accepted confidence level. Anything
  else stays open and moves on to the next stage, or to a person.
- Each stage can have a budget (calls and/or dollars). Items are handled largest
  amount first, so a budget is spent where the money is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ai import NONE, DecisionItem, ProviderError
from .layers import Layer, MatchState, describe_lag, lag_days
from .models import Transaction


@dataclass
class StageConfig:
    kind: str                                 # "decider" | "llm"
    provider: str = "ollama"                  # "ollama" | "anthropic" | "simulated"
    model: str = ""
    name: str = ""                            # layer name in results; defaults to kind
    threshold: float = 0.9                    # decider: minimum probability to accept
    accept_confidence: list = field(default_factory=lambda: ["high"])  # llm: accepted levels
    max_candidates: int = 4
    max_calls: int | None = None              # budget: model calls per run
    max_cost_usd: float | None = None         # budget: dollars per run
    price_in_per_mtok: float = 0.0            # $ per million input tokens
    price_out_per_mtok: float = 0.0           # $ per million output tokens
    max_consecutive_errors: int = 3
    extra: dict = field(default_factory=dict)  # provider settings: base_url, timeout_s, think, ...

    @classmethod
    def from_dict(cls, d: dict) -> "StageConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "extra"}
        extra = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__ and not k.startswith("_")}
        return cls(**known, extra=extra)

    def provider_spec(self) -> dict:
        return {"kind": self.kind, "provider": self.provider, "model": self.model, **self.extra}

    @property
    def layer_name(self) -> str:
        return self.name or self.kind


def describe_txn(t: Transaction) -> str:
    ref = f" | ref {t.reference_id}" if t.reference_id else ""
    return f"{t.txn_date.isoformat()} | {t.amount:+,.2f} | {t.description}{ref}"


def _combination_groups(state: MatchState) -> dict[str, list[list[str]]]:
    groups: dict[str, list[list[str]]] = {}
    for s in state.suggestions:
        if s.layer == "combination" and len(s.bank_ids) == 1 and len(s.ledger_ids) > 1:
            groups.setdefault(s.bank_ids[0], []).append(s.ledger_ids)
    return groups


def build_item(state: MatchState, b: Transaction, max_candidates: int,
               by_amount: dict[int, list[Transaction]] | None = None,
               groups: dict[str, list[list[str]]] | None = None) -> DecisionItem | None:
    """The question for one open bank line, from candidates still open right now."""
    cfg = state.cfg
    if by_amount is None:
        by_amount = {}
        for l in state.open_ledger():
            by_amount.setdefault(l.amount_cents, []).append(l)
    groups = _combination_groups(state) if groups is None else groups

    singles = []
    for cents in range(b.amount_cents - cfg.amount_tolerance_cents,
                       b.amount_cents + cfg.amount_tolerance_cents + 1):
        for l in by_amount.get(cents, []):
            if l.id in state.unmatched and -cfg.bank_lead_days <= lag_days(b, l) <= cfg.fuzzy_window_days:
                singles.append(l)
    singles.sort(key=lambda l: (abs(lag_days(b, l)), l.source_row))
    options: list[list[Transaction]] = [[l] for l in singles]
    for ids in groups.get(b.id, []):
        if all(i in state.unmatched for i in ids):
            options.append([state.by_id[i] for i in ids])
    options = options[:max_candidates]
    if not options:
        return None

    keys = [chr(ord("A") + i) for i in range(len(options))]
    desc, ids = {}, {}
    for k, opt in zip(keys, options):
        if len(opt) == 1:
            l = opt[0]
            desc[k] = f"{describe_txn(l)} ({describe_lag(lag_days(b, l))})"
        else:
            parts = "; ".join(f"{l.txn_date:%m-%d} {l.amount:,.2f} {l.description}" for l in opt)
            desc[k] = f"{len(opt)} entries that together total {b.amount:+,.2f}: {parts}"
        ids[k] = [l.id for l in opt]
    desc[NONE] = "None of these is the same transaction"
    return DecisionItem(
        bank_id=b.id,
        state=(f"Bank statement line: {describe_txn(b)}\n"
               f"The candidate book entries below have amounts that already agree with it. "
               f"Banks usually post 0 to {cfg.exact_window_days} days after the book date, and "
               f"bank descriptions are often abbreviated."),
        instructions=("Which book entry records the same transaction as the bank line? "
                      "Choose 'none' if no entry clearly does."),
        options=desc, option_ids=ids, value_cents=abs(b.amount_cents))


def build_items(state: MatchState, max_candidates: int) -> list[DecisionItem]:
    """One item per open bank line that has candidates with matching amounts, largest amount first."""
    by_amount: dict[int, list[Transaction]] = {}
    for l in state.open_ledger():
        by_amount.setdefault(l.amount_cents, []).append(l)
    groups = _combination_groups(state)
    items = [it for b in state.open_bank()
             if (it := build_item(state, b, max_candidates, by_amount, groups)) is not None]
    items.sort(key=lambda it: -it.value_cents)
    return items


class ModelStage(Layer):
    """A model stage in the funnel. `provider` is any object with decide(DecisionItem)."""

    def __init__(self, cfg: StageConfig, provider):
        self.cfg, self.provider = cfg, provider
        self.name = cfg.layer_name

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.cfg.price_in_per_mtok + tokens_out * self.cfg.price_out_per_mtok) / 1e6

    def _accepts(self, choice: str, confidence: float, level_ok: bool) -> bool:
        if choice == NONE:
            return False
        if self.cfg.kind == "decider":
            return confidence >= self.cfg.threshold
        return level_ok

    def run(self, state: MatchState) -> None:
        calls, spent, errors = 0, 0.0, 0
        groups = _combination_groups(state)
        for planned in build_items(state, self.cfg.max_candidates):
            if planned.bank_id not in state.unmatched:
                continue
            # Earlier decisions in this stage may have claimed some candidates: ask about what's left.
            item = build_item(state, state.by_id[planned.bank_id], self.cfg.max_candidates, groups=groups)
            if item is None:
                continue
            if self.cfg.max_calls is not None and calls >= self.cfg.max_calls:
                state.usage.append({"stage": self.name, "bank_id": item.bank_id, "outcome": "over_budget",
                                    "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0})
                continue
            if self.cfg.max_cost_usd is not None and spent >= self.cfg.max_cost_usd:
                state.usage.append({"stage": self.name, "bank_id": item.bank_id, "outcome": "over_budget",
                                    "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0})
                continue
            try:
                d = self.provider.decide(item)
                errors = 0
            except ProviderError as e:
                errors += 1
                state.usage.append({"stage": self.name, "bank_id": item.bank_id, "outcome": "error",
                                    "error": str(e)[:200], "input_tokens": 0, "output_tokens": 0,
                                    "latency_s": 0.0, "cost_usd": 0.0})
                if errors >= self.cfg.max_consecutive_errors:
                    raise ProviderError(f"{self.name} stage stopped after {errors} errors in a row: {e}") from e
                continue
            calls += 1
            cost = self._cost(d.usage.input_tokens, d.usage.output_tokens)
            spent += cost

            level = next((lvl for lvl, v in {"high": 0.9, "medium": 0.7, "low": 0.4}.items()
                          if abs(v - d.confidence) < 1e-9), None)
            level_ok = level in self.cfg.accept_confidence
            ids = item.option_ids.get(d.choice, [])
            outcome = "escalated"
            if self._accepts(d.choice, d.confidence, level_ok) and ids \
                    and all(i in state.unmatched for i in ids):
                b = state.by_id[item.bank_id]
                state.add_match([b], [state.by_id[i] for i in ids], self.name, d.confidence * 100,
                                f"{self.name} chose option {d.choice} "
                                f"({d.confidence:.0%}{', ' + d.reason if d.reason else ''})")
                outcome = "matched"
            elif d.choice == NONE:
                outcome = "said_none"
            state.usage.append({"stage": self.name, "bank_id": item.bank_id, "outcome": outcome,
                                "choice": d.choice, "confidence": round(d.confidence, 4),
                                "input_tokens": d.usage.input_tokens, "output_tokens": d.usage.output_tokens,
                                "estimated_tokens": d.usage.estimated_tokens,
                                "latency_s": round(d.usage.latency_s, 4), "cost_usd": cost})
