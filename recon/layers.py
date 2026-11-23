"""The matching funnel: four pluggable layers, strongest evidence first.

Each layer only sees transactions earlier layers left unmatched. A layer can
match, suggest (for a person to confirm), or mark items ambiguous so later
layers won't auto-match them. Ambiguity is never guessed away.

To add a layer (the future plugin slot), subclass Layer, give it a unique
`name`, decorate it with @register_layer and add its name to MatchConfig.layers.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import (LAYER_COMBINATION, LAYER_EXACT, LAYER_FUZZY, LAYER_REFERENCE,
                     Match, MatchConfig, Suggestion, Transaction)
from .text import similarity

LAYER_REGISTRY: dict[str, type["Layer"]] = {}


def register_layer(cls: type["Layer"]) -> type["Layer"]:
    if not cls.name:
        raise ValueError("a layer needs a name")
    LAYER_REGISTRY[cls.name] = cls
    return cls


# ---------------------------------------------------------------------------
# Shared state passed through the funnel
# ---------------------------------------------------------------------------


class MatchState:
    def __init__(self, bank: list[Transaction], ledger: list[Transaction], cfg: MatchConfig):
        self.cfg = cfg
        self.bank = sorted(bank, key=lambda t: t.source_row)
        self.ledger = sorted(ledger, key=lambda t: t.source_row)
        self.by_id = {t.id: t for t in self.bank + self.ledger}
        self.unmatched: set[str] = set(self.by_id)
        self.matches: list[Match] = []
        self.suggestions: list[Suggestion] = []
        self.ambiguous: set[str] = set()
        self._suggested_pairs: set[tuple] = set()

    def open_bank(self) -> list[Transaction]:
        return [t for t in self.bank if t.id in self.unmatched]

    def open_ledger(self) -> list[Transaction]:
        return [t for t in self.ledger if t.id in self.unmatched]

    def add_match(self, bank: list[Transaction], ledger: list[Transaction], layer: str,
                  confidence: float, reason: str) -> Match:
        ids = [t.id for t in bank + ledger]
        assert all(i in self.unmatched for i in ids), "transaction matched twice"
        match_type = ("one_to_one" if len(bank) == 1 and len(ledger) == 1
                      else "one_to_many" if len(bank) == 1 else "many_to_one")
        m = Match(match_id=f"M{len(self.matches) + 1:05d}", layer=layer, match_type=match_type,
                  confidence=round(confidence, 1), reason=reason, transaction_ids=ids)
        self.matches.append(m)
        self.unmatched.difference_update(ids)
        return m

    def suggest(self, bank: list[Transaction], ledger: list[Transaction], layer: str,
                confidence: float, reason: str, transposition: bool = False) -> None:
        key = (tuple(t.id for t in bank), tuple(t.id for t in ledger))
        if key in self._suggested_pairs:
            return
        self._suggested_pairs.add(key)
        self.suggestions.append(Suggestion([t.id for t in bank], [t.id for t in ledger],
                                           layer, round(confidence, 1), reason, transposition))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lag_days(bank: Transaction, ledger: Transaction) -> int:
    """How many days after the ledger date the bank posted (negative = before)."""
    return (bank.txn_date - ledger.txn_date).days


def in_window(bank: Transaction, ledger: Transaction, days: int, cfg: MatchConfig) -> bool:
    return -cfg.bank_lead_days <= lag_days(bank, ledger) <= days


def same_sign(a: int, b: int) -> bool:
    return (a > 0) == (b > 0)


def is_transposition(a: int, b: int) -> bool:
    """Two digits swapped: same digits in a different order (difference is a multiple of 9)."""
    if a == b or not same_sign(a, b):
        return False
    da, db = str(abs(a)), str(abs(b))
    return len(da) == len(db) and sorted(da) == sorted(db)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def describe_lag(days: int) -> str:
    if days == 0:
        return "same day"
    if days > 0:
        return f"bank posted {plural(days, 'day')} later"
    return f"bank posted {plural(-days, 'day')} earlier"


class Layer:
    name = ""

    def run(self, state: MatchState) -> None:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Layer 1: shared reference ID
# ---------------------------------------------------------------------------


@register_layer
class ReferenceLayer(Layer):
    """A bank line and a ledger line carrying the same reference are the same money.

    If several ledger lines share the reference and amount (an integration retry
    posting twice), the closest-dated one is matched and the rest are left for
    exception handling, where they're flagged as likely duplicates.
    """
    name = LAYER_REFERENCE

    def run(self, state: MatchState) -> None:
        cfg = state.cfg
        index: dict[str, list[Transaction]] = defaultdict(list)
        for led in state.open_ledger():
            if led.reference_id:
                index[led.reference_id].append(led)

        for b in state.open_bank():
            if not b.reference_id or b.reference_id not in index:
                continue
            cands = [l for l in index[b.reference_id] if l.id in state.unmatched
                     and same_sign(l.amount_cents, b.amount_cents)]
            equal = [l for l in cands
                     if abs(l.amount_cents - b.amount_cents) <= cfg.amount_tolerance_cents]
            if equal:
                chosen = min(equal, key=lambda l: (abs(lag_days(b, l)), l.source_row))
                extra = f"; {plural(len(equal) - 1, 'other ledger line')} share it" if len(equal) > 1 else ""
                state.add_match([b], [chosen], self.name, 100,
                                f"Shared reference {b.reference_id}, same amount{extra}")
            elif cands:
                for l in cands:
                    state.suggest([b], [l], self.name, 70,
                                  f"Shared reference {b.reference_id} but amounts differ "
                                  f"({b.amount:.2f} vs {l.amount:.2f})",
                                  transposition=is_transposition(b.amount_cents, l.amount_cents))


# ---------------------------------------------------------------------------
# Layer 2: exact amount within the date window
# ---------------------------------------------------------------------------


@register_layer
class ExactLayer(Layer):
    """Same amount, bank date inside the window, and exactly one candidate both ways."""
    name = LAYER_EXACT

    def run(self, state: MatchState) -> None:
        cfg = state.cfg
        tol = cfg.amount_tolerance_cents
        by_amount: dict[int, list[Transaction]] = defaultdict(list)
        for l in state.open_ledger():
            by_amount[l.amount_cents].append(l)

        def candidates(b: Transaction) -> list[Transaction]:
            pool: Iterable[Transaction]
            if tol == 0:
                pool = by_amount.get(b.amount_cents, [])
            else:
                pool = [l for amt, ls in by_amount.items() if abs(amt - b.amount_cents) <= tol for l in ls]
            return [l for l in pool if in_window(b, l, cfg.exact_window_days, cfg)]

        forward = {b.id: candidates(b) for b in state.open_bank()}
        reverse: dict[str, list[str]] = defaultdict(list)
        for bid, ls in forward.items():
            for l in ls:
                reverse[l.id].append(bid)

        for bid, ls in forward.items():
            b = state.by_id[bid]
            if len(ls) == 1 and len(reverse[ls[0].id]) == 1:
                l = ls[0]
                state.add_match([b], [l], self.name, 95,
                                f"Same amount, {describe_lag(lag_days(b, l))}, only candidate")
            elif ls:
                rivals = {bid, *(x for l in ls for x in reverse[l.id])}
                state.ambiguous.update(rivals)
                state.ambiguous.update(l.id for l in ls)
                for l in ls:
                    state.suggest([b], [l], self.name, 70,
                                  f"Same amount, {describe_lag(lag_days(b, l))}, but "
                                  f"{plural(len(rivals) - 1 + len(ls) - 1, 'rival')} fit equally well")


# ---------------------------------------------------------------------------
# Layer 3: scored fuzzy match
# ---------------------------------------------------------------------------


@register_layer
class FuzzyLayer(Layer):
    """Score pairs on amount closeness, date distance and description similarity.

    Auto-accepts only when the score clears `auto_accept`, the pair is each
    other's clear best (by `ambiguity_margin`), and neither side was marked
    ambiguous upstream. Otherwise good-enough pairs become suggestions.
    """
    name = LAYER_FUZZY
    AMOUNT_BAND = 0.02  # relative difference at which amount closeness reaches 0

    def amount_score(self, a: int, b: int, cfg: MatchConfig) -> tuple[float, bool]:
        if abs(a - b) <= cfg.amount_tolerance_cents:
            return 1.0, False
        if is_transposition(a, b):
            return 0.6, True
        rel = abs(a - b) / max(abs(a), abs(b))
        return max(0.0, 1 - rel / self.AMOUNT_BAND), False

    def date_score(self, days: int, cfg: MatchConfig) -> float:
        days = abs(days)
        if days <= cfg.exact_window_days:
            return 1.0
        span = max(1, cfg.fuzzy_window_days - cfg.exact_window_days)
        return max(0.0, 1 - 0.6 * (days - cfg.exact_window_days) / span)

    def run(self, state: MatchState) -> None:
        cfg = state.cfg
        w = cfg.weights
        ledger = state.open_ledger()
        scored: list[tuple[float, Transaction, Transaction, str, bool]] = []

        for b in state.open_bank():
            for l in ledger:
                if not same_sign(b.amount_cents, l.amount_cents):
                    continue
                days = lag_days(b, l)
                if not -cfg.bank_lead_days <= days <= cfg.fuzzy_window_days:
                    continue
                a, transposed = self.amount_score(b.amount_cents, l.amount_cents, cfg)
                d = self.date_score(days, cfg)
                ceiling = 100 * (w["amount"] * a + w["date"] * d + w["description"])
                if ceiling < cfg.suggest and not transposed:
                    continue
                s = similarity(b.description, l.description)
                score = 100 * (w["amount"] * a + w["date"] * d + w["description"] * s)
                if score < cfg.suggest and not transposed:
                    continue
                amount_txt = ("amount digits transposed" if transposed
                              else "same amount" if a == 1.0
                              else f"amount differs by {abs(b.amount - l.amount):.2f}")
                reason = (f"Score {score:.0f}: {amount_txt}, {describe_lag(days)}, "
                          f"description {s:.0%} similar")
                scored.append((score, b, l, reason, transposed))

        scored.sort(key=lambda x: -x[0])
        assigned: set[str] = set()

        def runner_up(tid: str, exclude: str) -> float:
            best = 0.0
            for score, b, l, _, _ in scored:
                if tid not in (b.id, l.id):
                    continue
                other = l.id if b.id == tid else b.id
                if other == exclude or other in assigned:
                    continue
                best = max(best, score)
            return best

        for score, b, l, reason, transposed in scored:
            if b.id in assigned or l.id in assigned:
                continue
            if score < cfg.auto_accept or transposed:
                continue
            if b.id in state.ambiguous or l.id in state.ambiguous:
                continue
            margin = score - max(runner_up(b.id, l.id), runner_up(l.id, b.id))
            if margin < cfg.ambiguity_margin:
                state.ambiguous.update({b.id, l.id})
                continue
            state.add_match([b], [l], self.name, score, reason)
            assigned.update({b.id, l.id})

        per_bank: dict[str, int] = defaultdict(int)
        for score, b, l, reason, transposed in scored:
            if b.id in state.unmatched and l.id in state.unmatched and per_bank[b.id] < 3:
                state.suggest([b], [l], self.name, score, reason, transposition=transposed)
                per_bank[b.id] += 1


# ---------------------------------------------------------------------------
# Layer 4: combinations (one-to-many and many-to-one)
# ---------------------------------------------------------------------------


def find_subsets(target: int, pool: list[Transaction], max_size: int, tol: int,
                 limit: int = 2) -> list[list[Transaction]]:
    """Groups of 2..max_size items whose amounts sum to target. Stops after `limit`."""
    items = sorted(pool, key=lambda t: -abs(t.amount_cents))
    vals = [abs(t.amount_cents) for t in items]
    suffix = [0] * (len(vals) + 1)
    for i in range(len(vals) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + vals[i]
    goal = abs(target)
    found: list[list[Transaction]] = []
    chosen: list[Transaction] = []

    def dfs(start: int, remaining: int) -> None:
        if len(found) >= limit:
            return
        if abs(remaining) <= tol and len(chosen) >= 2:
            found.append(list(chosen))
            return
        if len(chosen) == max_size or suffix[start] < remaining - tol:
            return
        for j in range(start, len(vals)):
            if vals[j] > remaining + tol:
                continue
            chosen.append(items[j])
            dfs(j + 1, remaining - vals[j])
            chosen.pop()
            if len(found) >= limit:
                return

    dfs(0, goal)
    return found


@register_layer
class CombinationLayer(Layer):
    """One bank line equal to several ledger lines (lump deposit), or the reverse.

    A lump deposit batches payments received around the same time, so the search
    slides a narrow date band (combo_band_days) across the window instead of
    searching every line in it at once. That matters: with many candidates, some
    other set of 4-5 amounts will sum to the same total by coincidence. The group
    is matched only if it is the single combination found across all bands.
    """
    name = LAYER_COMBINATION
    BAND_DAYS = 3

    def _candidates(self, one: Transaction, many: list[Transaction], one_is_bank: bool,
                    state: MatchState) -> list[tuple[int, Transaction]]:
        cfg = state.cfg
        out = []
        for t in many:
            if t.id not in state.unmatched or not same_sign(t.amount_cents, one.amount_cents):
                continue
            if abs(t.amount_cents) >= abs(one.amount_cents):
                continue
            days = lag_days(one, t) if one_is_bank else lag_days(t, one)
            if -cfg.bank_lead_days <= days <= cfg.combo_window_days:
                out.append((days, t))
        return out

    def _solutions(self, one: Transaction, cands: list[tuple[int, Transaction]],
                   cfg: MatchConfig) -> list[list[Transaction]]:
        distinct: dict[frozenset, list[Transaction]] = {}
        lo = -cfg.bank_lead_days
        for start in range(lo, cfg.combo_window_days - self.BAND_DAYS + 2):
            band = [t for d, t in cands if start <= d < start + self.BAND_DAYS]
            band = band[: cfg.combo_pool_size]
            if len(band) < 2:
                continue
            for sol in find_subsets(one.amount_cents, band, cfg.max_group_size,
                                    cfg.amount_tolerance_cents):
                distinct.setdefault(frozenset(t.id for t in sol), sol)
            if len(distinct) >= 3:
                break
        return list(distinct.values())

    def _pass(self, ones: list[Transaction], many: list[Transaction], one_is_bank: bool,
              state: MatchState) -> None:
        cfg = state.cfg
        for one in sorted(ones, key=lambda t: -abs(t.amount_cents)):
            if one.id not in state.unmatched:
                continue
            cands = self._candidates(one, many, one_is_bank, state)
            if len(cands) < 2:
                continue
            sols = self._solutions(one, cands, cfg)
            if not sols:
                continue
            side = "ledger" if one_is_bank else "bank"
            if len(sols) == 1:
                group = sorted(sols[0], key=lambda t: t.source_row)
                bank, ledger = ([one], group) if one_is_bank else (group, [one])
                state.add_match(bank, ledger, self.name, 90,
                                f"{len(group)} {side} lines sum exactly to {one.amount:.2f}; "
                                f"no other combination fits")
                continue
            state.ambiguous.add(one.id)
            # Smaller, tighter groups first: coincidental combinations tend to be larger.
            def spread(sol: list[Transaction]) -> int:
                days = [t.txn_date.toordinal() for t in sol]
                return max(days) - min(days)
            sols.sort(key=lambda sol: (len(sol), spread(sol)))
            for rank, sol in enumerate(sols):
                group = sorted(sol, key=lambda t: t.source_row)
                bank, ledger = ([one], group) if one_is_bank else (group, [one])
                state.suggest(bank, ledger, self.name, 60 - 5 * rank,
                              f"{len(group)} {side} lines sum to {one.amount:.2f}, "
                              f"but {len(sols) - 1} other combination(s) also fit")

    def run(self, state: MatchState) -> None:
        self._pass(state.open_bank(), state.ledger, True, state)
        self._pass(state.open_ledger(), state.bank, False, state)
