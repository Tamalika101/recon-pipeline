"""Core records of the reconciliation engine.

These mirror the data model in the design doc (TRANSACTION, MATCH, EXCEPTION)
so the engine's output can be written straight to Postgres later.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date

# Layer names, in funnel order. "manual" is reserved for matches a person confirms.
LAYER_REFERENCE = "reference_id"
LAYER_EXACT = "exact"
LAYER_FUZZY = "fuzzy"
LAYER_COMBINATION = "combination"
LAYER_MANUAL = "manual"

# Exception reason buckets.
BUCKET_TIMING = "timing"
BUCKET_MISSING = "missing_from_books"
BUCKET_ERROR = "error"
BUCKET_PROBLEM = "real_problem"


@dataclass(frozen=True)
class Transaction:
    id: str                 # "<source>:<source_row>", e.g. "bank:12"
    source: str             # "bank" | "ledger" (schema is open to "processor")
    source_row: int         # 1-based data row in the uploaded file
    txn_date: date
    amount_cents: int       # signed: + means money into the bank account
    description: str        # as uploaded
    clean_description: str  # normalized for comparison
    reference_id: str = ""
    external_id: str = ""   # e.g. the ledger's Entry ID

    @property
    def amount(self) -> float:
        return self.amount_cents / 100


@dataclass
class Match:
    match_id: str
    layer: str
    match_type: str         # one_to_one | one_to_many | many_to_one
    confidence: float       # 0-100
    reason: str
    transaction_ids: list[str]


@dataclass
class Suggestion:
    bank_ids: list[str]
    ledger_ids: list[str]
    layer: str
    confidence: float
    reason: str
    transposition: bool = False


@dataclass
class ExceptionItem:
    transaction_id: str
    suggested_bucket: str | None   # None = the system can't tell; a person decides
    reason: str
    suggestions: list[Suggestion] = field(default_factory=list)
    decided_bucket: str | None = None   # set when a person resolves the item
    note: str = ""

    @property
    def resolved(self) -> bool:
        return self.decided_bucket is not None

    @property
    def bucket(self) -> str | None:
        """The person's decision if there is one, otherwise the system's suggestion."""
        return self.decided_bucket or self.suggested_bucket


BUCKETS = [BUCKET_TIMING, BUCKET_MISSING, BUCKET_ERROR, BUCKET_PROBLEM]


@dataclass
class MatchConfig:
    """Per-firm matching settings. Each run stores a copy of what it used."""
    exact_window_days: int = 5
    fuzzy_window_days: int = 10
    bank_lead_days: int = 0          # days a bank line may post *before* its ledger line
    amount_tolerance_cents: int = 0
    auto_accept: float = 90.0        # fuzzy score at or above this is matched automatically
    suggest: float = 60.0            # fuzzy score at or above this is suggested to a person
    ambiguity_margin: float = 10.0   # best candidate must beat the runner-up by this much
    max_group_size: int = 5
    combo_window_days: int = 7
    combo_pool_size: int = 25
    timing_days: int = 5             # ledger items this close to period end look like timing
    weights: dict = field(default_factory=lambda: {"amount": 0.5, "date": 0.2, "description": 0.3})
    layers: list = field(default_factory=lambda: [LAYER_REFERENCE, LAYER_EXACT,
                                                  LAYER_FUZZY, LAYER_COMBINATION])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
