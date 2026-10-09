"""Content-addressed inputs, append-only run evidence, and explicit replication.

Only trusted callers may complete runs. This package records evidence; it does
not execute candidate code, validate an evaluator, or make statistical decisions.
"""

from .errors import (
    ConflictError,
    InvalidInputError,
    InvalidTransitionError,
    LedgerError,
    NotFoundError,
    SchemaError,
)
from .identity import (
    IDENTITY_VERSION,
    ExperimentInputs,
    Reproducibility,
    canonical_json,
    content_hash,
)
from .records import (
    Direction,
    Event,
    EventKind,
    Metric,
    Preparation,
    PreparationStage,
    ProbeResult,
    Purpose,
    Result,
    Run,
    Stage,
    Status,
    compare_results,
    validate_tolerances,
)
from .schema import SCHEMA_VERSION
from .store import Ledger
from .validation import JSONValue

__all__ = [
    "IDENTITY_VERSION",
    "SCHEMA_VERSION",
    "ConflictError",
    "Direction",
    "Event",
    "EventKind",
    "ExperimentInputs",
    "InvalidInputError",
    "InvalidTransitionError",
    "JSONValue",
    "Ledger",
    "LedgerError",
    "Metric",
    "NotFoundError",
    "Preparation",
    "PreparationStage",
    "ProbeResult",
    "Purpose",
    "Reproducibility",
    "Result",
    "Run",
    "SchemaError",
    "Stage",
    "Status",
    "canonical_json",
    "compare_results",
    "content_hash",
    "validate_tolerances",
]
