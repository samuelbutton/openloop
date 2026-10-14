"""Typed operator activity, separate from experiment measurements. No I/O."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .validation import (
    JSONValue,
    check_digest,
    check_positive,
    check_text,
    coerce_enum,
    freeze_object,
)

MAX_IMPLEMENTATION_ATTEMPTS = 3


class ActivityKind(StrEnum):
    HYPOTHESIS = "hypothesis"
    PROPOSAL_FAILURE = "proposal_failure"
    IMPLEMENTATION_STARTED = "implementation_started"
    IMPLEMENTATION_PROGRESS = "implementation_progress"
    IMPLEMENTATION_FAILURE = "implementation_failure"
    IMPLEMENTATION_SUCCESS = "implementation_success"
    CANDIDATE_BOUND = "candidate_bound"
    IDEA_FAILURE = "idea_failure"
    HYPOTHESIS_OUTCOME = "hypothesis_outcome"
    SPEND_RESERVATION = "spend_reservation"
    SPEND_SETTLEMENT = "spend_settlement"


class HypothesisOutcome(StrEnum):
    """Decided end state of one hypothesis card."""

    VERIFIED = "verified"
    DISCARDED = "discarded"
    INCONCLUSIVE = "inconclusive"
    SCREEN_DROPPED = "screen_dropped"


@dataclass(frozen=True)
class Activity:
    """One append-only operator occurrence. Payloads are not experiment scores."""

    sequence: int
    kind: ActivityKind
    subject_hash: str
    timestamp: str
    payload: Mapping[str, JSONValue]
    parent_sequence: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        check_positive(self.sequence)
        check_digest(self.subject_hash)
        check_text(self.timestamp)
        if self.parent_sequence is not None:
            check_positive(self.parent_sequence)
        object.__setattr__(
            self, "kind", coerce_enum(ActivityKind, self.kind, "activity kind")
        )
        object.__setattr__(self, "payload", freeze_object(self.payload))
