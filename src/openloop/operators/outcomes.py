"""Hypothesis outcomes and calibration of model predictions. No model authority.

Boundary: a trusted coordinator passes decider Decisions. This module checks that
a Decision belongs to the card's bound candidate and records exactly one outcome
per card. The summary is a pure read of those records. Outcomes are reports for
the operator of the system; they are not placed in proposer context, so
held-out-derived outcomes cannot leak into later proposals.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from openloop.decider import Decision, DecisionStage, Verdict
from openloop.ledger import (
    Activity,
    ActivityKind,
    ConflictError,
    Direction,
    HypothesisOutcome,
    InvalidInputError,
    JSONValue,
    Ledger,
    canonical_json,
    content_hash,
)

from .models import HypothesisCard


def outcome_for(decision: Decision) -> HypothesisOutcome:
    """Map a terminal decider verdict to an outcome; reject non-terminal ones.

    Held-out verified is the only verified outcome. A confirmation promotion is
    not terminal because held-out can still reject the selected candidate. A
    confirmation discard with no samples was excluded by a top-k limit, never
    measured, so it is a screen drop.
    """
    stage, verdict = decision.stage, decision.verdict
    if stage is DecisionStage.SCREEN and verdict is Verdict.DISCARD:
        return HypothesisOutcome.SCREEN_DROPPED
    if stage is DecisionStage.CONFIRM and verdict is Verdict.DISCARD:
        return (
            HypothesisOutcome.DISCARDED
            if decision.sample_count > 0
            else HypothesisOutcome.SCREEN_DROPPED
        )
    if stage is not DecisionStage.SCREEN and verdict is Verdict.INCONCLUSIVE:
        return HypothesisOutcome.INCONCLUSIVE
    if stage is DecisionStage.HELD_OUT and verdict is Verdict.VERIFIED:
        return HypothesisOutcome.VERIFIED
    if stage is DecisionStage.HELD_OUT and verdict is Verdict.DISCARD:
        return HypothesisOutcome.DISCARDED
    raise InvalidInputError(
        f"A {stage.value} {verdict.value} decision is not a terminal outcome"
    )


def _recorded_card(history: tuple[Activity, ...]) -> Activity:
    hypothesis = next(
        (event for event in history if event.kind is ActivityKind.HYPOTHESIS), None
    )
    if hypothesis is None:
        raise InvalidInputError("Outcome requires a recorded hypothesis")
    return hypothesis


def _mapping(value: object, label: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise InvalidInputError(f"Recorded {label} is malformed")
    return value  # pyright: ignore[reportUnknownVariableType]


def _oriented(gain: float | None, direction: Direction) -> float | None:
    """Turn a decider gain (improvement positive) into a signed delta."""
    if gain is None:
        return None
    return -gain if direction is Direction.MINIMIZE else gain


def record_outcome_for_hash(
    ledger: Ledger, card_hash: str, decision: Decision
) -> Activity:
    """Record the single outcome of a recorded card; see `record_outcome`."""
    outcome = outcome_for(decision)
    history = ledger.activities(subject_hash=card_hash)
    hypothesis = _recorded_card(history)
    binding = next(
        (event for event in history if event.kind is ActivityKind.CANDIDATE_BOUND),
        None,
    )
    if (
        binding is None
        or binding.payload.get("candidate_hash") != decision.candidate_hash
    ):
        raise InvalidInputError(
            "Decision does not match the bound implemented candidate"
        )
    decision_hash = content_hash(decision)
    for failure in (e for e in history if e.kind is ActivityKind.IDEA_FAILURE):
        if (
            outcome is not HypothesisOutcome.DISCARDED
            or failure.payload.get("decision_hash") != decision_hash
        ):
            raise ConflictError(
                "An idea failure implies a discarded outcome from the same decision"
            )
    card = _mapping(hypothesis.payload.get("card"), "card")
    prediction = _mapping(card.get("prediction"), "prediction")
    direction = Direction(str(prediction.get("direction")))
    measured = decision.sample_count > 0 and decision.median_gain is not None
    low, high = decision.interval
    observed: dict[str, JSONValue] | None = None
    if measured:
        interval: tuple[float, float] | None = None
        if low is not None and high is not None:
            ends = (-low, -high) if direction is Direction.MINIMIZE else (low, high)
            interval = (min(ends), max(ends))
        observed = {
            "median_gain": decision.median_gain,
            "delta": _oriented(decision.median_gain, direction),
            "interval": interval,
            "sample_count": decision.sample_count,
        }
    return ledger.record_activity(
        ActivityKind.HYPOTHESIS_OUTCOME,
        card_hash,
        {
            "outcome": outcome,
            "decision_hash": decision_hash,
            "decision": decision,
            "stage": decision.stage,
            "verdict": decision.verdict,
            "protocol_hash": decision.protocol_hash,
            "candidate_hash": decision.candidate_hash,
            "predicted": {
                "metric": prediction.get("metric"),
                "direction": direction,
                "fidelity": prediction.get("fidelity"),
                "delta": prediction.get("delta"),
                "probability_of_improvement": prediction.get(
                    "probability_of_improvement"
                ),
                "interval": prediction.get("interval"),
            },
            "observed": observed,
        },
        key=f"outcome:{card_hash}",
        parent_sequence=hypothesis.sequence,
    )


def record_outcome(
    ledger: Ledger, card: HypothesisCard, decision: Decision
) -> Activity:
    """Record exactly one outcome for a card from a trusted decider decision.

    verified: held-out pass. discarded: a measured confirmation or held-out
    discard. inconclusive: an incomplete confirmation or held-out schedule that
    the coordinator stops extending. screen_dropped: a screen discard, an
    allocator drop, or a confirmation excluded by a top-k limit. The record
    links the card's hypothesis, the decision's content hash, the protocol hash,
    the prediction, and the observed effect when measured. The candidate must
    already be bound with `bind_candidate`. A card that has an idea failure may
    record only a discarded outcome from the same decision.
    """
    history = ledger.activities(subject_hash=card.hash)
    recorded = _recorded_card(history)
    if canonical_json(recorded.payload.get("card")) != canonical_json(card):
        raise InvalidInputError("Outcome requires the exact recorded hypothesis card")
    return record_outcome_for_hash(ledger, card.hash, decision)


@dataclass(frozen=True)
class CalibrationSummary:
    """Prediction quality over recorded outcomes.

    The probability target is whether the observed paired median gain is
    positive. Only verified and discarded outcomes with a measured gain are
    scored: inconclusive outcomes have an incomplete schedule and screen drops
    rest on two noisy pairs or on no measurement. Counts include every outcome.
    """

    counts: Mapping[str, int]
    scored: int
    brier_score: float | None
    mean_signed_error: float | None
    interval_covered: int
    interval_total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def calibration_summary(ledger: Ledger) -> CalibrationSummary:
    """Brier score, mean signed delta error (predicted - observed), and counts."""
    counts = {item.value: 0 for item in HypothesisOutcome}
    squared: list[float] = []
    errors: list[float] = []
    covered = total = 0
    for event in ledger.activities():
        if event.kind is not ActivityKind.HYPOTHESIS_OUTCOME:
            continue
        outcome = HypothesisOutcome(str(event.payload.get("outcome")))
        counts[outcome.value] += 1
        if outcome not in (HypothesisOutcome.VERIFIED, HypothesisOutcome.DISCARDED):
            continue
        predicted = _mapping(event.payload.get("predicted"), "prediction")
        observed = event.payload.get("observed")
        if not isinstance(observed, Mapping):
            continue
        gain = _number(observed.get("median_gain"))
        delta = _number(observed.get("delta"))
        probability = _number(predicted.get("probability_of_improvement"))
        guess = _number(predicted.get("delta"))
        if gain is None or delta is None or probability is None or guess is None:
            continue
        squared.append((probability - (1.0 if gain > 0 else 0.0)) ** 2)
        errors.append(guess - delta)
        interval = predicted.get("interval")
        if isinstance(interval, list | tuple) and len(interval) == 2:  # pyright: ignore[reportUnknownArgumentType]
            low, high = _number(interval[0]), _number(interval[1])
            if low is not None and high is not None:
                total += 1
                covered += low <= delta <= high
    return CalibrationSummary(
        counts,
        len(squared),
        sum(squared) / len(squared) if squared else None,
        sum(errors) / len(errors) if errors else None,
        covered,
        total,
    )


__all__ = [
    "CalibrationSummary",
    "calibration_summary",
    "outcome_for",
    "record_outcome",
    "record_outcome_for_hash",
]
