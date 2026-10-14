"""Ledger facts that say what a finding changed and where it came from.

Only recorded inputs, activity sequences, and numbers appear here. Hypothesis
titles, rationale, and edit text never do: a model's story is not a fact.
"""

from collections.abc import Mapping

from openloop.decider import Decision, Pair
from openloop.ledger import (
    Activity,
    ActivityKind,
    Direction,
    InvalidInputError,
    JSONValue,
    Ledger,
    Metric,
    canonical_json,
)
from openloop.ledger.validation import plain

type Facts = dict[str, JSONValue]
_PREDICTION_FIELDS = (
    "metric",
    "direction",
    "fidelity",
    "delta",
    "probability_of_improvement",
    "interval",
)


def describe_change(pairs: tuple[Pair, ...]) -> Facts:
    """Compare the baseline and candidate inputs recorded for the final pairs.

    Configuration keys are compared at the top level; a changed value is shown
    whole, old to new. `code_changed` is true when the source or dependency
    hash differs. Every pair must show the same difference apart from the seed.
    """
    shapes = {
        (
            pair.baseline.inputs.source_hash,
            pair.baseline.inputs.dependencies_hash,
            pair.candidate.inputs.source_hash,
            pair.candidate.inputs.dependencies_hash,
            canonical_json(pair.baseline.inputs.config),
            canonical_json(pair.candidate.inputs.config),
        )
        for pair in pairs
    }
    if len(shapes) != 1:
        raise InvalidInputError("Held-out inputs differ between seeds beyond the seed")
    baseline, candidate = pairs[0].baseline.inputs, pairs[0].candidate.inputs
    old, new = plain(baseline.config), plain(candidate.config)
    assert isinstance(old, dict)  # Config is a JSON object.
    assert isinstance(new, dict)
    return {
        "baseline_candidate_hash": baseline.candidate_hash,
        "candidate_hash": candidate.candidate_hash,
        "config": {
            "added": {key: new[key] for key in sorted(new) if key not in old},
            "removed": {key: old[key] for key in sorted(old) if key not in new},
            "changed": {
                key: {"from": old[key], "to": new[key]}
                for key in sorted(old)
                if key in new and old[key] != new[key]
            },
        },
        "code_changed": (baseline.source_hash, baseline.dependencies_hash)
        != (candidate.source_hash, candidate.dependencies_hash),
        "source_hash": {
            "baseline": baseline.source_hash,
            "candidate": candidate.source_hash,
        },
        "dependencies_hash": {
            "baseline": baseline.dependencies_hash,
            "candidate": candidate.dependencies_hash,
        },
    }


def observe(final: Decision, metric: Metric, name: str, fidelity: str) -> Facts:
    """State the measured effect as a signed metric delta and interval.

    The improvement-oriented gain is kept beside it. Unbounded endpoints stay
    unbounded.
    """
    gain = final.median_gain
    low, high = final.interval
    sign = -1.0 if metric.direction is Direction.MINIMIZE else 1.0
    ends: list[JSONValue] = [
        None if value is None else sign * value for value in (low, high)
    ]
    if sign < 0:
        ends.reverse()
    return {
        "metric": name,
        "direction": metric.direction,
        "fidelity": fidelity,
        "median_gain": gain,
        "delta": None if gain is None else sign * gain,
        "interval": ends,
        "sample_count": final.sample_count,
    }


def operator_links(
    ledger: Ledger, candidate_hash: str, observation: Facts
) -> list[JSONValue]:
    """Return every operator card bound to this candidate; empty when unlinked.

    Each link lists the card hash and the implementation success, snapshot,
    binding, and outcome records, with the card's recorded prediction next to
    the observation. A missing record is null.
    """
    everything = ledger.activities()
    links: list[JSONValue] = []
    for binding in everything:
        if (
            binding.kind is not ActivityKind.CANDIDATE_BOUND
            or binding.payload.get("candidate_hash") != candidate_hash
        ):
            continue
        history = [
            item for item in everything if item.subject_hash == binding.subject_hash
        ]
        success = _one(
            history, ActivityKind.IMPLEMENTATION_SUCCESS, binding.parent_sequence
        )
        hypothesis = _one(history, ActivityKind.HYPOTHESIS, None)
        outcome = _one(history, ActivityKind.HYPOTHESIS_OUTCOME, None)
        links.append(
            {
                "card_hash": binding.subject_hash,
                "bound_sequence": binding.sequence,
                "implementation_success_sequence": _sequence(success),
                "snapshot_hash": None
                if success is None
                else plain(success.payload.get("snapshot_hash")),
                "outcome_sequence": _sequence(outcome),
                "prediction": _prediction(hypothesis),
                "observation": observation,
            }
        )
    return links


def _sequence(activity: Activity | None) -> int | None:
    return None if activity is None else activity.sequence


def _one(
    history: list[Activity], kind: ActivityKind, sequence: int | None
) -> Activity | None:
    return next(
        (
            item
            for item in history
            if item.kind is kind and (sequence is None or item.sequence == sequence)
        ),
        None,
    )


def _prediction(hypothesis: Activity | None) -> JSONValue:
    card = None if hypothesis is None else hypothesis.payload.get("card")
    prediction = card.get("prediction") if isinstance(card, Mapping) else None
    if not isinstance(prediction, Mapping):
        return None
    return {name: plain(prediction.get(name)) for name in _PREDICTION_FIELDS}
