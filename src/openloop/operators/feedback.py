"""Trusted coordinator bindings and measured idea failures. No model authority."""

from collections.abc import Mapping

from openloop.decider import Decision, DecisionStage, Pair, Sample, Verdict
from openloop.ledger import (
    Activity,
    ActivityKind,
    ConflictError,
    ExperimentInputs,
    HypothesisOutcome,
    InvalidInputError,
    Ledger,
    Status,
    content_hash,
)
from openloop.loops import Phase

from .implementer import ImplementedCandidate
from .outcomes import record_outcome_for_hash


def _success(ledger: Ledger, candidate: ImplementedCandidate) -> Activity:
    success = next(
        (
            event
            for event in ledger.activities(subject_hash=candidate.card_hash)
            if event.sequence == candidate.success_sequence
        ),
        None,
    )
    if (
        success is None
        or success.kind is not ActivityKind.IMPLEMENTATION_SUCCESS
        or success.payload.get("snapshot_hash") != candidate.snapshot_hash
    ):
        raise InvalidInputError(
            "Feedback requires the recorded successful implementation"
        )
    return success


def bind_candidate(
    ledger: Ledger, candidate: ImplementedCandidate, inputs: ExperimentInputs
) -> Activity:
    """Bind a produced source snapshot before confirmation starts.

    This association is a trusted coordinator assertion; different loop adapters
    derive their source and configuration identities in different ways.
    """
    _success(ledger, candidate)
    history = ledger.activities(subject_hash=candidate.card_hash)
    hypothesis = next(
        event for event in history if event.kind is ActivityKind.HYPOTHESIS
    )
    card = hypothesis.payload["card"]
    if (
        not isinstance(card, Mapping)
        or inputs.loop_hash != card.get("loop_hash")
        or inputs.phase != Phase.SCREEN
    ):
        raise InvalidInputError("Candidate binding requires this loop's screen inputs")
    prior = [event for event in history if event.kind is ActivityKind.CANDIDATE_BOUND]
    if prior and prior[0].payload.get("candidate_hash") != inputs.candidate_hash:
        raise InvalidInputError("Implementation is already bound to another candidate")
    if prior:
        return prior[0]
    if any(
        run.inputs.phase == Phase.CONFIRM
        for run in ledger.query_all(candidate_hash=inputs.candidate_hash)
    ):
        raise InvalidInputError("Bind the implementation before confirmation starts")
    return ledger.record_activity(
        ActivityKind.CANDIDATE_BOUND,
        candidate.card_hash,
        {"candidate_hash": inputs.candidate_hash, "inputs": inputs},
        key=f"candidate-binding:{candidate.card_hash}",
        parent_sequence=candidate.success_sequence,
    )


def record_idea_failure(
    ledger: Ledger, candidate: ImplementedCandidate, decision: Decision
) -> Activity:
    """Record a trusted measured confirmation discard, never a smoke failure.

    Unmeasured allocator drops, inconclusive decisions, and held-out decisions
    are excluded. The caller owns the statistical protocol and decider authority.

    `decide` lists the 2 * sample_count paired comparison runs first, in
    baseline/candidate order, then the noise-calibration runs. The pairs are
    checked as paired evidence. The calibration runs must be distinct, successful
    confirmation-phase baseline runs in the pairs' comparison context.

    Recording an idea failure also records the card's discarded outcome from the
    same decision, so the two records cannot disagree. The call is idempotent:
    repeating it completes a missing outcome after an interruption.
    """
    _success(ledger, candidate)
    if (
        decision.verdict is not Verdict.DISCARD
        or decision.stage is not DecisionStage.CONFIRM
        or decision.sample_count < 1
        or not decision.evidence
    ):
        raise InvalidInputError("Idea failure requires a measured confirmation discard")
    history = ledger.activities(subject_hash=candidate.card_hash)
    binding = next(
        (event for event in history if event.kind is ActivityKind.CANDIDATE_BOUND), None
    )
    if (
        binding is None
        or binding.payload.get("candidate_hash") != decision.candidate_hash
    ):
        raise InvalidInputError(
            "Decision does not match the bound implemented candidate"
        )
    decision_hash = content_hash(decision)
    prior = next(
        (event for event in history if event.kind is ActivityKind.HYPOTHESIS_OUTCOME),
        None,
    )
    if prior is not None and (
        prior.payload.get("outcome") != HypothesisOutcome.DISCARDED
        or prior.payload.get("decision_hash") != decision_hash
    ):
        raise ConflictError("The recorded outcome differs from this discarded decision")
    card = next(
        event for event in history if event.kind is ActivityKind.HYPOTHESIS
    ).payload["card"]
    if not isinstance(card, Mapping) or not isinstance(card.get("baseline"), str):
        raise InvalidInputError("Missing recorded baseline")
    baseline_ref = card["baseline"]
    assert isinstance(baseline_ref, str)
    baseline = ledger.get(baseline_ref)
    prediction = card.get("prediction")
    if not isinstance(prediction, Mapping) or not isinstance(
        prediction.get("metric"), str
    ):
        raise InvalidInputError("Missing recorded prediction metric")
    metric_name = prediction["metric"]
    assert isinstance(metric_name, str)
    pair_count = decision.sample_count * 2
    if len(decision.evidence) <= pair_count:
        raise InvalidInputError("Decision lists no noise-calibration evidence")
    runs = tuple(ledger.get(reference) for reference in decision.evidence)
    pair_runs, calibration_runs = runs[:pair_count], runs[pair_count:]
    if len({run.attempt_id for run in runs}) != len(runs):
        raise InvalidInputError("Decision requires distinct paired evidence")
    if any(
        run.status is not Status.SUCCEEDED
        or run.result is None
        or run.reused_from is not None
        or run.inputs.phase != Phase.CONFIRM
        for run in runs
    ):
        raise InvalidInputError(
            "Idea failure requires successful confirmation evidence"
        )
    context: str | None = None
    for offset in range(0, pair_count, 2):
        pair = Pair(
            Sample.from_run(pair_runs[offset], metric_name),
            Sample.from_run(pair_runs[offset + 1], metric_name),
        )
        context = pair.baseline.context_hash
        if (
            pair.baseline.inputs.candidate_hash != baseline.inputs.candidate_hash
            or pair.candidate.inputs.candidate_hash != decision.candidate_hash
            or pair.baseline.inputs.loop_hash != card.get("loop_hash")
        ):
            raise InvalidInputError(
                "Decision comparison differs from the bound hypothesis"
            )
    for run in calibration_runs:
        sample = Sample.from_run(run, metric_name)
        if (
            sample.inputs.candidate_hash != baseline.inputs.candidate_hash
            or sample.inputs.loop_hash != card.get("loop_hash")
            or sample.context_hash != context
        ):
            raise InvalidInputError(
                "Noise calibration differs from the protocol baseline"
            )
    if any(
        metric.split != "validation"
        for run in runs
        if run.result
        for metric in run.result.metrics.values()
    ):
        raise InvalidInputError("Held-out evidence must not enter idea feedback")
    failure = ledger.record_activity(
        ActivityKind.IDEA_FAILURE,
        candidate.card_hash,
        {"decision": decision, "decision_hash": decision_hash},
        key=f"idea-failure:{candidate.card_hash}:{decision.protocol_hash}",
        parent_sequence=binding.sequence,
    )
    record_outcome_for_hash(ledger, candidate.card_hash, decision)
    return failure
