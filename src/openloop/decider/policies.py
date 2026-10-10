"""Pure policy evaluation over frozen plans and trusted measurements.

Screening is a heuristic. Only fresh, fixed-count confirmation can promote a
candidate formally. Bonferroni bounds family-wise error, hence also FDR, for a
test set fixed independently of confirmation scores. No optional stopping.
The held-out stage tests one pre-selected candidate at plain alpha.
"""

from statistics import fmean

from openloop.ledger import InvalidInputError
from openloop.ledger.validation import coerce_enum, finite_number

from .models import (
    POLICY_RULES,
    Decision,
    DecisionStage,
    NoiseKind,
    Pair,
    Policy,
    Protocol,
    Verdict,
)
from .statistics import (
    Calibration,
    autoscientists_gate,
    median_gain,
    median_interval,
    sign_p_value,
)


def validate_calibration(
    protocol: Protocol, stage: DecisionStage, noise: Calibration
) -> None:
    """Require locked baseline noise at the declared validation fidelity."""
    plan = protocol.screen if stage is DecisionStage.SCREEN else protocol.confirm
    seeds = (
        protocol.screen_noise_seeds
        if stage is DecisionStage.SCREEN
        else protocol.confirm_noise_seeds
    )
    if (
        len(noise.pairs) != POLICY_RULES[protocol.policy].calibration_pairs
        or noise.kind is not NoiseKind.DISTINCT_SEED
        or tuple(sample.inputs.seed for sample in noise.samples) != seeds
        or noise.samples[0].context_hash != plan.context_hash
        or noise.samples[0].inputs.candidate_hash != plan.template.candidate_hash
    ):
        raise InvalidInputError(
            "Noise must be frozen from the declared baseline seed pairs"
        )


def decide(
    protocol: Protocol,
    stage: DecisionStage,
    candidate: str,
    pairs: tuple[Pair, ...],
    noise: Calibration,
    *,
    test_alpha: float | None = None,
) -> Decision:
    """Return a pure decision. Held-out authorization belongs to the coordinator.

    Confirmation requires the caller's Bonferroni threshold, at most alpha. Screen
    takes none. Held-out tests one pre-selected candidate at the protocol alpha.
    """
    if candidate not in protocol.candidates:
        raise InvalidInputError("Candidate is outside the frozen comparison family")
    stage = coerce_enum(DecisionStage, stage, "decision stage")
    pairs = tuple(pairs)
    plan = {
        DecisionStage.SCREEN: protocol.screen,
        DecisionStage.CONFIRM: protocol.confirm,
        DecisionStage.HELD_OUT: protocol.held_out,
    }[stage]
    plan.validate(pairs, candidate)
    threshold = _threshold(protocol, stage, plan.seeds, test_alpha)
    validate_calibration(protocol, stage, noise)
    calibration_ids = {sample.attempt_id for sample in noise.samples}
    if any(
        sample.attempt_id in calibration_ids
        for pair in pairs
        for sample in (pair.baseline, pair.candidate)
    ):
        raise InvalidInputError("Calibration cannot count as comparison evidence")
    gains = tuple(pair.gain for pair in pairs)
    evidence = tuple(
        sample.reference for pair in pairs for sample in (pair.baseline, pair.candidate)
    ) + tuple(sample.reference for sample in noise.samples)
    verdict = Verdict.INCONCLUSIVE
    reason = "The declared seed schedule is incomplete"
    p_value = None
    interval = (None, None)
    if stage is DecisionStage.SCREEN:
        if protocol.policy is Policy.AUTOSCIENTISTS and pairs:
            # The new candidate score is compared to the ORIGINAL champion.
            second_gain = (
                pairs[1].candidate.score - pairs[0].baseline.score
                if len(pairs) == 2
                else None
            )
            verdict = autoscientists_gate(
                gains[0], noise.sigma, second_gain=second_gain
            )
            reason = "AutoScientists A.6 screen; confirmation is still required"
        elif len(pairs) == len(plan.seeds):
            margin = protocol.screen_margin_sigmas * noise.sigma
            verdict = (
                Verdict.PROMOTE
                if fmean(gains) > max(protocol.minimum_effect, margin)
                else Verdict.DISCARD
            )
            reason = (
                "Mean screen gain must exceed minimum effect and the declared "
                "multiple of per-run noise"
            )
    elif threshold is not None and len(pairs) == len(plan.seeds):
        p_value = sign_p_value(gains, minimum_effect=protocol.minimum_effect)
        interval = median_interval(gains, alpha=threshold)
        if p_value <= threshold:
            verdict = (
                Verdict.VERIFIED if stage is DecisionStage.HELD_OUT else Verdict.PROMOTE
            )
            reason = "Exact paired sign test passes the recorded threshold"
        else:
            verdict = Verdict.DISCARD
            reason = "Exact paired sign test does not pass the recorded threshold"
    return Decision(
        protocol.hash,
        candidate,
        stage,
        verdict,
        reason,
        evidence,
        len(pairs),
        median_gain(gains),
        interval,
        p_value,
        noise.sigma,
        threshold,
    )


def _threshold(
    protocol: Protocol,
    stage: DecisionStage,
    seeds: tuple[int, ...],
    test_alpha: float | None,
) -> float | None:
    """Return the sign-test threshold this stage must use, or None for screening."""
    if stage is not DecisionStage.CONFIRM:
        if test_alpha is not None:
            raise InvalidInputError("Only confirmation takes a Bonferroni threshold")
        return protocol.alpha if stage is DecisionStage.HELD_OUT else None
    if test_alpha is None:
        raise InvalidInputError("Confirmation requires an explicit test threshold")
    test_alpha = finite_number(test_alpha, "Test threshold")
    if not 0 < test_alpha <= protocol.alpha:
        raise InvalidInputError("Test threshold must be in (0, alpha]")
    if 2 ** -len(seeds) > test_alpha:
        raise InvalidInputError("Confirmation needs enough seeds to reject a null")
    return test_alpha
