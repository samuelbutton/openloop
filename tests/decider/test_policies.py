"""Frozen comparisons, fixed-count confirmation, and the original champion anchor."""

from dataclasses import replace

import pytest

from openloop.decider import Calibration, DecisionStage, Pair, Policy, Verdict, decide
from openloop.ledger import Direction, InvalidInputError
from openloop.studies.t0_family import T0Study

from .conftest import scored_pairs

FAMILY_ALPHA = 0.05 / 8


def test_confirmation_waits_for_all_seeds(
    study: T0Study, high_noise: Calibration
) -> None:
    pairs = scored_pairs(study, study.protocol.confirm, (1.0,) * 16)
    assert study.improvement is not None
    partial = decide(
        study.protocol,
        DecisionStage.CONFIRM,
        study.improvement,
        pairs[:15],
        high_noise,
        test_alpha=FAMILY_ALPHA,
    )
    assert partial.verdict is Verdict.INCONCLUSIVE
    assert partial.p_value is None
    result = decide(
        study.protocol,
        DecisionStage.CONFIRM,
        study.improvement,
        pairs,
        high_noise,
        test_alpha=FAMILY_ALPHA,
    )
    assert result.verdict is Verdict.PROMOTE
    assert result.sample_count == 16  # Four inner observations are not four seeds.
    assert result.median_gain == 1
    assert result.interval == (1, 1)
    assert result.p_value == 2**-16
    assert result.test_alpha == FAMILY_ALPHA
    assert len(result.evidence) == 42


def test_minimum_effect_and_null_are_not_improvements(
    study: T0Study, high_noise: Calibration
) -> None:
    protocol = replace(study.protocol, minimum_effect=0.1)
    assert study.improvement is not None
    for effect in (0.0, 0.1, -1.0):
        pairs = scored_pairs(study, protocol.confirm, (effect,) * 16)
        assert (
            decide(
                protocol,
                DecisionStage.CONFIRM,
                study.improvement,
                pairs,
                high_noise,
                test_alpha=FAMILY_ALPHA,
            ).verdict
            is Verdict.DISCARD
        )


def test_autoscientists_second_seed_uses_original_anchor(
    study: T0Study, low_noise: Calibration
) -> None:
    protocol = replace(study.protocol, policy=Policy.AUTOSCIENTISTS)
    pairs = scored_pairs(study, protocol.screen, (low_noise.sigma, 1.0))
    # A better new baseline makes the paired second gain positive, although
    # the second candidate is worse than the original frozen champion.
    second = Pair(
        replace(pairs[1].baseline, metric=replace(pairs[1].baseline.metric, value=2)),
        replace(pairs[1].candidate, metric=replace(pairs[1].candidate.metric, value=1)),
    )
    assert study.improvement is not None
    assert (
        decide(
            protocol, DecisionStage.SCREEN, study.improvement, pairs[:1], low_noise
        ).verdict
        is Verdict.INCONCLUSIVE
    )
    assert (
        decide(
            protocol,
            DecisionStage.SCREEN,
            study.improvement,
            (pairs[0], second),
            low_noise,
        ).verdict
        is Verdict.DISCARD
    )


def test_directions_have_the_same_oriented_gain(study: T0Study) -> None:
    pair = scored_pairs(study, study.protocol.screen, (1.0,))[0]
    maximum = Pair(
        replace(
            pair.baseline,
            metric=replace(pair.baseline.metric, value=0, direction=Direction.MAXIMIZE),
        ),
        replace(
            pair.candidate,
            metric=replace(
                pair.candidate.metric, value=1, direction=Direction.MAXIMIZE
            ),
        ),
    )
    assert maximum.gain == pair.gain == 1


@pytest.mark.parametrize(
    "field",
    [
        "data_hash",
        "evaluator_hash",
        "environment_hash",
        "fidelity",
        "budget_amount",
        "loop_hash",
    ],
)
def test_comparison_context_cannot_change(study: T0Study, field: str) -> None:
    pair = scored_pairs(study, study.protocol.screen, (1.0,))[0]
    value = (
        2 if field == "budget_amount" else "other" if field == "fidelity" else "b" * 64
    )
    with pytest.raises(InvalidInputError, match="differs"):
        Pair(
            pair.baseline,
            replace(
                pair.candidate, inputs=replace(pair.candidate.inputs, **{field: value})
            ),
        )


def test_wrong_schedule_and_repeated_evidence_are_rejected(
    study: T0Study, high_noise: Calibration
) -> None:
    pairs = scored_pairs(study, study.protocol.confirm, (1.0,) * 16)
    assert study.improvement is not None
    for invalid in (pairs[::-1], (pairs[0], pairs[0]), pairs + pairs):
        with pytest.raises(InvalidInputError):
            decide(
                study.protocol,
                DecisionStage.CONFIRM,
                study.improvement,
                invalid,
                high_noise,
                test_alpha=FAMILY_ALPHA,
            )
    with pytest.raises(InvalidInputError, match="family"):
        decide(
            study.protocol,
            DecisionStage.CONFIRM,
            "0" * 64,
            pairs,
            high_noise,
            test_alpha=FAMILY_ALPHA,
        )
    with pytest.raises(InvalidInputError, match="Noise"):
        decide(
            study.protocol,
            DecisionStage.CONFIRM,
            study.improvement,
            pairs,
            Calibration(high_noise.pairs[:3]),
            test_alpha=FAMILY_ALPHA,
        )


def test_smaller_test_set_gives_a_less_strict_threshold(
    study: T0Study, high_noise: Calibration
) -> None:
    pairs = scored_pairs(study, study.protocol.confirm, (1.0,) * 12 + (-1.0,) * 4)
    assert study.improvement is not None
    family = decide(
        study.protocol,
        DecisionStage.CONFIRM,
        study.improvement,
        pairs,
        high_noise,
        test_alpha=FAMILY_ALPHA,
    )
    single_protocol = replace(study.protocol, candidates=(study.improvement,))
    single = decide(
        single_protocol,
        DecisionStage.CONFIRM,
        study.improvement,
        pairs,
        high_noise,
        test_alpha=0.05,
    )
    assert family.p_value == single.p_value
    assert family.verdict is Verdict.DISCARD
    assert single.verdict is Verdict.PROMOTE


def screen_verdict(
    study: T0Study, noise: Calibration, gain: float, **changes: float | Policy
) -> Verdict:
    protocol = replace(study.protocol, **changes)
    assert study.improvement is not None
    pairs = scored_pairs(study, protocol.screen, (gain, gain))
    return decide(
        protocol, DecisionStage.SCREEN, study.improvement, pairs, noise
    ).verdict


def test_screen_margin_boundaries(study: T0Study, low_noise: Calibration) -> None:
    sigma = low_noise.sigma
    assert study.protocol.screen_margin_sigmas == 0.5
    assert screen_verdict(study, low_noise, 0.5 * sigma) is Verdict.DISCARD
    assert screen_verdict(study, low_noise, 0.5 * sigma * 1.01) is Verdict.PROMOTE
    assert screen_verdict(study, low_noise, 0.5 * sigma * 0.99) is Verdict.DISCARD
    # A larger margin discards what the default promotes.
    assert (
        screen_verdict(study, low_noise, 1.5 * sigma, screen_margin_sigmas=2.0)
        is Verdict.DISCARD
    )
    assert (
        screen_verdict(study, low_noise, 2.5 * sigma, screen_margin_sigmas=2.0)
        is Verdict.PROMOTE
    )
    # Margin zero leaves only the strict positive-gain condition.
    assert (
        screen_verdict(study, low_noise, 1e-9, screen_margin_sigmas=0.0)
        is Verdict.PROMOTE
    )
    assert (
        screen_verdict(study, low_noise, 0.0, screen_margin_sigmas=0.0)
        is Verdict.DISCARD
    )
    # The minimum effect still binds when it exceeds the noise margin.
    assert screen_verdict(study, low_noise, 1.0, minimum_effect=1.0) is Verdict.DISCARD


def test_held_out_uses_plain_alpha(study: T0Study, high_noise: Calibration) -> None:
    assert study.improvement is not None
    plan = study.protocol.held_out
    for wins, verdict in ((12, Verdict.VERIFIED), (11, Verdict.DISCARD)):
        gains = (1.0,) * wins + (-1.0,) * (16 - wins)
        result = decide(
            study.protocol,
            DecisionStage.HELD_OUT,
            study.improvement,
            scored_pairs(study, plan, gains),
            high_noise,
        )
        assert result.verdict is verdict
        assert result.test_alpha == 0.05


def test_threshold_is_explicit_and_checked(
    study: T0Study, high_noise: Calibration, low_noise: Calibration
) -> None:
    assert study.improvement is not None
    confirm = scored_pairs(study, study.protocol.confirm, (1.0,) * 16)
    screen = scored_pairs(study, study.protocol.screen, (1.0, 1.0))
    for threshold in (None, 0.0, 0.06, float("nan"), 2**-17):
        with pytest.raises(InvalidInputError):
            decide(
                study.protocol,
                DecisionStage.CONFIRM,
                study.improvement,
                confirm,
                high_noise,
                test_alpha=threshold,
            )
    with pytest.raises(InvalidInputError, match="Only confirmation"):
        decide(
            study.protocol,
            DecisionStage.SCREEN,
            study.improvement,
            screen,
            low_noise,
            test_alpha=0.05,
        )
    screened = decide(
        study.protocol,
        DecisionStage.SCREEN,
        study.improvement,
        screen,
        low_noise,
    )
    assert screened.test_alpha is None
