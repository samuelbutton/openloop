"""Stage order, evidence immutability, and one-use held-out permission."""

import asyncio
from dataclasses import replace

import pytest

from openloop.decider import (
    Calibration,
    Campaign,
    ConfirmationScope,
    DecisionStage,
    Pair,
    Policy,
    StagePlan,
    Verdict,
)
from openloop.ledger import InvalidInputError, InvalidTransitionError
from openloop.studies.t0_family import Arm, T0Study

from .conftest import scored_pairs


def selected_campaign(study: T0Study, low: Calibration, high: Calibration) -> Campaign:
    """Resolve the whole family using actual T0 draws."""
    campaign = Campaign(study.protocol, low, high)
    for candidate in study.configs:
        screen = asyncio.run(study.pairs(candidate, study.protocol.screen))
        if (
            campaign.evaluate(DecisionStage.SCREEN, candidate, screen).verdict
            is Verdict.PROMOTE
        ):
            confirm = asyncio.run(study.pairs(candidate, study.protocol.confirm))
            campaign.evaluate(DecisionStage.CONFIRM, candidate, confirm)
    assert study.improvement is not None
    campaign.close_selection(study.improvement)
    return campaign


def test_stages_and_rejected_evidence_do_not_mutate_state(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    campaign = Campaign(study.protocol, low_noise, high_noise)
    assert study.improvement is not None
    confirm = scored_pairs(study, study.protocol.confirm, (1.0,) * 16)
    with pytest.raises(InvalidTransitionError, match="screen"):
        campaign.evaluate(DecisionStage.CONFIRM, study.improvement, confirm)
    assert campaign.decisions == ()
    screen = scored_pairs(study, study.protocol.screen, (1.0, 1.0))
    campaign.evaluate(DecisionStage.SCREEN, study.improvement, screen)
    prefix = campaign.evaluate(DecisionStage.CONFIRM, study.improvement, confirm[:15])
    assert prefix.verdict is Verdict.INCONCLUSIVE
    before = campaign.decisions
    altered = replace(
        confirm[0].candidate, metric=replace(confirm[0].candidate.metric, value=-2)
    )
    with pytest.raises(InvalidInputError, match="prefix"):
        campaign.evaluate(
            DecisionStage.CONFIRM,
            study.improvement,
            (Pair(confirm[0].baseline, altered), *confirm[1:]),
        )
    assert campaign.decisions == before
    assert (
        campaign.evaluate(DecisionStage.CONFIRM, study.improvement, confirm).verdict
        is Verdict.PROMOTE
    )
    with pytest.raises(InvalidTransitionError, match="revised"):
        campaign.evaluate(DecisionStage.CONFIRM, study.improvement, confirm)
    with pytest.raises(InvalidTransitionError, match="family"):
        campaign.close_selection(study.improvement)
    assert campaign.selected is None


@pytest.mark.parametrize("policy", list(Policy))
def test_both_policies_verify_only_after_selection_closes(policy: Policy) -> None:
    study = T0Study(99, gain=1.0, worsening=3.0, arm=Arm("p", policy=policy))
    low = asyncio.run(study.calibration(DecisionStage.SCREEN))
    high = asyncio.run(study.calibration(DecisionStage.CONFIRM))
    campaign = Campaign(study.protocol, low, high)
    with pytest.raises(InvalidTransitionError, match="closed"):
        asyncio.run(campaign.finalize(study.pairs))
    assert not campaign.final_consumed
    campaign = selected_campaign(study, low, high)
    assert campaign.final is None
    assert study.improvement is not None
    with pytest.raises(InvalidTransitionError, match="closed"):
        campaign.evaluate(DecisionStage.SCREEN, study.improvement, ())
    final = asyncio.run(campaign.finalize(study.pairs))
    assert final.verdict is Verdict.VERIFIED
    assert campaign.final is final
    assert campaign.final_consumed
    with pytest.raises(InvalidTransitionError, match="unused"):
        asyncio.run(campaign.finalize(study.pairs))


@pytest.mark.parametrize("failure", ["crash", "wrong_split", "partial"])
def test_failed_final_batch_consumes_permission(
    study: T0Study, low_noise: Calibration, high_noise: Calibration, failure: str
) -> None:
    campaign = selected_campaign(study, low_noise, high_noise)

    async def evaluator(candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        if failure == "crash":
            raise RuntimeError("Final evaluator failed")
        pairs = await study.pairs(
            candidate, plan if failure == "partial" else study.protocol.confirm
        )
        return pairs[:1] if failure == "partial" else pairs

    with pytest.raises(RuntimeError if failure == "crash" else InvalidInputError):
        asyncio.run(campaign.finalize(evaluator))
    assert campaign.final_consumed
    assert campaign.final is None
    with pytest.raises(InvalidTransitionError, match="unused"):
        asyncio.run(campaign.finalize(study.pairs))


def test_concurrent_final_calls_invoke_only_one_evaluator(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    campaign = selected_campaign(study, low_noise, high_noise)

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def evaluator(candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
            entered.set()
            await release.wait()
            return await study.pairs(candidate, plan)

        pending = asyncio.create_task(campaign.finalize(evaluator))
        await entered.wait()
        assert campaign.final is None
        with pytest.raises(InvalidTransitionError, match="unused"):
            await campaign.finalize(evaluator)
        release.set()
        assert (await pending).verdict is Verdict.VERIFIED

    asyncio.run(scenario())


def test_final_cannot_relabel_a_validation_attempt(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    campaign = selected_campaign(study, low_noise, high_noise)
    assert study.improvement is not None
    validation = asyncio.run(study.pairs(study.improvement, study.protocol.confirm))

    async def reused(candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        final = await study.pairs(candidate, plan)
        first = Pair(
            final[0].baseline,
            replace(final[0].candidate, attempt_id=validation[0].candidate.attempt_id),
        )
        return (first, *final[1:])

    with pytest.raises(InvalidInputError, match="conflicting"):
        asyncio.run(campaign.finalize(reused))
    assert campaign.final_consumed
    assert campaign.final is None


def screened_campaign(
    study: T0Study,
    low: Calibration,
    high: Calibration,
    gains: dict[str, float],
    *,
    top_k: int | None = None,
) -> Campaign:
    """Screen the whole family with controlled mean gains; unlisted ones get zero."""
    protocol = replace(
        study.protocol,
        confirmation_scope=ConfirmationScope.SCREENED,
        confirm_top_k=top_k,
    )
    campaign = Campaign(protocol, low, high)
    for candidate in study.configs:
        gain = gains.get(candidate, 0.0)
        pairs = scored_pairs(study, protocol.screen, (gain, gain), candidate)
        campaign.evaluate(DecisionStage.SCREEN, candidate, pairs)
    return campaign


def test_screened_scope_refuses_confirmation_before_screens_resolve(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    protocol = replace(study.protocol, confirmation_scope=ConfirmationScope.SCREENED)
    campaign = Campaign(protocol, low_noise, high_noise)
    first, *_ = study.configs
    campaign.evaluate(
        DecisionStage.SCREEN,
        first,
        scored_pairs(study, protocol.screen, (1.0, 1.0), first),
    )
    confirm = scored_pairs(study, protocol.confirm, (1.0,) * 16, first)
    before = campaign.decisions
    with pytest.raises(InvalidTransitionError, match="Resolve every screen"):
        campaign.evaluate(DecisionStage.CONFIRM, first, confirm)
    assert campaign.decisions == before
    assert campaign.confirmation_set is None
    assert campaign.confirmation_alpha is None
    with pytest.raises(InvalidTransitionError, match="Resolve"):
        campaign.close_selection()


def test_screened_scope_freezes_set_and_threshold(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    first, second, third, *_ = study.configs
    campaign = screened_campaign(
        study, low_noise, high_noise, {first: 1.0, second: 1.0, third: 1.0}
    )
    confirm = scored_pairs(study, study.protocol.confirm, (1.0,) * 16, first)
    decision = campaign.evaluate(DecisionStage.CONFIRM, first, confirm)
    assert campaign.confirmation_set == tuple(sorted((first, second, third)))
    assert campaign.confirmation_alpha == 0.05 / 3
    assert decision.test_alpha == 0.05 / 3
    assert decision.verdict is Verdict.PROMOTE
    # 12 of 16 passes at alpha / 3 only if p <= 0.0167; it is 0.0384, so it fails.
    weak = scored_pairs(
        study, study.protocol.confirm, (1.0,) * 12 + (-1.0,) * 4, second
    )
    assert (
        campaign.evaluate(DecisionStage.CONFIRM, second, weak).verdict
        is Verdict.DISCARD
    )
    # A candidate that failed screening is never added to the frozen set.
    other = next(item for item in study.configs if item not in (first, second, third))
    with pytest.raises(InvalidTransitionError, match="promoted screen"):
        campaign.evaluate(
            DecisionStage.CONFIRM,
            other,
            scored_pairs(study, study.protocol.confirm, (1.0,) * 16, other),
        )
    assert campaign.confirmation_alpha == 0.05 / 3


def test_top_k_ranks_by_mean_gain_and_breaks_ties_by_hash(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    tied_low, tied_high = sorted(list(study.configs)[:2])
    best = list(study.configs)[2]
    campaign = screened_campaign(
        study,
        low_noise,
        high_noise,
        {best: 2.0, tied_low: 1.0, tied_high: 1.0},
        top_k=2,
    )
    confirm = scored_pairs(study, study.protocol.confirm, (1.0,) * 16, best)
    decision = campaign.evaluate(DecisionStage.CONFIRM, best, confirm)
    assert campaign.confirmation_set == (best, tied_low)
    assert decision.test_alpha == 0.05 / 2
    dropped = next(
        item
        for item in campaign.decisions
        if item.candidate_hash == tied_high and item.stage is DecisionStage.CONFIRM
    )
    assert dropped.verdict is Verdict.DISCARD
    assert "top-k" in dropped.reason
    assert dropped.sample_count == 0
    assert dropped.p_value is None
    assert dropped.test_alpha is None
    with pytest.raises(InvalidTransitionError, match="revised"):
        campaign.evaluate(
            DecisionStage.CONFIRM,
            tied_high,
            scored_pairs(study, study.protocol.confirm, (1.0,) * 16, tied_high),
        )
    campaign.evaluate(
        DecisionStage.CONFIRM,
        tied_low,
        scored_pairs(study, study.protocol.confirm, (1.0,) * 16, tied_low),
    )
    campaign.close_selection(best)
    assert campaign.selected == best


@pytest.mark.parametrize("scope", list(ConfirmationScope))
def test_empty_confirmation_set_closes_with_no_winner(
    study: T0Study,
    low_noise: Calibration,
    high_noise: Calibration,
    scope: ConfirmationScope,
) -> None:
    protocol = replace(study.protocol, confirmation_scope=scope)
    campaign = Campaign(protocol, low_noise, high_noise)
    with pytest.raises(InvalidTransitionError, match="Resolve"):
        campaign.close_selection()
    for candidate in study.configs:
        campaign.evaluate(
            DecisionStage.SCREEN,
            candidate,
            scored_pairs(study, protocol.screen, (0.0, 0.0), candidate),
        )
    campaign.close_selection()
    assert campaign.closed
    assert campaign.selected is None
    if scope is ConfirmationScope.SCREENED:
        assert campaign.confirmation_set == ()
    with pytest.raises(InvalidTransitionError, match="closed"):
        campaign.close_selection()
    with pytest.raises(InvalidTransitionError, match="closed"):
        asyncio.run(campaign.finalize(study.pairs))
    assert not campaign.final_consumed


def test_no_winner_close_is_refused_when_a_candidate_is_confirmed(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    first = next(iter(study.configs))
    campaign = screened_campaign(study, low_noise, high_noise, {first: 1.0})
    campaign.evaluate(
        DecisionStage.CONFIRM,
        first,
        scored_pairs(study, study.protocol.confirm, (1.0,) * 16, first),
    )
    with pytest.raises(InvalidTransitionError, match="confirmed candidate"):
        campaign.close_selection()
    assert not campaign.closed


def test_family_scope_threshold_is_alpha_over_family_size(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    campaign = Campaign(study.protocol, low_noise, high_noise)
    assert campaign.confirmation_set is None
    assert campaign.confirmation_alpha == 0.05 / 8
    assert study.improvement is not None
    campaign.evaluate(
        DecisionStage.SCREEN,
        study.improvement,
        scored_pairs(study, study.protocol.screen, (1.0, 1.0)),
    )
    decision = campaign.evaluate(
        DecisionStage.CONFIRM,
        study.improvement,
        scored_pairs(study, study.protocol.confirm, (1.0,) * 16),
    )
    assert decision.test_alpha == 0.05 / 8


def test_screened_campaign_verifies_with_plain_alpha_on_held_out(
    study: T0Study, low_noise: Calibration, high_noise: Calibration
) -> None:
    first = next(iter(study.configs))
    campaign = screened_campaign(study, low_noise, high_noise, {first: 1.0})
    campaign.evaluate(
        DecisionStage.CONFIRM,
        first,
        scored_pairs(study, study.protocol.confirm, (1.0,) * 16, first),
    )
    campaign.close_selection(first)

    batch = scored_pairs(
        study, study.protocol.held_out, (1.0,) * 12 + (-1.0,) * 4, first
    )

    async def twelve_of_sixteen(candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        return batch

    final = asyncio.run(campaign.finalize(twelve_of_sixteen))
    assert final.verdict is Verdict.VERIFIED
    assert final.test_alpha == 0.05
