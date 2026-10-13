"""An allocator run as the screen: validation, ordering, and error-budget split."""

import asyncio
from dataclasses import replace

import pytest

from openloop.decider import (
    Campaign,
    ConfirmationScope,
    DecisionStage,
    Policy,
    Verdict,
)
from openloop.ledger import InvalidInputError, InvalidTransitionError
from openloop.studies.t0_family import Arm, T0Study

from .conftest import scored_pairs

ALLOCATOR_SEEDS = (100, 101, 102)


def setup(scope: ConfirmationScope) -> tuple[T0Study, Campaign]:
    study = T0Study(
        99,
        gain=1.0,
        worsening=3.0,
        arm=Arm("a", policy=Policy.ALLOCATOR, confirmation_scope=scope),
    )
    low = asyncio.run(study.calibration(DecisionStage.SCREEN))
    high = asyncio.run(study.calibration(DecisionStage.CONFIRM))
    return study, Campaign(study.protocol, low, high)


def record(campaign: Campaign, candidate: str, *, promoted: bool, **changes: object):
    arguments = {
        "promoted": promoted,
        "evidence": ("run-1",) if promoted else (),
        "seeds": ALLOCATOR_SEEDS,
        "source": "a" * 64,
    } | changes
    return campaign.record_external_screen(candidate, **arguments)  # pyright: ignore[reportArgumentType]


def test_recorded_screen_is_a_decision_with_references_and_no_statistics() -> None:
    study, campaign = setup(ConfirmationScope.FAMILY)
    assert study.improvement is not None
    kept = record(campaign, study.improvement, promoted=True, evidence=("r1", "r2"))
    assert (kept.stage, kept.verdict) == (DecisionStage.SCREEN, Verdict.PROMOTE)
    assert kept.evidence == ("r1", "r2")
    assert (kept.p_value, kept.test_alpha, kept.sample_count) == (None, None, 0)
    other = next(c for c in study.configs if c != study.improvement)
    assert record(campaign, other, promoted=False).verdict is Verdict.DISCARD
    assert len(campaign.decisions) == 2


def test_only_the_allocator_policy_takes_an_external_screen() -> None:
    study = T0Study(99, gain=1.0, worsening=3.0)
    campaign = Campaign(
        study.protocol,
        asyncio.run(study.calibration(DecisionStage.SCREEN)),
        asyncio.run(study.calibration(DecisionStage.CONFIRM)),
    )
    assert study.improvement is not None
    with pytest.raises(InvalidTransitionError, match="allocator policy"):
        record(campaign, study.improvement, promoted=True)
    assert campaign.decisions == ()


def test_allocator_policy_refuses_a_native_screen() -> None:
    study, campaign = setup(ConfirmationScope.FAMILY)
    assert study.improvement is not None
    pairs = asyncio.run(study.pairs(study.improvement, study.protocol.screen))
    with pytest.raises(InvalidTransitionError, match="allocator screen"):
        campaign.evaluate(DecisionStage.SCREEN, study.improvement, pairs)
    assert campaign.decisions == ()


def test_invalid_records_are_rejected_without_changing_state() -> None:
    study, campaign = setup(ConfirmationScope.FAMILY)
    assert study.improvement is not None
    protocol = study.protocol
    with pytest.raises(InvalidInputError, match="outside"):
        record(campaign, "f" * 64, promoted=False)
    with pytest.raises(InvalidInputError, match="references"):
        record(campaign, study.improvement, promoted=True, evidence=())
    for reserved in (
        protocol.confirm.seeds[0],
        protocol.held_out.seeds[-1],
        protocol.confirm_noise_seeds[3],
    ):
        with pytest.raises(InvalidInputError, match="disjoint"):
            record(campaign, study.improvement, promoted=True, seeds=(100, reserved))
    with pytest.raises(InvalidInputError, match="seeds"):
        record(campaign, study.improvement, promoted=True, seeds=())
    assert campaign.decisions == ()
    record(campaign, study.improvement, promoted=True)
    with pytest.raises(InvalidTransitionError, match="once"):
        record(campaign, study.improvement, promoted=False)
    assert campaign.decisions[0].verdict is Verdict.PROMOTE


def test_no_screen_may_be_recorded_after_any_confirmation() -> None:
    study, campaign = setup(ConfirmationScope.FAMILY)
    assert study.improvement is not None
    record(campaign, study.improvement, promoted=True)
    confirm = scored_pairs(study, study.protocol.confirm, (1.0,) * 16)
    campaign.evaluate(DecisionStage.CONFIRM, study.improvement, confirm[:3])
    other = next(c for c in study.configs if c != study.improvement)
    with pytest.raises(InvalidTransitionError, match="precede"):
        record(campaign, other, promoted=False)


def test_family_scope_splits_alpha_by_the_whole_family() -> None:
    study, campaign = setup(ConfirmationScope.FAMILY)
    assert campaign.confirmation_alpha == study.protocol.alpha / len(study.configs)


def test_screened_scope_waits_for_every_screen_then_splits_by_survivors() -> None:
    study, campaign = setup(ConfirmationScope.SCREENED)
    assert study.improvement is not None
    members = list(study.configs)
    confirm = scored_pairs(study, study.protocol.confirm, (1.0,) * 16)
    record(campaign, study.improvement, promoted=True)
    with pytest.raises(InvalidTransitionError, match="Resolve every screen"):
        campaign.evaluate(DecisionStage.CONFIRM, study.improvement, confirm)
    survivors = {study.improvement, members[2]}
    for member in members:
        if member != study.improvement:
            record(campaign, member, promoted=member in survivors)
    assert campaign.confirmation_alpha is None
    decision = campaign.evaluate(DecisionStage.CONFIRM, study.improvement, confirm)
    assert campaign.confirmation_set == tuple(sorted(survivors))
    assert decision.test_alpha == study.protocol.alpha / 2
    assert decision.verdict is Verdict.PROMOTE


def test_allocator_protocol_rules_and_identity() -> None:
    study, _ = setup(ConfirmationScope.SCREENED)
    protocol = study.protocol
    with pytest.raises(InvalidInputError, match="ranking"):
        replace(protocol, confirm_top_k=1)
    with pytest.raises(InvalidInputError, match="margin"):
        replace(protocol, screen_margin_sigmas=2.0)
    assert protocol.hash != replace(protocol, policy=Policy.NOISE_AWARE).hash
