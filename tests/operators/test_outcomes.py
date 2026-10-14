"""One outcome per card, linked to decisions, and prediction calibration."""

import asyncio
from collections.abc import Mapping
from dataclasses import replace

import pytest

from openloop.decider import Campaign, DecisionStage, Verdict
from openloop.ledger import (
    ActivityKind,
    ConflictError,
    HypothesisOutcome,
    InvalidInputError,
    Ledger,
    content_hash,
)
from openloop.operators import calibration_summary, record_idea_failure, record_outcome
from openloop.studies.t0_family import T0Study

from .conftest import SandboxStandIn, bind_card, discarded_trial, make_world


def test_outcomes_and_calibration_summary(
    ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    trial, discarded = discarded_trial()
    study = T0Study(trial, gain=1.0, worsening=3.0)
    world = make_world(ledger, study)
    protocol = study.protocol
    improvement = study.improvement
    assert improvement is not None
    campaign = Campaign(protocol, world.screen_noise, world.confirm_noise)
    screens = {
        member: campaign.evaluate(
            DecisionStage.SCREEN, member, world.pairs(member, protocol.screen)
        )
        for member in protocol.candidates
    }
    assert screens[improvement].verdict is Verdict.PROMOTE
    others = [m for m in protocol.candidates if m not in (improvement, discarded)]
    dropped = next(m for m in others if screens[m].verdict is Verdict.DISCARD)
    unfinished = next(m for m in others if m != dropped)

    # Bind before any confirmation run exists for a candidate.
    interval = (-1.5, -0.5)
    good = bind_card(
        world, sandbox, improvement, probability=0.9, interval=interval, title="Real"
    )
    bad = bind_card(world, sandbox, discarded, probability=0.8, title="Null")
    drop = bind_card(world, sandbox, dropped, probability=0.3, title="Dropped")
    pending = bind_card(world, sandbox, unfinished, probability=0.5, title="Unfinished")

    confirms = {}
    for member in protocol.candidates:
        if screens[member].verdict is Verdict.PROMOTE:
            confirms[member] = campaign.evaluate(
                DecisionStage.CONFIRM, member, world.pairs(member, protocol.confirm)
            )
    assert confirms[improvement].verdict is Verdict.PROMOTE
    assert confirms[discarded].verdict is Verdict.DISCARD
    campaign.close_selection(improvement)
    final = asyncio.run(campaign.finalize(study.pairs))
    assert final.verdict is Verdict.VERIFIED
    assert final.median_gain is not None

    # Non-terminal and mismatched decisions are rejected without a record.
    before = ledger.activities()
    for case in (
        confirms[improvement],  # confirmation promotion awaits held-out
        screens[improvement],  # a screen promotion is not terminal
        final,  # belongs to the improvement, not the dropped candidate's card
    ):
        with pytest.raises(InvalidInputError):
            record_outcome(ledger, drop.card, case)
    assert ledger.activities() == before

    won = record_outcome(ledger, good.card, final)
    lost = record_outcome(ledger, bad.card, confirms[discarded])
    skipped = record_outcome(ledger, drop.card, screens[dropped])
    unresolved = replace(
        confirms[unfinished] if unfinished in confirms else screens[unfinished],
        stage=DecisionStage.CONFIRM,
        verdict=Verdict.INCONCLUSIVE,
    )  # mapping check only; the schedule was not truly cut short
    record_outcome(ledger, pending.card, unresolved)
    assert [e.payload["outcome"] for e in (won, lost, skipped)] == [
        HypothesisOutcome.VERIFIED,
        HypothesisOutcome.DISCARDED,
        HypothesisOutcome.SCREEN_DROPPED,
    ]
    assert won.payload["decision_hash"] == content_hash(final)
    assert skipped.payload["decision_hash"] == content_hash(screens[dropped])
    assert won.payload["protocol_hash"] == protocol.hash
    hypothesis = ledger.activities(subject_hash=good.card.hash)[0]
    assert won.parent_sequence == hypothesis.sequence
    observed = won.payload["observed"]
    assert isinstance(observed, Mapping)
    # The metric is minimized, so a positive gain is a negative delta.
    assert observed["delta"] == -final.median_gain
    assert observed["median_gain"] == final.median_gain
    predicted = won.payload["predicted"]
    assert isinstance(predicted, Mapping)
    assert predicted["probability_of_improvement"] == 0.9
    assert predicted["delta"] == -0.1
    assert record_outcome(ledger, good.card, final) == won
    with pytest.raises(ConflictError):
        record_outcome(ledger, good.card, replace(final, verdict=Verdict.DISCARD))

    summary = calibration_summary(ledger)
    assert dict(summary.counts) == {
        "verified": 1,
        "discarded": 1,
        "inconclusive": 1,
        "screen_dropped": 1,
    }
    lost_gain = confirms[discarded].median_gain
    assert lost_gain is not None
    scored = [(0.9, final.median_gain), (0.8, lost_gain)]
    assert summary.scored == 2
    assert summary.brier_score == pytest.approx(
        sum((p - (1.0 if gain > 0 else 0.0)) ** 2 for p, gain in scored) / 2
    )
    # predicted (-0.1) minus observed delta (-gain)
    assert summary.mean_signed_error == pytest.approx(
        sum(-0.1 + gain for _, gain in scored) / 2
    )
    low, high = interval
    assert (summary.interval_covered, summary.interval_total) == (
        int(low <= -final.median_gain <= high),
        1,
    )


def test_idea_failure_and_outcome_cannot_disagree(
    ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    from .test_feedback import confirm_discard

    _, bound, decision = confirm_discard(ledger, sandbox)
    record_idea_failure(ledger, bound.candidate, decision)
    assert (
        sum(
            e.kind is ActivityKind.HYPOTHESIS_OUTCOME
            for e in ledger.activities(subject_hash=bound.card.hash)
        )
        == 1
    )
    verified = replace(decision, stage=DecisionStage.HELD_OUT, verdict=Verdict.VERIFIED)
    with pytest.raises(ConflictError, match="idea failure"):
        record_outcome(ledger, bound.card, verified)
    assert calibration_summary(ledger).counts["discarded"] == 1


def test_empty_summary_is_none_not_zero(ledger: Ledger) -> None:
    summary = calibration_summary(ledger)
    assert summary.scored == 0
    assert summary.brier_score is None
    assert summary.mean_signed_error is None
