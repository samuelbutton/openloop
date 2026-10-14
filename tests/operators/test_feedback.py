"""Idea failures come from real decider decisions and a frozen binding."""

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
from openloop.operators import record_idea_failure, record_outcome
from openloop.studies.t0_family import T0Study

from .conftest import SandboxStandIn, bind_card, discarded_trial, make_world


def confirm_discard(ledger: Ledger, sandbox: SandboxStandIn):
    """Run a real Campaign: calibrate, screen, bind, confirm, and get a DISCARD."""
    trial, candidate = discarded_trial()
    world = make_world(ledger, T0Study(trial, gain=1.0, worsening=3.0))
    campaign = Campaign(world.study.protocol, world.screen_noise, world.confirm_noise)
    screen = campaign.evaluate(
        DecisionStage.SCREEN,
        candidate,
        world.pairs(candidate, world.study.protocol.screen),
    )
    assert screen.verdict is Verdict.PROMOTE
    bound = bind_card(world, sandbox, candidate)
    decision = campaign.evaluate(
        DecisionStage.CONFIRM,
        candidate,
        world.pairs(candidate, world.study.protocol.confirm),
    )
    assert decision.verdict is Verdict.DISCARD
    return world, bound, decision


def test_real_confirm_discard_records_failure_and_outcome(
    ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    _, bound, decision = confirm_discard(ledger, sandbox)
    # decide() lists 16 pairs (32 runs), then the 10 confirm-noise runs.
    assert decision.sample_count == 16
    assert len(decision.evidence) == 2 * 16 + 10
    event = record_idea_failure(ledger, bound.candidate, decision)
    assert event.kind is ActivityKind.IDEA_FAILURE
    assert event.parent_sequence is not None
    assert record_idea_failure(ledger, bound.candidate, decision) == event
    history = ledger.activities(subject_hash=bound.card.hash)
    outcomes = [e for e in history if e.kind is ActivityKind.HYPOTHESIS_OUTCOME]
    assert len(outcomes) == 1
    assert outcomes[0].payload["outcome"] == HypothesisOutcome.DISCARDED
    assert outcomes[0].payload["decision_hash"] == content_hash(decision)
    assert event.payload["decision_hash"] == content_hash(decision)
    assert all(e.kind is not ActivityKind.IMPLEMENTATION_FAILURE for e in history)
    assert record_outcome(ledger, bound.card, decision) == outcomes[0]


def test_invalid_decisions_record_nothing(
    ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    _, bound, decision = confirm_discard(ledger, sandbox)
    pairs = decision.evidence[:32]
    calibration = decision.evidence[32:]
    swapped = tuple(
        reference
        for offset in range(0, 32, 2)
        for reference in (pairs[offset + 1], pairs[offset])
    )
    invalid = (
        replace(decision, sample_count=0),
        replace(decision, verdict=Verdict.INCONCLUSIVE),
        replace(decision, stage=DecisionStage.HELD_OUT),
        replace(decision, candidate_hash="d" * 64),
        replace(decision, evidence=pairs),
        replace(decision, evidence=(pairs[0], pairs[0])),
        replace(decision, evidence=swapped + calibration),
        replace(decision, evidence=pairs + (pairs[1],) * 2),
        replace(decision, evidence=(*pairs[:-1], *calibration)),
    )
    before = ledger.activities()
    for case in invalid:
        with pytest.raises(InvalidInputError):
            record_idea_failure(ledger, bound.candidate, case)
    assert ledger.activities() == before


def test_idea_failure_requires_a_discarded_outcome(
    ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    _, bound, decision = confirm_discard(ledger, sandbox)
    other = replace(decision, reason="A different reason")
    record_outcome(ledger, bound.card, decision)
    with pytest.raises(ConflictError, match="differs"):
        record_idea_failure(ledger, bound.candidate, other)
    event = record_idea_failure(ledger, bound.candidate, decision)
    assert event.kind is ActivityKind.IDEA_FAILURE
