"""Spend gate: reserve before paid calls, settle from usage, keep unknowns open."""

import asyncio
import sys
from collections.abc import Mapping

import pytest

from openloop.ledger import Activity, ActivityKind, InvalidInputError, JSONValue, Ledger
from openloop.loops import T0
from openloop.operators import (
    CallBudget,
    ImplementationRequest,
    Implementer,
    ModelReply,
    Patch,
    Pricing,
    ProposalDraft,
    ProposalRequest,
    Proposer,
    SpendGate,
    SpendLimitError,
)
from openloop.operators.models import prompt_digest

from .conftest import SandboxStandIn

# Arbitrary test prices. The package hard-codes none.
PRICING = Pricing("scripted", "model-alias", 2.0, 8.0)
BUDGET = CallBudget(PRICING, 100_000, 1_000)
MAX_COST = 100_000 * 2 + 1_000 * 8  # micro-USD: USD per million equals per token


def reply[T](
    value: T, *, usage: bool = True, simulated: bool = False, name: str = "r"
) -> ModelReply[T]:
    return ModelReply(
        value,
        "scripted",
        "model-snapshot",
        name,
        simulated=simulated,
        input_tokens=500 if usage else None,
        output_tokens=100 if usage else None,
        reasoning_tokens=40 if usage else None,
        cached_input_tokens=0 if usage else None,
    )


DRAFT = ProposalDraft("Move", "Because", "Edit", -0.1, 0.6, (-0.3, 0.1))


def paid_proposer(loop: T0, ledger: Ledger, provider, *, limit: float = 0.5):
    return Proposer(
        loop.spec,
        ledger,
        provider,
        gate=SpendGate(ledger, limit_usd=limit),
        budget=BUDGET,
    )


def spend_of(event: Activity) -> Mapping[str, JSONValue]:
    note = event.payload["spend"]
    assert isinstance(note, Mapping)
    return note


def kinds(ledger: Ledger):
    return [event.kind for event in ledger.activities()]


def test_pricing_and_usage_validation() -> None:
    assert BUDGET.max_cost_micro_usd == MAX_COST
    assert PRICING.cost_micro_usd(500, 100) == 1_800
    assert Pricing("p", "m", 0.1, 0.3).cost_micro_usd(3, 1) == 1  # rounds up
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(InvalidInputError):
            Pricing("p", "m", bad, 1.0)
    with pytest.raises(InvalidInputError):
        ModelReply("v", "p", "m", "id", input_tokens=-1)
    with pytest.raises(InvalidInputError):
        ModelReply("v", "p", "m", "id", input_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        ModelReply("v", "p", "m", "id", output_tokens=1, reasoning_tokens=2)
    with pytest.raises(InvalidInputError):
        ModelReply("v", "p", "m", "id", input_tokens=1, cached_input_tokens=2)
    with pytest.raises(InvalidInputError):
        ProposalDraft("t", "r", "c", 0.0, 1.5)
    with pytest.raises(InvalidInputError):
        ProposalDraft("t", "r", "c", 0.0, 0.5, (1.0, -1.0))
    with pytest.raises(InvalidInputError):
        SpendGate.__new__(SpendGate).__init__(None, limit_usd=0)  # type: ignore[arg-type]


def test_operators_require_a_gate_or_explicit_simulation(
    loop: T0, ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    async def provider(request: ProposalRequest):
        return reply(DRAFT)

    gate = SpendGate(ledger, limit_usd=1)
    with pytest.raises(InvalidInputError, match="SpendGate"):
        Proposer(loop.spec, ledger, provider)
    with pytest.raises(InvalidInputError, match="simulated-only"):
        Proposer(
            loop.spec, ledger, provider, gate=gate, budget=BUDGET, simulated_only=True
        )
    with pytest.raises(InvalidInputError, match="SpendGate"):
        Implementer(sandbox, ledger, provider, base_files={"a": "b"}, smoke=("x",))  # type: ignore[arg-type]


def test_reserve_then_settle_from_usage(loop: T0, ledger: Ledger, baseline) -> None:
    seen: list[ProposalRequest] = []

    async def provider(request: ProposalRequest):
        seen.append(request)
        assert kinds(ledger) == [ActivityKind.SPEND_RESERVATION]
        return reply(DRAFT)

    card = asyncio.run(
        paid_proposer(loop, ledger, provider).propose("draft", baseline=baseline.id)
    )
    assert kinds(ledger) == [
        ActivityKind.SPEND_RESERVATION,
        ActivityKind.SPEND_SETTLEMENT,
        ActivityKind.HYPOTHESIS,
    ]
    reservation, settlement, hypothesis = ledger.activities()
    digest = prompt_digest(seen[0].prompt)
    assert reservation.payload["prompt_hash"] == digest
    assert hypothesis.payload["prompt_hash"] == digest
    assert reservation.payload["pricing_hash"] == PRICING.hash
    assert reservation.payload["max_cost_micro_usd"] == MAX_COST
    assert settlement.parent_sequence == reservation.sequence
    assert settlement.payload["cost_micro_usd"] == 1_800
    assert settlement.payload["model"] == "model-snapshot"
    assert settlement.payload["configured_model"] == "model-alias"
    assert settlement.payload["reasoning_tokens"] == 40
    reply_record = hypothesis.payload["reply"]
    assert isinstance(reply_record, Mapping)
    assert reply_record["input_tokens"] == 500
    assert card.prediction.probability_of_improvement == 0.6
    gate = SpendGate(ledger, limit_usd=0.5)
    committed = gate.committed()
    assert (committed.settled_micro_usd, committed.open_micro_usd) == (1_800, 0)


def test_limit_counts_open_reservations_and_blocks_the_call(
    loop: T0, ledger: Ledger, baseline
) -> None:
    calls = 0

    async def provider(request: ProposalRequest):
        nonlocal calls
        calls += 1
        return reply(DRAFT, usage=False)

    proposer = paid_proposer(loop, ledger, provider, limit=0.3)
    asyncio.run(proposer.propose("draft", baseline=baseline.id))
    hypothesis = ledger.activities()[-1]
    assert spend_of(hypothesis)["state"] == "open"  # type: ignore[index]
    assert "usage" in str(spend_of(hypothesis)["why"])
    assert SpendGate(ledger, limit_usd=1).committed().open_micro_usd == MAX_COST
    # A second full reservation (0.208 + 0.208) exceeds 0.3.
    with pytest.raises(SpendLimitError, match="exceeded"):
        asyncio.run(proposer.propose("draft", baseline=baseline.id))
    assert calls == 1
    assert ledger.activities()[-1].kind is ActivityKind.PROPOSAL_FAILURE
    assert kinds(ledger).count(ActivityKind.SPEND_RESERVATION) == 1


def test_error_and_cancel_keep_the_reservation_open(
    loop: T0, ledger: Ledger, baseline
) -> None:
    outcomes = [RuntimeError("provider down"), asyncio.CancelledError()]

    async def provider(request: ProposalRequest):
        raise outcomes.pop(0)

    proposer = paid_proposer(loop, ledger, provider)
    with pytest.raises(RuntimeError):
        asyncio.run(proposer.propose("draft", baseline=baseline.id))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(proposer.propose("draft", baseline=baseline.id))
    committed = SpendGate(ledger, limit_usd=1).committed()
    assert committed.open_reservations == 2
    assert kinds(ledger) == [
        ActivityKind.SPEND_RESERVATION,
        ActivityKind.PROPOSAL_FAILURE,
        ActivityKind.SPEND_RESERVATION,
        ActivityKind.PROPOSAL_FAILURE,
    ]
    failure = ledger.activities()[1]
    assert spend_of(failure)["state"] == "open"  # type: ignore[index]
    assert "RuntimeError" in str(spend_of(failure)["why"])
    assert committed.open_micro_usd == 2 * MAX_COST


def test_retry_needs_a_new_reservation_and_reconciliation_closes_one(
    loop: T0, ledger: Ledger, baseline
) -> None:
    async def provider(request: ProposalRequest):
        return reply(DRAFT, name=f"r{len(ledger.activities())}")

    proposer = paid_proposer(loop, ledger, provider)
    asyncio.run(proposer.propose("draft", baseline=baseline.id))
    asyncio.run(proposer.propose("draft", baseline=baseline.id))
    assert kinds(ledger).count(ActivityKind.SPEND_RESERVATION) == 2
    gate = SpendGate(ledger, limit_usd=1)
    reservation = gate.reserve(BUDGET, purpose="manual", prompt="hello")
    assert gate.committed().open_micro_usd == MAX_COST
    gate.settle_reported(reservation, cost_micro_usd=7, reason="invoice line 12")
    assert gate.committed().open_micro_usd == 0
    assert gate.committed().settled_micro_usd == 2 * 1_800 + 7
    oversize = "x" * 100_001
    with pytest.raises(SpendLimitError, match="input token budget"):
        gate.reserve(BUDGET, purpose="manual", prompt=oversize)


def test_simulated_only_refuses_and_records_a_paid_reply(
    loop: T0, ledger: Ledger, baseline
) -> None:
    async def provider(request: ProposalRequest):
        return reply(DRAFT)

    with pytest.raises(SpendLimitError, match="non-simulated"):
        asyncio.run(
            Proposer(loop.spec, ledger, provider, simulated_only=True).propose(
                "draft", baseline=baseline.id
            )
        )
    assert kinds(ledger) == [ActivityKind.PROPOSAL_FAILURE]
    assert spend_of(ledger.activities()[0])["state"] == "unreserved"  # type: ignore[index]


def test_simulated_reply_through_a_gate_costs_nothing(
    loop: T0, ledger: Ledger, baseline
) -> None:
    async def provider(request: ProposalRequest):
        return reply(DRAFT, simulated=True)

    asyncio.run(
        paid_proposer(loop, ledger, provider).propose("draft", baseline=baseline.id)
    )
    assert ledger.activities()[1].payload["basis"] == "simulated"
    assert SpendGate(ledger, limit_usd=1).committed().total_micro_usd == 0


def test_implementer_gate_settles_and_stops_at_the_limit(
    proposer, baseline, ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    card = asyncio.run(proposer.propose("draft", baseline=baseline.id))
    calls = 0

    async def editor(request: ImplementationRequest):
        nonlocal calls
        calls += 1
        return reply(Patch({"candidate.txt": "after"}), usage=calls != 1)

    # Room for one full reservation, not two.
    implementer = Implementer(
        sandbox,
        ledger,
        editor,
        base_files={"candidate.txt": "before"},
        smoke=(sys.executable, "-c", "pass"),
        gate=SpendGate(ledger, limit_usd=0.3),
        budget=BUDGET,
    )
    # Attempt 1 has unknown usage: reservation stays open and counts. Its smoke
    # passes, so the card is implemented and the reservation remains open.
    candidate = asyncio.run(implementer.implement(card))
    assert candidate is not None
    success = ledger.activities(subject_hash=card.hash)[-1]
    assert success.kind is ActivityKind.IMPLEMENTATION_SUCCESS
    assert spend_of(success)["state"] == "open"  # type: ignore[index]
    assert success.payload["prompt_hash"]
    assert SpendGate(ledger, limit_usd=1).committed().open_micro_usd == MAX_COST


def test_implementer_limit_refusal_is_recorded_without_a_provider_call(
    proposer, baseline, ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    card = asyncio.run(proposer.propose("draft", baseline=baseline.id))
    gate = SpendGate(ledger, limit_usd=0.3)
    gate.reserve(BUDGET, purpose="other", prompt="earlier")  # leaves 0.092

    async def editor(request: ImplementationRequest):
        raise AssertionError("must not be called")

    implementer = Implementer(
        sandbox,
        ledger,
        editor,
        base_files={"candidate.txt": "before"},
        smoke=(sys.executable, "-c", "pass"),
        gate=gate,
        budget=BUDGET,
    )
    with pytest.raises(SpendLimitError):
        asyncio.run(implementer.implement(card))
    events = ledger.activities(subject_hash=card.hash)
    assert events[-1].kind is ActivityKind.IMPLEMENTATION_FAILURE
    assert "Spend limit" in str(events[-1].payload["reason"])
    assert sandbox.jobs == []
