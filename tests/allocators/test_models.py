"""Contract boundaries and budget invariants for all allocation plugins."""

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.allocators import AllocationSnapshot, Greedy, RandomSearch
from openloop.ledger import InvalidInputError, Ledger, Metric, Purpose, Result
from openloop.loops import T0, ContractError, LoopAction

from .helpers import ENVIRONMENT, complete, job, snapshot


def test_cache_aliases_do_not_inflate_evidence(ledger: Ledger) -> None:
    loop = T0()
    first = job(loop, 0)
    original = complete(ledger, loop.spec, first, 1.0)
    repeat = ledger.submit(first.inputs, purpose=Purpose.REPLICATION)
    ledger.start_stage(repeat.id, "execution")
    ledger.start_stage(repeat.id, "evaluation")
    ledger.complete(repeat.id, Result({"loss": Metric(2, loop.spec.metric_unit)}))
    alias = ledger.submit(first.inputs)
    state = snapshot(loop, ledger, (first, job(loop, 0, seed=1)))
    assert len(state.evidence) == 2
    assert state.evidence[0].attempt_id == original.attempt_id
    assert alias.attempt_id == original.attempt_id
    assert Greedy("1x").select(state).jobs == (job(loop, 0, seed=1),)


def test_snapshot_from_a_full_ledger_never_reselects_finished_work(
    ledger: Ledger,
) -> None:
    loop = T0()
    pool = tuple(job(loop, index) for index in range(120))
    for offered in pool[:110]:
        complete(ledger, loop.spec, offered, 1.0)
    assert len(ledger.query(limit=100)) == 100  # A default-sized page misses 10.
    state = snapshot(loop, ledger, pool, budget=1000, slots=120)
    assert len(state.evidence) == 110
    chosen = Greedy("1x").select(state).jobs
    assert {item.inputs.hash for item in chosen} == {
        item.inputs.hash for item in pool[110:]
    }


def test_pending_and_failed_inputs_are_not_selected_or_scored(ledger: Ledger) -> None:
    loop = T0()
    pending, failed, fresh = tuple(job(loop, index) for index in range(3))
    ledger.submit(pending.inputs)
    run = ledger.submit(failed.inputs)
    ledger.start_stage(run.id, "execution")
    ledger.fail(run.id, "worker crashed")
    state = snapshot(loop, ledger, (pending, failed, fresh), slots=3)
    before = ledger.query_all()
    assert state.evidence == ()
    assert RandomSearch("1x").select(state).jobs == (fresh,)
    assert ledger.query_all() == before


@pytest.mark.parametrize("field", ["data_hash", "budget_amount", "environment_hash"])
def test_invalid_jobs_are_rejected_before_any_write(ledger: Ledger, field: str) -> None:
    loop = T0()
    offered = job(loop, 0)
    value = 999 if field == "budget_amount" else "b" * 64
    invalid = replace(offered, inputs=replace(offered.inputs, **{field: value}))
    with pytest.raises(ContractError):
        snapshot(loop, ledger, (invalid,))
    assert ledger.query_all() == ()


def test_confirm_jobs_are_not_adaptive_selection_work(ledger: Ledger) -> None:
    loop = T0()
    inputs = loop.spec.inputs(
        {"coordinates": [0, 0]}, LoopAction(0, phase="confirm"), ENVIRONMENT
    )
    with pytest.raises(ContractError, match="screen"):
        snapshot(loop, ledger, (loop.build_job(inputs),))


def test_confirmation_evidence_is_not_a_selection_score(ledger: Ledger) -> None:
    loop = T0()
    first, other = job(loop, 0, seed=1), job(loop, 1, seed=1)
    complete(ledger, loop.spec, job(loop, 0), 1.0)
    complete(ledger, loop.spec, job(loop, 1), 2.0)
    inputs = loop.spec.inputs(
        {"coordinates": [1, 0]}, LoopAction(0, phase="confirm"), ENVIRONMENT
    )
    complete(ledger, loop.spec, loop.build_job(inputs), -999.0)
    state = snapshot(loop, ledger, (first, other))
    assert len(state.evidence) == 2
    assert Greedy("1x").select(state).jobs == (first,)


def test_invalid_metric_or_conflicting_attempt_is_rejected(ledger: Ledger) -> None:
    loop = T0()
    offered = job(loop, 0)
    run = complete(ledger, loop.spec, offered, 1.0)
    changed = replace(run, result=Result({"loss": Metric(1, "incorrect unit")}))
    with pytest.raises(ContractError, match="validation metric"):
        AllocationSnapshot(loop.spec, ENVIRONMENT, (offered,), (changed,), 10)
    with pytest.raises(ContractError, match="Conflicting"):
        AllocationSnapshot(
            loop.spec, ENVIRONMENT, (offered,), (run, replace(run, status="failed")), 10
        )  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budget_is_rejected(budget) -> None:
    with pytest.raises(InvalidInputError, match="budget"):
        AllocationSnapshot(T0().spec, ENVIRONMENT, (), (), budget)


def test_duplicate_offers_are_rejected(ledger: Ledger) -> None:
    loop = T0()
    offered = job(loop, 0)
    with pytest.raises(ContractError, match="unique"):
        snapshot(loop, ledger, (offered, offered))


@given(
    budget=st.integers(0, 30),
    slots=st.integers(1, 12),
    order=st.permutations(tuple(range(6))),
)
def test_selection_is_bounded_unique_and_independent_of_offer_order(
    budget: int, slots: int, order: tuple[int, ...]
) -> None:
    loop = T0()
    jobs = tuple(job(loop, index, fidelity="4x") for index in range(6))
    state = AllocationSnapshot(loop.spec, ENVIRONMENT, jobs, (), budget, slots=slots)
    policy = RandomSearch("4x", seed=10)
    allocation = policy.select(state)
    assert allocation.work_units <= budget
    assert len(allocation.jobs) <= slots
    assert len({item.inputs.hash for item in allocation.jobs}) == len(allocation.jobs)
    assert allocation == policy.select(
        replace(state, jobs=tuple(jobs[index] for index in order))
    )
