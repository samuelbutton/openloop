"""Synchronous promotion barriers and finite Hyperband bracket plans."""

from dataclasses import replace

import pytest

from openloop.allocators import Hyperband, SuccessiveHalving
from openloop.ledger import Ledger, Metric, Purpose, Result
from openloop.loops import T0, ContractError, Fidelity, LoopAction

from .helpers import complete, job, snapshot

RUNGS = (LoopAction(0, "1x"), LoopAction(7, "4x"))


def test_halving_waits_for_all_results_then_promotes_only_top_candidates(
    ledger: Ledger,
) -> None:
    loop = T0()
    low = tuple(job(loop, index) for index in range(4))
    high = tuple(job(loop, index, seed=7, fidelity="4x") for index in range(4))
    policy = SuccessiveHalving(
        tuple(item.inputs.candidate_hash for item in low), RUNGS, eta=2
    )
    complete(ledger, loop.spec, low[0], 4)
    complete(ledger, loop.spec, low[1], 3)
    complete(ledger, loop.spec, low[2], 2)
    pending = ledger.submit(low[3].inputs)
    offers = (*low, *high)
    assert policy.select(snapshot(loop, ledger, offers, slots=10)).jobs == ()
    ledger.start_stage(pending.id, "execution")
    ledger.start_stage(pending.id, "evaluation")
    ledger.complete(pending.id, Result({"loss": Metric(1, loop.spec.metric_unit)}))
    promoted = policy.select(snapshot(loop, ledger, offers, slots=10))
    assert {item.inputs.hash for item in promoted.jobs} == {
        item.inputs.hash for item in high[2:]
    }
    assert promoted.work_units == 8
    assert (
        len(policy.select(snapshot(loop, ledger, offers, budget=7, slots=10)).jobs) == 1
    )
    assert policy.select(snapshot(loop, ledger, offers, budget=3)).jobs == ()
    for index in (2, 3):
        complete(ledger, loop.spec, high[index], float(index))
    assert policy.select(snapshot(loop, ledger, offers)).jobs == ()


def test_failure_blocks_a_rung_until_an_explicit_core_retry_succeeds(
    ledger: Ledger,
) -> None:
    loop = T0()
    low = tuple(job(loop, index) for index in range(2))
    high = tuple(job(loop, index, seed=7, fidelity="4x") for index in range(2))
    policy = SuccessiveHalving(
        tuple(item.inputs.candidate_hash for item in low), RUNGS, eta=2
    )
    complete(ledger, loop.spec, low[0], 1)
    failed = ledger.submit(low[1].inputs)
    ledger.start_stage(failed.id, "execution")
    ledger.fail(failed.id, "killed worker")
    assert policy.select(snapshot(loop, ledger, (*low, *high))).jobs == ()
    retry = ledger.submit(low[1].inputs, purpose=Purpose.RETRY, retry_of=failed.id)
    ledger.start_stage(retry.id, "execution")
    ledger.start_stage(retry.id, "evaluation")
    ledger.complete(retry.id, Result({"loss": Metric(0.5, loop.spec.metric_unit)}))
    assert policy.select(snapshot(loop, ledger, (*low, *high))).jobs == (high[1],)


def test_late_replication_and_cache_alias_cannot_revise_a_promotion(
    ledger: Ledger,
) -> None:
    loop = T0()
    low = tuple(job(loop, index) for index in range(2))
    high = tuple(job(loop, index, seed=7, fidelity="4x") for index in range(2))
    policy = SuccessiveHalving(
        tuple(item.inputs.candidate_hash for item in low), RUNGS, eta=2
    )
    complete(ledger, loop.spec, low[0], 1)
    complete(ledger, loop.spec, low[1], 2)
    state = snapshot(loop, ledger, (*low, *high))
    selected = policy.select(state)
    replica = ledger.submit(low[0].inputs, purpose=Purpose.REPLICATION)
    ledger.start_stage(replica.id, "execution")
    ledger.start_stage(replica.id, "evaluation")
    ledger.complete(replica.id, Result({"loss": Metric(999, loop.spec.metric_unit)}))
    ledger.submit(low[0].inputs)
    assert policy.select(snapshot(loop, ledger, (*low, *high))) == selected


def test_hyperband_runs_distinct_brackets_including_full_fidelity_random_search(
    ledger: Ledger,
) -> None:
    loop = T0()
    low = tuple(job(loop, index) for index in range(6))
    high = tuple(job(loop, index, seed=7, fidelity="4x") for index in range(6))
    candidates = tuple(item.inputs.candidate_hash for item in low)
    policy = Hyperband(candidates, RUNGS, eta=2, seed=42)
    assert policy.bracket_sizes == (2, 2)
    first, second = policy.brackets(loop.spec)
    assert not set(first.candidates) & set(second.candidates)
    assert second.rungs == (RUNGS[1],)
    assert policy.brackets(loop.spec) == replace(
        policy, candidates=tuple(reversed(candidates))
    ).brackets(loop.spec)
    offers = (*low, *high)
    initial = policy.select(snapshot(loop, ledger, offers, slots=10))
    assert len(initial.jobs) == 2
    assert initial.work_units == 2
    for index, offered in enumerate(initial.jobs):
        complete(ledger, loop.spec, offered, float(index))
    promotion = policy.select(snapshot(loop, ledger, offers, slots=10))
    assert len(promotion.jobs) == 1
    assert promotion.work_units == 4
    complete(ledger, loop.spec, promotion.jobs[0], 1)
    direct = policy.select(snapshot(loop, ledger, offers, slots=10))
    assert {item.inputs.candidate_hash for item in direct.jobs} == set(
        second.candidates
    )
    assert direct.work_units == 8
    for offered in direct.jobs:
        complete(ledger, loop.spec, offered, 1)
    assert policy.select(snapshot(loop, ledger, offers)).jobs == ()


def test_late_completion_of_an_older_attempt_cannot_revise_promotion(
    ledger: Ledger,
) -> None:
    loop = T0()
    low = tuple(job(loop, index) for index in range(2))
    high = tuple(job(loop, index, seed=7, fidelity="4x") for index in range(2))
    policy = SuccessiveHalving(
        tuple(item.inputs.candidate_hash for item in low), RUNGS, eta=2
    )
    older = ledger.submit(low[0].inputs)
    ledger.start_stage(older.id, "execution")
    first_completed = ledger.submit(low[0].inputs, purpose=Purpose.REPLICATION)
    ledger.start_stage(first_completed.id, "execution")
    ledger.start_stage(first_completed.id, "evaluation")
    ledger.complete(
        first_completed.id, Result({"loss": Metric(1, loop.spec.metric_unit)})
    )
    complete(ledger, loop.spec, low[1], 2)
    selected = policy.select(snapshot(loop, ledger, (*low, *high)))
    assert selected.jobs == (high[0],)
    ledger.start_stage(older.id, "evaluation")
    ledger.complete(older.id, Result({"loss": Metric(999, loop.spec.metric_unit)}))
    assert policy.select(snapshot(loop, ledger, (*low, *high))) == selected


def test_hyperband_geometric_grid_matches_published_bracket_counts() -> None:
    spec = replace(
        T0().spec,
        fidelities=(
            Fidelity("1", 1),
            Fidelity("3", 3),
            Fidelity("9", 9),
            Fidelity("27", 27),
            Fidelity("81", 81),
        ),
    )
    rungs = tuple(LoopAction(0, fidelity.name) for fidelity in spec.fidelities)
    candidates = tuple(f"{index:064x}" for index in range(200))
    policy = Hyperband(candidates, rungs)
    assert policy.bracket_sizes == (81, 34, 15, 8, 5)
    brackets = policy.brackets(spec)
    assert tuple(
        spec.fidelity(bracket.rungs[0].fidelity).amount for bracket in brackets
    ) == (1, 3, 9, 27, 81)


@pytest.mark.parametrize("eta", [1, True, 2.5])
def test_invalid_halving_factor_is_rejected(eta) -> None:
    with pytest.raises(ContractError, match="eta"):
        SuccessiveHalving(("a" * 64,), RUNGS, eta=eta)


@pytest.mark.parametrize(
    "rungs",
    [(), (LoopAction(0, phase="confirm"),), (LoopAction(0, purpose="replication"),)],
)
def test_invalid_rungs_are_rejected(rungs: tuple[LoopAction, ...]) -> None:
    with pytest.raises(ContractError):
        SuccessiveHalving(("a" * 64,), rungs)


def test_reversed_fidelities_and_small_pool_are_rejected(ledger: Ledger) -> None:
    loop = T0()
    policy = SuccessiveHalving(("a" * 64,), tuple(reversed(RUNGS)))
    with pytest.raises(ContractError, match="increase"):
        policy.select(snapshot(loop, ledger, ()))
    with pytest.raises(ContractError, match="too small"):
        Hyperband(("a" * 64,), RUNGS)
