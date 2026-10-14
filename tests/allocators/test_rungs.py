"""Allocation status, rung-atomic admission, failure policy, and multi-seed rungs."""

import pytest

from openloop.allocators import (
    Allocation,
    AllocationSnapshot,
    AllocationStatus,
    Greedy,
    Hyperband,
    OnFailure,
    RandomSearch,
    Rung,
    SuccessiveHalving,
)
from openloop.ledger import InvalidInputError, Ledger, Metric, Result
from openloop.loops import T0, Job

from .helpers import ENVIRONMENT, complete, fail, geometric_loop, job, snapshot

RUNGS = (Rung("1x", (0,)), Rung("4x", (7,)))


def pool(
    loop: T0, size: int = 4, seeds: tuple[int, ...] = (0,), high: tuple[int, ...] = (7,)
) -> tuple[tuple[Job, ...], tuple[Job, ...], tuple[str, ...]]:
    low = tuple(job(loop, i, seed=s) for i in range(size) for s in seeds)
    top = tuple(job(loop, i, seed=s, fidelity="4x") for i in range(size) for s in high)
    candidates = tuple(job(loop, i).inputs.candidate_hash for i in range(size))
    return low, top, candidates


def test_each_status_is_reported_with_a_reason(ledger: Ledger) -> None:
    loop = T0()
    low, high, candidates = pool(loop)
    policy = SuccessiveHalving(candidates, RUNGS, eta=2)
    offers = (*low, *high)

    def status(**changes: int) -> Allocation:
        return policy.select(snapshot(loop, ledger, offers, slots=10, **changes))

    assert status().status is AllocationStatus.SELECTED
    assert status(budget=3).status is AllocationStatus.BUDGET_EXHAUSTED
    no_slots = policy.select(snapshot(loop, ledger, offers, slots=0))
    assert (no_slots.status, no_slots.jobs) == (AllocationStatus.NO_SLOTS, ())
    missing = policy.select(snapshot(loop, ledger, low[:3], slots=10))
    assert missing.status is AllocationStatus.MISSING_JOBS
    pending = ledger.submit(low[0].inputs)
    for offered in low[1:]:
        complete(ledger, loop.spec, offered, 1.0)
    waiting = status()
    assert (waiting.status, waiting.jobs) == (AllocationStatus.WAITING, ())
    ledger.start_stage(pending.id, "execution")
    ledger.fail(pending.id, "worker crashed")
    blocked = status()
    assert (blocked.status, blocked.jobs) == (AllocationStatus.BLOCKED, ())
    assert "retry" in blocked.reason
    for allocation in (waiting, blocked, no_slots, missing):
        assert allocation.reason


def test_done_is_reported_when_nothing_is_left(ledger: Ledger) -> None:
    loop = T0()
    low, high, candidates = pool(loop, 2)
    policy = SuccessiveHalving(candidates, RUNGS, eta=2)
    for offered in low:
        complete(ledger, loop.spec, offered, 1.0)
    final = policy.select(snapshot(loop, ledger, (*low, *high)))
    assert len(final.jobs) == 1
    complete(ledger, loop.spec, final.jobs[0], 1.0)
    done = policy.select(snapshot(loop, ledger, (*low, *high)))
    assert (done.jobs, done.status) == ((), AllocationStatus.DONE)
    assert Greedy("4x").select(snapshot(loop, ledger, ())).status is (
        AllocationStatus.DONE
    )


@pytest.mark.parametrize("policy", [Greedy("1x"), RandomSearch("1x")])
def test_one_shot_policies_report_budget_and_slots(
    ledger: Ledger, policy: Greedy | RandomSearch
) -> None:
    loop = T0()
    offers = tuple(job(loop, i, fidelity="1x") for i in range(3))
    assert policy.select(snapshot(loop, ledger, offers, budget=0)).status is (
        AllocationStatus.BUDGET_EXHAUSTED
    )
    assert policy.select(snapshot(loop, ledger, offers, slots=0)).status is (
        AllocationStatus.NO_SLOTS
    )
    assert policy.select(snapshot(loop, ledger, offers)).status is (
        AllocationStatus.SELECTED
    )


def test_a_rung_starts_only_when_all_its_unstarted_jobs_are_affordable(
    ledger: Ledger,
) -> None:
    loop = T0()
    low, high, candidates = pool(loop)
    policy = SuccessiveHalving(candidates, RUNGS, eta=2)
    offers = (*low, *high)
    short = policy.select(snapshot(loop, ledger, offers, budget=3, slots=10))
    assert (short.jobs, short.status) == ((), AllocationStatus.BUDGET_EXHAUSTED)
    first = policy.select(snapshot(loop, ledger, offers, budget=4, slots=2))
    assert len(first.jobs) == 2
    # The coordinator reserved those two jobs, so only the other two are priced.
    for started in first.jobs:
        ledger.submit(started.inputs)
    rest = policy.select(snapshot(loop, ledger, offers, budget=2, slots=10))
    assert len(rest.jobs) == 2
    assert not {item.inputs.hash for item in rest.jobs} & {
        item.inputs.hash for item in first.jobs
    }
    one_short = policy.select(snapshot(loop, ledger, offers, budget=1, slots=10))
    assert one_short.status is AllocationStatus.BUDGET_EXHAUSTED


def test_hyperband_stops_instead_of_skipping_to_a_cheaper_bracket(
    ledger: Ledger,
) -> None:
    loop = geometric_loop(1, 2, 1000)
    rungs = tuple(Rung(name, (0,)) for name in ("1", "2", "1000"))
    candidates = tuple(
        job(loop, i, fidelity="1").inputs.candidate_hash for i in range(10)
    )
    jobs = tuple(
        job(loop, i, fidelity=r.fidelity or "") for i in range(10) for r in rungs
    )
    policy = Hyperband(candidates, rungs, eta=2, seed=1)
    assert policy.bracket_sizes == (4, 3, 3)
    for _ in range(2):
        step = policy.select(
            AllocationSnapshot(
                loop.spec, ENVIRONMENT, jobs, ledger.query_all(), 100, slots=10
            )
        )
        for offered in step.jobs:
            complete(ledger, loop.spec, offered, 1.0)
    state = AllocationSnapshot(
        loop.spec, ENVIRONMENT, jobs, ledger.query_all(), 100, slots=10
    )
    stopped = policy.select(state)
    # The next bracket's first rung costs 6 and fits; the unfinished one does not.
    assert (stopped.jobs, stopped.status) == ((), AllocationStatus.BUDGET_EXHAUSTED)


def test_block_is_the_default_and_part_of_the_policy_identity(
    ledger: Ledger,
) -> None:
    loop = T0()
    _, _, candidates = pool(loop)
    block = SuccessiveHalving(candidates, RUNGS, eta=2)
    assert block.on_failure is OnFailure.BLOCK
    worst = SuccessiveHalving(candidates, RUNGS, eta=2, on_failure="rank_worst")  # pyright: ignore[reportArgumentType]
    assert worst.on_failure is OnFailure.RANK_WORST
    state = snapshot(loop, ledger, ())
    assert block.select(state).policy_hash != worst.select(state).policy_hash
    with pytest.raises(InvalidInputError, match="failure policy"):
        SuccessiveHalving(candidates, RUNGS, on_failure="ignore")  # pyright: ignore[reportArgumentType]


def test_rank_worst_never_promotes_a_failure_and_keeps_it_out_of_the_scores(
    ledger: Ledger,
) -> None:
    loop = T0()
    low, high, candidates = pool(loop)
    policy = SuccessiveHalving(
        candidates, RUNGS, eta=2, on_failure=OnFailure.RANK_WORST
    )
    fail(ledger, low[0])
    for offered, value in zip(low[1:], (5.0, 1.0, 5.0), strict=True):
        complete(ledger, loop.spec, offered, value)
    state = snapshot(loop, ledger, (*low, *high), slots=10)
    assert len(state.evidence) == 3
    promoted = policy.select(state)
    assert promoted.status is AllocationStatus.SELECTED
    # floor(4 / 2) = 2 slots; the failed candidate is not promoted to fill one.
    assert len(promoted.jobs) == 2
    assert high[0] not in promoted.jobs
    assert high[2] in promoted.jobs
    assert len(ledger.query_all()) == 4


def test_rank_worst_with_too_few_scores_promotes_fewer_and_can_end_empty(
    ledger: Ledger,
) -> None:
    loop = T0()
    low, high, candidates = pool(loop)
    policy = SuccessiveHalving(
        candidates, RUNGS, eta=2, on_failure=OnFailure.RANK_WORST
    )
    for offered in low[:3]:
        fail(ledger, offered)
    complete(ledger, loop.spec, low[3], 1.0)
    only = policy.select(snapshot(loop, ledger, (*low, *high), slots=10))
    assert only.jobs == (high[3],)
    complete(ledger, loop.spec, high[3], 1.0)
    assert policy.select(snapshot(loop, ledger, (*low, *high))).status is (
        AllocationStatus.DONE
    )


def test_rank_worst_all_failures_end_the_bracket(ledger: Ledger) -> None:
    loop = T0()
    low, high, candidates = pool(loop, 2)
    policy = SuccessiveHalving(
        candidates, RUNGS, eta=2, on_failure=OnFailure.RANK_WORST
    )
    for offered in low:
        fail(ledger, offered)
    done = policy.select(snapshot(loop, ledger, (*low, *high)))
    assert (done.jobs, done.status) == ((), AllocationStatus.DONE)
    assert "failed" in done.reason


def test_rank_worst_stops_spending_seeds_on_a_failed_candidate(ledger: Ledger) -> None:
    loop = T0()
    rungs = (Rung("1x", (0, 1)), Rung("4x", (7,)))
    low, high, candidates = pool(loop, 3, seeds=(0, 1))
    policy = SuccessiveHalving(
        candidates, rungs, eta=3, on_failure=OnFailure.RANK_WORST
    )
    fail(ledger, low[0])
    started = policy.select(snapshot(loop, ledger, (*low, *high), slots=10))
    assert {item.inputs.candidate_hash for item in started.jobs} == set(candidates[1:])
    assert len(started.jobs) == 4
    block = SuccessiveHalving(candidates, rungs, eta=3)
    assert len(block.select(snapshot(loop, ledger, (*low, *high), slots=10)).jobs) == 5


def test_multi_seed_rung_promotes_on_the_mean_and_needs_every_seed(
    ledger: Ledger,
) -> None:
    loop = T0()
    seeds = (0, 1, 2)
    rungs = (Rung("1x", seeds), Rung("4x", (7,)))
    low, high, candidates = pool(loop, 4, seeds=seeds)
    policy = SuccessiveHalving(candidates, rungs, eta=2)
    offers = (*low, *high)
    first = policy.select(snapshot(loop, ledger, offers, slots=20))
    assert len(first.jobs) == 12
    assert policy.select(
        snapshot(loop, ledger, offers, budget=11, slots=20)
    ).status is (AllocationStatus.BUDGET_EXHAUSTED)
    # Candidate 0 has the best single sample (the winner's curse) but a poor mean.
    scores = {0: (0.0, 10.0, 10.0), 1: (3.0, 3.0, 3.0), 2: (4.0, 4.0, 4.0)}
    scores[3] = (9.0, 9.0, 9.0)
    for index, values in scores.items():
        for seed, value in zip(seeds, values, strict=True):
            if (index, seed) != (3, 2):
                complete(ledger, loop.spec, low[index * 3 + seed], value)
    assert policy.select(snapshot(loop, ledger, offers, slots=20)).jobs == (low[11],)
    last = ledger.submit(low[11].inputs)
    waiting = policy.select(snapshot(loop, ledger, offers, slots=20))
    assert (waiting.jobs, waiting.status) == ((), AllocationStatus.WAITING)
    ledger.start_stage(last.id, "execution")
    ledger.start_stage(last.id, "evaluation")
    ledger.complete(last.id, Result({"loss": Metric(9.0, loop.spec.metric_unit)}))
    promoted = policy.select(snapshot(loop, ledger, offers, slots=20))
    assert {item.inputs.hash for item in promoted.jobs} == {
        high[1].inputs.hash,
        high[2].inputs.hash,
    }


def test_single_sample_would_promote_the_lucky_candidate(ledger: Ledger) -> None:
    loop = T0()
    single = (Rung("1x", (0,)), Rung("4x", (7,)))
    low, high, candidates = pool(loop, 4)
    for offered, value in zip(low, (0.0, 3.0, 4.0, 9.0), strict=True):
        complete(ledger, loop.spec, offered, value)
    promoted = SuccessiveHalving(candidates, single, eta=2).select(
        snapshot(loop, ledger, (*low, *high), slots=10)
    )
    assert high[0] in promoted.jobs


def test_multi_seed_mean_overrules_the_lucky_first_sample(ledger: Ledger) -> None:
    loop = T0()
    seeds = (0, 1, 2)
    rungs = (Rung("1x", seeds), Rung("4x", (7,)))
    low, high, candidates = pool(loop, 4, seeds=seeds)
    scores = {0: (0.0, 10.0, 10.0), 1: (3.0, 3.0, 3.0), 2: (4.0, 4.0, 4.0)}
    scores[3] = (9.0, 9.0, 9.0)
    for index, values in scores.items():
        for seed, value in zip(seeds, values, strict=True):
            complete(ledger, loop.spec, low[index * 3 + seed], value)
    promoted = SuccessiveHalving(candidates, rungs, eta=2).select(
        snapshot(loop, ledger, (*low, *high), slots=10)
    )
    assert {item.inputs.hash for item in promoted.jobs} == {
        high[1].inputs.hash,
        high[2].inputs.hash,
    }
    assert (
        SuccessiveHalving(candidates, rungs, eta=2)
        .select(snapshot(loop, ledger, ()))
        .policy_hash
        != SuccessiveHalving(candidates, RUNGS, eta=2)
        .select(snapshot(loop, ledger, ()))
        .policy_hash
    )
