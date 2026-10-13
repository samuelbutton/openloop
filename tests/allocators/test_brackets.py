"""Full promotion counts, so an off-by-one at any rung changes a number."""

from openloop.allocators import (
    AllocationSnapshot,
    AllocationStatus,
    Hyperband,
    Rung,
    SuccessiveHalving,
)
from openloop.ledger import Ledger
from openloop.loops import T0, Job

from .helpers import ENVIRONMENT, complete, geometric_loop, job


def drive(
    policy: SuccessiveHalving | Hyperband,
    loop: T0,
    ledger: Ledger,
    jobs: tuple[Job, ...],
    rank: dict[str, int],
) -> list[tuple[str, int]]:
    """Run to completion; return (fidelity, job count) per rung started.

    A candidate's score is its pool index, so a lower index always wins.
    """
    started: list[tuple[str, int]] = []
    for _ in range(100):
        state = AllocationSnapshot(
            loop.spec, ENVIRONMENT, jobs, ledger.query(limit=10**6), 10**9, slots=10**6
        )
        allocation = policy.select(state)
        if not allocation.jobs:
            assert allocation.status is AllocationStatus.DONE
            return started
        assert len({item.inputs.fidelity for item in allocation.jobs}) == 1
        started.append((allocation.jobs[0].inputs.fidelity, len(allocation.jobs)))
        for offered in allocation.jobs:
            complete(ledger, loop.spec, offered, rank[offered.inputs.candidate_hash])
    raise AssertionError("Allocation did not finish")


def test_three_rung_halving_promotes_nine_three_one(ledger: Ledger) -> None:
    loop = geometric_loop(1, 3, 9)
    rungs = tuple(Rung(str(amount), (0,)) for amount in (1, 3, 9))
    jobs = tuple(
        job(loop, index, fidelity=rung.fidelity or "")
        for index in range(9)
        for rung in rungs
    )
    candidates = tuple(
        job(loop, index, fidelity="1").inputs.candidate_hash for index in range(9)
    )
    policy = SuccessiveHalving(candidates, rungs, eta=3)
    rank = {candidate: index for index, candidate in enumerate(candidates)}
    started = drive(policy, loop, ledger, jobs, rank)
    assert started == [("1", 9), ("3", 3), ("9", 1)]
    ran = {(run.inputs.fidelity, run.inputs.candidate_hash) for run in ledger.query()}
    assert {candidate for fidelity, candidate in ran if fidelity == "3"} == set(
        candidates[:3]
    )
    assert {candidate for fidelity, candidate in ran if fidelity == "9"} == {
        candidates[0]
    }


def test_five_rung_hyperband_runs_every_published_bracket_size(
    ledger: Ledger,
) -> None:
    amounts = (1, 3, 9, 27, 81)
    loop = geometric_loop(*amounts)
    rungs = tuple(Rung(str(amount), (0,)) for amount in amounts)
    candidates = tuple(
        job(loop, index, fidelity="1").inputs.candidate_hash for index in range(143)
    )
    jobs = tuple(
        job(loop, index, fidelity=rung.fidelity or "")
        for index in range(143)
        for rung in rungs
    )
    policy = Hyperband(candidates, rungs, eta=3, seed=3)
    assert policy.bracket_sizes == (81, 34, 15, 8, 5)
    rank = {candidate: index for index, candidate in enumerate(candidates)}
    started = drive(policy, loop, ledger, jobs, rank)
    assert started == [
        ("1", 81), ("3", 27), ("9", 9), ("27", 3), ("81", 1),
        ("3", 34), ("9", 11), ("27", 3), ("81", 1),
        ("9", 15), ("27", 5), ("81", 1),
        ("27", 8), ("81", 2),
        ("81", 5),
    ]  # fmt: skip
    assert sum(int(fidelity) * count for fidelity, count in started) == 1902
    assert len(ledger.query(limit=10**6)) == sum(count for _, count in started)
