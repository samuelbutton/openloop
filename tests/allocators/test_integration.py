"""CPU simulations through the real loop contract, isolated ledgers, and policies."""

import asyncio

import pytest

from openloop.allocators import (
    AllocationSnapshot,
    Allocator,
    Bandit,
    Greedy,
    Hyperband,
    RandomSearch,
    Rung,
    SuccessiveHalving,
)
from openloop.ledger import Ledger
from openloop.loops import (
    T0,
    ExperimentEnv,
    Job,
    LoopAction,
    NASBench201,
    PlantedTruth,
    TabularLoop,
)
from openloop.loops.tabular import Benchmark, RecordedOutcome, RecordedTable

from .helpers import ENVIRONMENT, job


def execute(
    policy: Allocator,
    loop: T0 | TabularLoop,
    ledger: Ledger,
    jobs: tuple[Job, ...],
    budget: int,
) -> int:
    envs = {
        offered.inputs.candidate_hash: ExperimentEnv(
            loop,
            ledger,
            runner=loop.run,
            config=offered.inputs.config,
            environment_hash=ENVIRONMENT,
            budget=budget,
        )
        for offered in jobs
    }
    consumed = 0
    for _ in range(len(jobs) + 1):
        state = AllocationSnapshot(
            loop.spec, ENVIRONMENT, jobs, ledger.query(), budget - consumed, slots=2
        )
        selected = policy.select(state)
        if not selected.jobs:
            return consumed
        assert selected.work_units <= budget - consumed
        for offered in selected.jobs:
            inputs = offered.inputs
            result = asyncio.run(
                envs[inputs.candidate_hash].step(
                    LoopAction(inputs.seed, inputs.fidelity)
                )
            )
            assert result.executed
            consumed += result.work_units
    pytest.fail("Allocation did not terminate within its finite offers")


@pytest.mark.parametrize("name", ["greedy", "bandit", "random", "halving", "hyperband"])
def test_all_policies_execute_through_t0_without_duplicate_work(
    ledger: Ledger, name: str
) -> None:
    loop = T0(PlantedTruth(noise_std=0))
    jobs = tuple(
        job(loop, index, seed=seed, fidelity=fidelity)
        for index in range(6)
        for fidelity, seed in (("1x", 0), ("4x", 1))
    )
    candidates = tuple(job(loop, index).inputs.candidate_hash for index in range(6))
    rungs = (Rung("1x", (0,)), Rung("4x", (1,)))
    policies: dict[str, Allocator] = {
        "greedy": Greedy("1x"),
        "bandit": Bandit("1x", 0, 30),
        "random": RandomSearch("1x", seed=42),
        "halving": SuccessiveHalving(candidates, rungs, eta=2),
        "hyperband": Hyperband(candidates, rungs, eta=2, seed=42),
    }
    consumed = execute(policies[name], loop, ledger, jobs, 100)
    runs = ledger.query()
    assert len({run.inputs.hash for run in runs}) == len(runs)
    assert consumed == sum(run.inputs.budget_amount for run in runs)
    assert consumed == (18 if name == "halving" else 14 if name == "hyperband" else 6)
    if name == "halving":
        promoted = {
            run.inputs.candidate_hash for run in runs if run.inputs.fidelity == "4x"
        }
        assert job(loop, 1).inputs.candidate_hash in promoted


def test_nas_halving_uses_available_seed_per_fidelity_and_full_work_cost(
    ledger: Ledger,
) -> None:
    table = RecordedTable(
        Benchmark.NAS201,
        {"fixture": "simulated"},
        "b" * 64,
        tuple(f"architecture {index}" for index in range(4)),
        tuple(
            RecordedOutcome(index, fidelity, seed, 70 + index)
            for index in range(4)
            for fidelity, seed in (("12", 111), ("200", 777))
        ),
    )
    loop = NASBench201(table)
    configs = tuple(
        loop.normalize_config({"architecture_index": index}) for index in range(4)
    )
    rungs = (Rung("12", (111,)), Rung("200", (777,)))
    jobs = tuple(
        loop.build_job(loop.spec.inputs(config, action, ENVIRONMENT))
        for config in configs
        for action in (LoopAction(111, "12"), LoopAction(777, "200"))
    )
    policy = SuccessiveHalving(
        tuple(loop.spec.candidate_hash(config) for config in configs), rungs, eta=2
    )
    assert execute(policy, loop, ledger, jobs, 448) == 448
    promoted = {
        run.inputs.candidate_hash
        for run in ledger.query()
        if run.inputs.fidelity == "200"
    }
    assert promoted == {loop.spec.candidate_hash(configs[index]) for index in (2, 3)}
