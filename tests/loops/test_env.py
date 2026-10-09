"""One candidate episode, shared ledger evidence, and guarded work admission."""

import asyncio
from dataclasses import replace

import pytest

from openloop.ledger import Ledger, Purpose, Status
from openloop.loops import (
    T0,
    ContractError,
    ExperimentEnv,
    FidelityName,
    Job,
    JobOutput,
    LoopAction,
    LoopEnv,
)


def episode(t0: T0, ledger: Ledger, budget: int = 5) -> ExperimentEnv:
    return ExperimentEnv(
        t0,
        ledger,
        runner=t0.run,
        config={"coordinates": [0, 0]},
        environment_hash="a" * 64,
        budget=budget,
    )


def test_single_contract_runs_both_fidelities(t0: T0, ledger: Ledger) -> None:
    env: LoopEnv = episode(t0, ledger)
    initial = asyncio.run(env.initial_observation())
    screen = asyncio.run(env.step(LoopAction(42)))
    confirm = asyncio.run(env.step(LoopAction(43, FidelityName.CONFIRM)))
    assert initial.remaining_budget == 5
    assert screen.observation.remaining_budget == 4
    assert confirm.observation.remaining_budget == 0
    assert confirm.episode_done
    assert initial.candidate_hash == confirm.observation.candidate_hash
    assert screen.work_units == 1
    assert confirm.work_units == 4
    assert len(ledger.query(status="succeeded")) == 2
    with pytest.raises(ContractError, match="remaining"):
        asyncio.run(env.step(LoopAction(44)))
    assert len(ledger.query()) == 2


def test_completed_cache_hit_is_not_new_work(t0: T0, ledger: Ledger) -> None:
    env = episode(t0, ledger)
    first = asyncio.run(env.step(LoopAction(42)))
    reused = asyncio.run(env.step(LoopAction(42)))
    assert first.executed
    assert not reused.executed
    assert reused.work_units == 0
    assert reused.attempt_id == first.attempt_id
    assert reused.observation.remaining_budget == 4
    assert len(ledger.query()) == 1
    replication = asyncio.run(env.step(LoopAction(42, purpose=Purpose.REPLICATION)))
    assert replication.attempt_id != first.attempt_id
    assert replication.observation.remaining_budget == 3


def test_request_replay_does_not_charge_twice(t0: T0, ledger: Ledger) -> None:
    env = episode(t0, ledger, budget=1)
    action = LoopAction(42, request_key="one-action")
    first = asyncio.run(env.step(action))
    assert asyncio.run(env.step(action)) == first
    with pytest.raises(ContractError, match="different action"):
        asyncio.run(env.step(replace(action, seed=43)))
    assert len(ledger.query()) == 1


def test_runner_failure_preserves_evidence_and_reservation(
    t0: T0, ledger: Ledger
) -> None:
    async def crash(job: Job) -> JobOutput:
        raise RuntimeError("Worker crashed")

    env = ExperimentEnv(
        t0,
        ledger,
        runner=crash,
        config={"coordinates": [0, 0]},
        environment_hash="a" * 64,
        budget=1,
    )
    with pytest.raises(RuntimeError, match="Worker crashed"):
        asyncio.run(env.step(LoopAction(42)))
    assert asyncio.run(env.initial_observation()).remaining_budget == 0
    (failed,) = ledger.query()
    assert failed.status is Status.FAILED
    assert failed.events[-1].payload["reason"] == "RuntimeError: Worker crashed"


def test_invalid_output_fails_without_completed_evidence(
    t0: T0, ledger: Ledger
) -> None:
    async def invalid(job: Job) -> JobOutput:
        return {"samples": (1.0,), "work_units": 999}

    env = ExperimentEnv(
        t0,
        ledger,
        runner=invalid,
        config={"coordinates": [0, 0]},
        environment_hash="a" * 64,
        budget=1,
    )
    with pytest.raises(ContractError, match="declared fidelity"):
        asyncio.run(env.step(LoopAction(42)))
    assert ledger.query(status="succeeded") == ()
    assert len(ledger.query(status="failed")) == 1


def test_concurrent_step_is_rejected_before_admission(t0: T0, ledger: Ledger) -> None:
    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(job: Job) -> JobOutput:
            entered.set()
            await release.wait()
            return await t0.run(job)

        env = ExperimentEnv(
            t0,
            ledger,
            runner=slow,
            config={"coordinates": [0, 0]},
            environment_hash="a" * 64,
            budget=2,
        )
        pending = asyncio.create_task(env.step(LoopAction(42)))
        await entered.wait()
        with pytest.raises(ContractError, match="concurrent"):
            await env.step(LoopAction(43))
        release.set()
        await pending

    asyncio.run(scenario())
    assert len(ledger.query()) == 1


def test_loop_cannot_change_its_frozen_spec(t0: T0, ledger: Ledger) -> None:
    env = episode(t0, ledger)
    t0._spec = replace(t0.spec, data_hash="b" * 64)
    with pytest.raises(ContractError, match="specification changed"):
        asyncio.run(env.step(LoopAction(42)))
    assert ledger.query() == ()


def test_contract_change_during_execution_cannot_publish(
    t0: T0, ledger: Ledger
) -> None:
    async def change(job: Job) -> JobOutput:
        output = await t0.run(job)
        t0._spec = replace(t0.spec, evaluator_hash="b" * 64)
        return output

    env = ExperimentEnv(
        t0,
        ledger,
        runner=change,
        config={"coordinates": [0, 0]},
        environment_hash="a" * 64,
        budget=1,
    )
    with pytest.raises(ContractError, match="changed during execution"):
        asyncio.run(env.step(LoopAction(42)))
    assert ledger.query(status="succeeded") == ()
    assert len(ledger.query(status="failed")) == 1
