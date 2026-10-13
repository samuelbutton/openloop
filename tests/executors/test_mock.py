"""Virtual lifecycle, capacity, and replay fidelity; no processes or real waits."""

import asyncio
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.executors import (
    Executor,
    ExecutorError,
    InvalidJobError,
    Limits,
    MockExecutor,
    ProcessJob,
    RecordedRun,
    State,
    SubmissionConflictError,
    UnknownJobError,
)
from openloop.ledger import Metric, Result

JOB = ProcessJob(("/not/an/executable",), gpu=True)
RESULT = Result({"val_bpb": Metric(1.43, "BPB")})


def recording(
    job: ProcessJob = JOB, *, seconds: float = 10, state: State = State.SUCCEEDED
) -> RecordedRun:
    return RecordedRun(
        job.hash,
        "a" * 64,
        seconds,
        state,
        result=RESULT if state is State.SUCCEEDED else None,
    )


def test_submit_poll_and_collect_replay_without_executing_a_command() -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(),))
        contract: Executor = executor
        identifier = await contract.submit(JOB, key="first")
        running = await executor.poll(identifier)
        assert running.simulated
        assert running.state is State.RUNNING
        assert running.recorded_result is None
        assert running.queued_seconds == 0
        assert executor.now == 0
        receipt = await executor.collect(identifier)
        assert receipt.state is State.SUCCEEDED
        assert receipt.recorded_result == RESULT
        assert receipt.elapsed_seconds == 10
        assert receipt.started_at == datetime(2000, 1, 1, tzinfo=UTC)
        assert receipt.started_at is not None
        assert receipt.finished_at == receipt.started_at + timedelta(seconds=10)
        assert receipt.artifacts == ()
        assert receipt.exit_code is None
        assert running.state is State.RUNNING  # Earlier snapshots remain immutable.
        assert executor.now == 10
        assert await executor.collect(identifier) == receipt
        await executor.close()

    asyncio.run(scenario())


def test_idempotency_and_recording_exhaustion() -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(seconds=3), recording(seconds=7)))
        first = await executor.submit(JOB, key="same")
        assert await executor.submit(JOB, key="same") == first
        with pytest.raises(SubmissionConflictError):
            await executor.submit(replace(JOB, stdin=b"changed"), key="same")
        second = await executor.submit(JOB, key="other")
        with pytest.raises(InvalidJobError, match="remaining recording"):
            await executor.submit(JOB, key="third")
        assert (await executor.collect(second)).elapsed_seconds == 7
        assert (await executor.collect(first)).elapsed_seconds == 3
        assert executor.now == 10
        await executor.close()

    asyncio.run(scenario())


def test_fifo_workers_and_collect_advance_the_shared_clock() -> None:
    async def scenario() -> None:
        executor = MockExecutor(
            (recording(seconds=4), recording(seconds=2), recording(seconds=3)),
            workers=2,
        )
        first, second, third = [
            await executor.submit(JOB, key=str(index)) for index in range(3)
        ]
        assert (await executor.poll(third)).state is State.QUEUED
        finished = executor.advance(2)
        assert tuple(item.job_id for item in finished) == (second,)
        assert (await executor.poll(third)).queued_seconds == 2
        result = await executor.collect(third)
        assert result.elapsed_seconds == 3
        assert (await executor.poll(first)).state is State.SUCCEEDED
        assert executor.now == 5
        await executor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("state", [State.FAILED, State.CANCELLED, State.TIMED_OUT])
def test_recorded_unsuccessful_outcomes_are_preserved(state: State) -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(state=state),))
        identifier = await executor.submit(JOB, key="failure")
        receipt = await executor.collect(identifier)
        assert receipt.state is state
        assert receipt.recorded_result is None
        await executor.close()

    asyncio.run(scenario())


def test_wall_limit_suppresses_an_unfinished_recorded_result() -> None:
    async def scenario() -> None:
        limited = replace(JOB, limits=Limits(wall_seconds=5))
        executor = MockExecutor((recording(limited, seconds=10),))
        identifier = await executor.submit(limited, key="timeout")
        receipt = await executor.collect(identifier)
        assert receipt.state is State.TIMED_OUT
        assert receipt.elapsed_seconds == 5
        assert receipt.recorded_result is None
        assert receipt.error == "Simulated wall limit exceeded"
        await executor.close()

    asyncio.run(scenario())


def test_queue_deadline_wins_a_tie_and_unstarted_work_has_no_duration() -> None:
    async def scenario() -> None:
        limited = replace(JOB, limits=Limits(queue_seconds=10))
        executor = MockExecutor((recording(), recording(limited)))
        first = await executor.submit(JOB, key="occupy")
        queued = await executor.submit(limited, key="expire")
        receipt = await executor.collect(queued)
        assert receipt.state is State.EXPIRED
        assert receipt.started_at is None
        assert receipt.queued_seconds is None
        assert receipt.elapsed_seconds is None
        assert receipt.recorded_result is None
        assert (await executor.poll(first)).state is State.SUCCEEDED
        await executor.close()

    asyncio.run(scenario())


def test_cancellation_frees_capacity_and_stale_events_do_not_finish_other_jobs() -> (
    None
):
    async def scenario() -> None:
        executor = MockExecutor(
            (recording(), recording(seconds=20), recording(seconds=30))
        )
        first, second, third = [
            await executor.submit(JOB, key=str(index)) for index in range(3)
        ]
        executor.advance(3)
        cancelled = await executor.cancel(first)
        assert cancelled.elapsed_seconds == 3
        assert cancelled.recorded_result is None
        assert (await executor.poll(second)).queued_seconds == 3
        queued = await executor.cancel(third)
        assert queued.started_at is None
        assert executor.advance(7) == ()
        assert (await executor.poll(second)).state is State.RUNNING
        assert (await executor.collect(second)).elapsed_seconds == 20
        assert executor.now == 23
        assert await executor.cancel(first) == cancelled
        await executor.close()

    asyncio.run(scenario())


def test_close_cancels_without_starting_queued_jobs_and_rejects_submission() -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(), recording(), recording()))
        first = await executor.submit(JOB, key="first")
        queued = await executor.submit(JOB, key="queued")
        await executor.close()
        assert (await executor.poll(first)).state is State.CANCELLED
        assert (await executor.poll(queued)).started_at is None
        await executor.close()
        with pytest.raises(ExecutorError, match="closed"):
            await executor.submit(JOB, key="after-close")
        for operation in (executor.poll, executor.cancel, executor.collect):
            with pytest.raises(UnknownJobError):
                await operation("unknown")

    asyncio.run(scenario())


def test_zero_duration_and_exact_wall_limit_complete() -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(seconds=0), recording(seconds=60)))
        zero = await executor.submit(JOB, key="zero")
        exact = await executor.submit(JOB, key="exact")
        assert executor.advance(0)[0].job_id == zero
        assert (await executor.collect(exact)).state is State.SUCCEEDED
        assert executor.now == 60
        await executor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("seconds", [-1, True, float("nan"), float("inf"), 1e308])
def test_invalid_advance_leaves_clock_and_work_unchanged(seconds) -> None:
    executor = MockExecutor((recording(),))
    with pytest.raises(InvalidJobError):
        executor.advance(seconds)
    assert executor.now == 0


@given(count=st.integers(1, 20), workers=st.integers(1, 20), seconds=st.integers(0, 30))
def test_capacity_and_time_are_deterministic(
    count: int, workers: int, seconds: int
) -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(seconds=seconds),) * count, workers=workers)
        identifiers = [
            await executor.submit(JOB, key=str(index)) for index in range(count)
        ]
        assert len(set(identifiers)) == count
        receipts = [await executor.collect(identifier) for identifier in identifiers]
        assert executor.now == math.ceil(count / workers) * seconds
        assert sum(item.elapsed_seconds or 0 for item in receipts) == count * seconds
        for index, item in enumerate(receipts):
            assert item.queued_seconds == (index // workers) * seconds
            assert item.simulated
        await executor.close()

    asyncio.run(scenario())


def test_ten_thousand_workers_replay_ten_thousand_jobs() -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(),) * 10_000, workers=10_000)
        identifiers = await asyncio.gather(
            *(executor.submit(JOB, key=str(index)) for index in range(10_000))
        )
        assert len(set(identifiers)) == 10_000
        receipts = executor.advance(10)
        assert len(receipts) == 10_000
        assert all(item.recorded_result == RESULT for item in receipts)
        assert executor.now == 10
        await executor.close()

    asyncio.run(scenario())
