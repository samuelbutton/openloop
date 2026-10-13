"""Virtual lifecycle, capacity, and replay fidelity; no processes or real waits."""

import asyncio
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

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

JOB = ProcessJob(("/not/an/executable",))
GPU_JOB = ProcessJob(("/not/an/executable",), gpu=True)
RESULT = Result({"val_bpb": Metric(1.43, "BPB")})


def recording(
    job: ProcessJob = JOB,
    *,
    seconds: float = 10,
    state: State = State.SUCCEEDED,
    score: float = 1.43,
    pool_key: str | None = None,
) -> RecordedRun:
    result = Result({"val_bpb": Metric(score, "BPB")})
    return RecordedRun(
        job.hash,
        f"{score:.4f}".encode().hex().ljust(64, "a")[:64],
        seconds,
        state,
        result=result if state is State.SUCCEEDED else None,
        pool_key=pool_key,
    )


def strict(*recordings: RecordedRun, **options: Any) -> MockExecutor:
    """Exact replay: each recording is served once, in input order."""
    return MockExecutor(recordings, strict=True, **options)


def test_submit_poll_and_collect_replay_without_executing_a_command() -> None:
    async def scenario() -> None:
        executor = strict(recording())
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
        executor = strict(recording(seconds=3), recording(seconds=7))
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
        executor = strict(
            recording(seconds=4),
            recording(seconds=2),
            recording(seconds=3),
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
        executor = strict(recording(state=state))
        identifier = await executor.submit(JOB, key="failure")
        receipt = await executor.collect(identifier)
        assert receipt.state is state
        assert receipt.recorded_result is None
        await executor.close()

    asyncio.run(scenario())


def test_wall_limit_suppresses_an_unfinished_recorded_result() -> None:
    async def scenario() -> None:
        limited = replace(JOB, limits=Limits(wall_seconds=5))
        executor = strict(recording(limited, seconds=10))
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
        executor = strict(recording(), recording(limited))
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
        executor = strict(recording(), recording(seconds=20), recording(seconds=30))
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
        executor = strict(recording(), recording(), recording())
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
        executor = strict(recording(seconds=0), recording(seconds=60))
        zero = await executor.submit(JOB, key="zero")
        exact = await executor.submit(JOB, key="exact")
        assert executor.advance(0)[0].job_id == zero
        assert (await executor.collect(exact)).state is State.SUCCEEDED
        assert executor.now == 60
        await executor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("seconds", [-1, True, float("nan"), float("inf"), 1e308])
def test_invalid_advance_leaves_clock_and_work_unchanged(seconds) -> None:
    executor = strict(recording())
    with pytest.raises(InvalidJobError):
        executor.advance(seconds)
    assert executor.now == 0


@given(count=st.integers(1, 20), workers=st.integers(1, 20), seconds=st.integers(0, 30))
def test_capacity_and_time_are_deterministic(
    count: int, workers: int, seconds: int
) -> None:
    async def scenario() -> None:
        executor = MockExecutor((recording(seconds=seconds),), workers=workers, seed=0)
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
        executor = MockExecutor((recording(),), workers=10_000, seed=0)
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


def test_gpu_jobs_contend_for_lanes_while_cpu_jobs_only_need_workers() -> None:
    async def scenario() -> None:
        executor = strict(
            recording(GPU_JOB, seconds=10),
            recording(GPU_JOB, seconds=10),
            recording(JOB, seconds=1),
            workers=4,
            gpu_lanes=1,
        )
        first = await executor.submit(GPU_JOB, key="g1")
        second = await executor.submit(GPU_JOB, key="g2")
        cpu = await executor.submit(JOB, key="c")
        # The second GPU job waits for the lane but holds no worker.
        assert (await executor.poll(first)).state is State.RUNNING
        assert (await executor.poll(second)).state is State.QUEUED
        assert (await executor.poll(cpu)).state is State.RUNNING
        assert (await executor.collect(second)).queued_seconds == 10
        assert executor.now == 20
        await executor.close()

    asyncio.run(scenario())


def test_more_lanes_remove_gpu_contention_and_workers_still_bind() -> None:
    async def scenario() -> None:
        wide = strict(recording(GPU_JOB), recording(GPU_JOB), workers=2, gpu_lanes=2)
        ids = [await wide.submit(GPU_JOB, key=str(n)) for n in range(2)]
        assert [(await wide.poll(i)).state for i in ids] == [State.RUNNING] * 2
        await wide.close()
        narrow = strict(recording(GPU_JOB), recording(GPU_JOB), workers=1, gpu_lanes=2)
        first = await narrow.submit(GPU_JOB, key="a")
        second = await narrow.submit(GPU_JOB, key="b")
        assert (await narrow.poll(first)).state is State.RUNNING
        assert (await narrow.poll(second)).state is State.QUEUED
        await narrow.close()

    asyncio.run(scenario())


def test_gpu_lane_release_on_cancel_starts_the_next_gpu_job() -> None:
    async def scenario() -> None:
        executor = strict(recording(GPU_JOB), recording(GPU_JOB), workers=2)
        first = await executor.submit(GPU_JOB, key="a")
        second = await executor.submit(GPU_JOB, key="b")
        await executor.cancel(first)
        assert (await executor.poll(second)).state is State.RUNNING
        await executor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("lanes", [0, -1, True, 1.5])
def test_invalid_gpu_lanes_are_rejected(lanes) -> None:
    with pytest.raises(InvalidJobError):
        strict(recording(), gpu_lanes=lanes)


def noisy_pool() -> tuple[RecordedRun, ...]:
    return tuple(
        recording(JOB, seconds=5 + index, score=1.0 + index / 10, pool_key="cand:8m")
        for index in range(4)
    )


def seeded_jobs(count: int) -> list[ProcessJob]:
    return [ProcessJob(("/not/an/executable", str(index))) for index in range(count)]


def pool_of(job: ProcessJob) -> str:
    return "cand:8m"


async def draws(executor: MockExecutor, jobs: list[ProcessJob]) -> list[str]:
    ids = [await executor.submit(job, key=str(n)) for n, job in enumerate(jobs)]
    return [(await executor.poll(i)).recording_hash for i in ids]  # type: ignore[attr-defined]


def test_resampling_is_seeded_deterministic_and_marks_reuse() -> None:
    async def scenario() -> None:
        jobs = seeded_jobs(12)
        first = MockExecutor(noisy_pool(), workers=12, seed=7, pool_of=pool_of)
        second = MockExecutor(noisy_pool(), workers=12, seed=7, pool_of=pool_of)
        other = MockExecutor(noisy_pool(), workers=12, seed=8, pool_of=pool_of)
        assert await draws(first, jobs) == await draws(second, jobs)
        assert await draws(other, jobs) != await draws(
            MockExecutor(noisy_pool(), workers=12, seed=7, pool_of=pool_of), jobs
        )
        receipts = [await first.poll(f"mock-{n}") for n in range(1, 13)]
        assert all(item.pool_key == "cand:8m" for item in receipts)
        # 12 draws from 4 recordings must reuse at least 8 of them.
        assert first.resampled_submissions >= 8
        seen: dict[str, int] = {}
        for item in receipts:
            assert item.reuse_count == seen.get(item.recording_hash, 0)
            assert item.resampled == (item.reuse_count > 0)
            seen[item.recording_hash] = seen.get(item.recording_hash, 0) + 1
        for executor in (first, second, other):
            await executor.close()

    asyncio.run(scenario())


def test_resampled_draws_do_not_depend_on_submission_interleaving() -> None:
    async def scenario() -> None:
        other_job = ProcessJob(("/not/an/executable", "other"))
        pools = (
            *noisy_pool(),
            recording(other_job, seconds=3, score=9.0, pool_key="other:8m"),
        )

        def pool(job: ProcessJob) -> str:
            return "other:8m" if job.argv[-1] == "other" else "cand:8m"

        jobs = seeded_jobs(6)
        straight = MockExecutor(pools, workers=8, seed=3, pool_of=pool)
        straight_hashes = await draws(straight, jobs)
        mixed = MockExecutor(pools, workers=8, seed=3, pool_of=pool)
        mixed_hashes: list[str] = []
        for n, job in enumerate(jobs):
            if n % 2:
                await mixed.submit(other_job, key=f"noise-{n}")
            identifier = await mixed.submit(job, key=f"c-{n}")
            mixed_hashes.append(
                (await mixed.poll(identifier)).recording_hash  # type: ignore[attr-defined]
            )
        assert mixed_hashes == straight_hashes
        await straight.close()
        await mixed.close()

    asyncio.run(scenario())


def test_resampling_repeats_scores_it_cannot_invent_new_ones() -> None:
    async def scenario() -> None:
        executor = MockExecutor(noisy_pool(), workers=50, seed=1, pool_of=pool_of)
        ids = [
            await executor.submit(job, key=str(n))
            for n, job in enumerate(seeded_jobs(50))
        ]
        scores = set()
        for identifier in ids:
            receipt = await executor.collect(identifier)
            assert receipt.recorded_result is not None
            scores.add(receipt.recorded_result.metrics["val_bpb"].value)
        assert scores <= {1.0, 1.1, 1.2, 1.3}
        await executor.close()

    asyncio.run(scenario())


def test_strict_mode_refuses_reuse_and_reports_none() -> None:
    async def scenario() -> None:
        executor = strict(*noisy_pool(), pool_of=pool_of, workers=4)
        jobs = seeded_jobs(5)
        ids = [await executor.submit(job, key=str(n)) for n, job in enumerate(jobs[:4])]
        with pytest.raises(InvalidJobError, match="remaining recording"):
            await executor.submit(jobs[4], key="fifth")
        receipts = [await executor.poll(i) for i in ids]
        assert len({item.recording_hash for item in receipts}) == 4
        assert not any(item.resampled for item in receipts)
        assert executor.resampled_submissions == 0
        await executor.close()

    asyncio.run(scenario())


def test_seed_is_required_unless_strict_and_unpooled_jobs_are_rejected() -> None:
    with pytest.raises(InvalidJobError, match="seed"):
        MockExecutor((recording(),))
    with pytest.raises(InvalidJobError, match="two pools"):
        MockExecutor((recording(pool_key="a"), recording(pool_key="b")), seed=0)

    async def scenario() -> None:
        executor = MockExecutor((recording(),), seed=0)
        with pytest.raises(InvalidJobError, match="pool"):
            await executor.submit(ProcessJob(("/other",)), key="x")
        await executor.close()

    asyncio.run(scenario())


def test_failed_submission_validation_does_not_consume_a_draw() -> None:
    async def scenario() -> None:
        huge = replace(JOB, limits=Limits(queue_seconds=1e300))
        executor = strict(recording(huge))
        with pytest.raises(InvalidJobError):
            await executor.submit(huge, key="overflow")
        # The pool is untouched: nothing was consumed by the rejected call.
        assert executor._draws == {}  # pyright: ignore[reportPrivateUsage]
        await executor.close()

    asyncio.run(scenario())


def test_wait_any_advances_to_the_first_finisher_and_breaks_ties_by_id() -> None:
    async def scenario() -> None:
        executor = strict(
            recording(seconds=5), recording(seconds=2), recording(seconds=2), workers=3
        )
        ids = [await executor.submit(JOB, key=str(n)) for n in range(3)]
        first = await executor.wait_any(ids)
        assert first.job_id == min(ids[1], ids[2])
        assert executor.now == 2
        assert (await executor.wait_any(ids)).job_id == first.job_id  # Still terminal.
        assert (await executor.wait_any([ids[0]])).job_id == ids[0]
        assert executor.now == 5
        with pytest.raises(InvalidJobError):
            await executor.wait_any([])
        with pytest.raises(UnknownJobError):
            await executor.wait_any(["unknown"])
        await executor.close()

    asyncio.run(scenario())


def test_cancel_and_close_carry_no_reason_like_the_real_executors() -> None:
    async def scenario() -> None:
        executor = strict(recording(), recording(), recording())
        running = await executor.submit(JOB, key="a")
        queued = await executor.submit(JOB, key="b")
        assert (await executor.cancel(running)).error is None
        assert (await executor.cancel(queued)).error is None
        assert (await executor.cancel(queued)).state is State.CANCELLED
        await executor.close()

    asyncio.run(scenario())
