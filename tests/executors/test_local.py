"""Reviewed local CPU jobs. GPU lane tests never import a GPU library."""

import asyncio
import hashlib
import os
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from openloop.executors import (
    ExecutorError,
    InvalidJobError,
    Limits,
    LocalExecutor,
    ProcessJob,
    State,
    SubmissionConflictError,
    UnknownJobError,
    local,
)


def job(program: str, *, limits: Limits | None = None, gpu: bool = False) -> ProcessJob:
    return ProcessJob(
        (sys.executable, "-c", program),
        reviewed=True,
        limits=limits or Limits(),
        gpu=gpu,
    )


def test_receipt_logs_times_and_idempotency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENLOOP_PARENT_MARKER", "not-inherited")

    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        command = replace(
            job(
                "import os,sys; print(os.getenv('OPENLOOP_PARENT_MARKER')); "
                "print(sys.stdin.read()); sys.stderr.write('err')"
            ),
            stdin=b"input",
        )
        identifier = await executor.submit(command, key="one")
        queued = await executor.poll(identifier)
        assert queued.state == State.QUEUED
        assert await executor.submit(command, key="one") == identifier
        with pytest.raises(SubmissionConflictError):
            await executor.submit(job("print('different')"), key="one")
        result = await executor.collect(identifier)
        assert result.state == State.SUCCEEDED
        assert result.exit_code == 0
        assert queued.started_at is None
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.submitted_at <= result.started_at <= result.finished_at
        assert result.elapsed_seconds is not None
        stdout, stderr = result.artifacts
        assert stdout.path.read_bytes() == b"None\ninput\n"
        assert stderr.path.read_bytes() == b"err"
        assert stdout.sha256 == hashlib.sha256(stdout.path.read_bytes()).hexdigest()
        assert await executor.collect(identifier) == result
        assert await executor.cancel(identifier) == result
        await executor.close()
        with pytest.raises(ExecutorError, match="closed"):
            await executor.submit(command, key="closed")

    asyncio.run(scenario())


def test_unreviewed_source_and_unknown_jobs_are_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        with pytest.raises(InvalidJobError, match="reviewed"):
            await executor.submit(ProcessJob(("true",)), key="unsafe")
        with pytest.raises(UnknownJobError):
            await executor.collect("missing")
        assert list(tmp_path.iterdir()) == []
        await executor.close()

    asyncio.run(scenario())


def test_exit_failure_and_launch_failure_are_receipts(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        failed = await executor.submit(job("raise SystemExit(7)"), key="exit")
        result = await executor.collect(failed)
        assert result.state == State.FAILED
        assert result.exit_code == 7
        missing = await executor.submit(
            ProcessJob(("/nonexistent/openloop-test",), reviewed=True), key="missing"
        )
        result = await executor.collect(missing)
        assert result.state == State.FAILED
        assert result.exit_code != 0
        await executor.close()

    asyncio.run(scenario())


def test_wall_and_log_limits_stop_workers(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        sleeping = await executor.submit(
            job("import time; time.sleep(20)", limits=Limits(wall_seconds=0.2)),
            key="time",
        )
        result = await executor.collect(sleeping)
        assert result.state == State.TIMED_OUT
        assert result.exit_code == -9
        noisy = await executor.submit(
            job(
                "import sys; sys.stdout.write('x'*100000)",
                limits=Limits(output_bytes=1000),
            ),
            key="log",
        )
        result = await executor.collect(noisy)
        assert result.state == State.FAILED
        assert result.error == "Declared output limit exceeded"
        assert result.artifacts[0].size_bytes == 1000
        await executor.close()

    asyncio.run(scenario())


def test_cpu_and_file_limits_apply_only_when_set(tmp_path: Path) -> None:
    program = (
        "import resource; "
        "print(resource.getrlimit(resource.RLIMIT_CPU)); "
        "print(resource.getrlimit(resource.RLIMIT_FSIZE))"
    )

    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        unset = await executor.submit(
            job(program, limits=Limits(memory_bytes=536870912, cpus=2)), key="unset"
        )
        result = await executor.collect(unset)
        inherited = f"{resource.getrlimit(resource.RLIMIT_CPU)}\n"
        inherited += f"{resource.getrlimit(resource.RLIMIT_FSIZE)}\n"
        assert result.state == State.SUCCEEDED
        assert result.artifacts[0].path.read_text() == inherited
        limited = await executor.submit(
            job(
                program,
                limits=Limits(
                    memory_bytes=536870912,
                    cpus=2,
                    cpu_seconds=9.5,
                    file_bytes=123456,
                ),
            ),
            key="set",
        )
        result = await executor.collect(limited)
        assert result.state == State.SUCCEEDED
        assert result.artifacts[0].path.read_text() == "(10, 10)\n(123456, 123456)\n"
        assert result.peak_rss_bytes is not None
        await executor.close()

    asyncio.run(scenario())


def test_log_limit_does_not_limit_other_files(tmp_path: Path) -> None:
    target = tmp_path / "big"

    async def scenario() -> None:
        executor = LocalExecutor(tmp_path / "logs")
        identifier = await executor.submit(
            job(
                f"open({str(target)!r}, 'wb').write(b'x' * 100000)",
                limits=Limits(output_bytes=1000),
            ),
            key="file",
        )
        assert (await executor.collect(identifier)).state == State.SUCCEEDED
        await executor.close()

    asyncio.run(scenario())
    assert target.stat().st_size == 100000


def test_memory_sampling_interval_is_configurable(tmp_path: Path, monkeypatch) -> None:
    intervals: list[float] = []
    real_wait = local.wait_for_exit

    async def record(process: asyncio.subprocess.Process, seconds: float) -> None:
        intervals.append(seconds)
        await real_wait(process, 0.01)

    monkeypatch.setattr(local, "wait_for_exit", record)

    async def scenario() -> None:
        assert LocalExecutor(tmp_path / "default").memory_sample_seconds == 1.0
        with pytest.raises(InvalidJobError):
            LocalExecutor(tmp_path / "invalid", memory_sample_seconds=0)
        executor = LocalExecutor(tmp_path / "custom", memory_sample_seconds=0.37)
        identifier = await executor.submit(
            job("import time; time.sleep(0.2)"), key="sample"
        )
        assert (await executor.collect(identifier)).state == State.SUCCEEDED
        await executor.close()

    asyncio.run(scenario())
    assert intervals
    assert set(intervals) == {0.37}


def test_oversized_allocation_is_stopped(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        identifier = await executor.submit(
            job(
                "import time; data=bytearray(128*1024**2); time.sleep(3)",
                limits=Limits(memory_bytes=64 * 1024**2),
            ),
            key="memory",
        )
        result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert result.exit_code != 0
        await executor.close()

    asyncio.run(scenario())


async def wait_file(path: Path) -> None:
    async with asyncio.timeout(5):
        while not path.exists():
            await asyncio.sleep(0.01)


def test_cancel_reaps_process_group_and_can_run_twice(tmp_path: Path) -> None:
    marker = tmp_path / "child-pid"
    program = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
        f"Path({str(marker)!r}).write_text(str(child.pid)); time.sleep(20)"
    )

    async def scenario() -> None:
        executor = LocalExecutor(tmp_path / "logs")
        identifier = await executor.submit(job(program), key="cancel")
        await wait_file(marker)
        results = await asyncio.gather(
            executor.cancel(identifier), executor.cancel(identifier)
        )
        assert all(result.state == State.CANCELLED for result in results)
        assert (await executor.collect(identifier)).state == State.CANCELLED
        await executor.close()

    asyncio.run(scenario())
    # The grandchild may be a zombie briefly until init reaps it.
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_gpu_lane_serializes_separate_executors(tmp_path: Path) -> None:
    marker = tmp_path / "first"
    second = tmp_path / "second"

    async def scenario() -> None:
        first_executor = LocalExecutor(tmp_path / "a")
        second_executor = LocalExecutor(tmp_path / "b")
        first = await first_executor.submit(
            job(
                "import time; from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('ready'); time.sleep(20)",
                gpu=True,
            ),
            key="first",
        )
        await wait_file(marker)
        queued = await second_executor.submit(
            job(
                f"from pathlib import Path; Path({str(second)!r}).write_text('ready')",
                gpu=True,
            ),
            key="second",
        )
        await asyncio.sleep(0.1)
        assert (await second_executor.poll(queued)).state == State.QUEUED
        assert not second.exists()
        await first_executor.cancel(first)
        assert (await second_executor.collect(queued)).state == State.SUCCEEDED
        assert second.exists()
        await first_executor.close()
        await second_executor.close()

    asyncio.run(scenario())


def test_cancelling_collect_does_not_cancel_job(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        identifier = await executor.submit(
            job("import time; time.sleep(0.15)"), key="collect"
        )
        collector = asyncio.create_task(executor.collect(identifier))
        await asyncio.sleep(0.02)
        collector.cancel()
        with pytest.raises(asyncio.CancelledError):
            await collector
        assert (await executor.collect(identifier)).state == State.SUCCEEDED
        immediate = await executor.submit(job("raise SystemExit(123)"), key="queued")
        cancelled = await executor.cancel(immediate)
        assert cancelled.state == State.CANCELLED
        assert cancelled.started_at is None
        await executor.close()

    asyncio.run(scenario())


def test_queue_limit_expires_job_that_never_started(tmp_path: Path) -> None:
    marker = tmp_path / "second"

    async def scenario() -> None:
        first_executor = LocalExecutor(tmp_path / "a")
        second_executor = LocalExecutor(tmp_path / "b")
        first = await first_executor.submit(
            job("import time; time.sleep(20)", gpu=True), key="first"
        )
        await wait_running(first_executor, first)
        queued = await second_executor.submit(
            job(
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                limits=Limits(queue_seconds=0.3, wall_seconds=60),
                gpu=True,
            ),
            key="second",
        )
        result = await second_executor.collect(queued)
        assert result.state == State.EXPIRED
        assert "never started" in (result.error or "")
        assert result.started_at is None
        assert result.queued_seconds is None
        assert result.elapsed_seconds is None
        assert result.exit_code is None
        assert result.artifacts == ()
        await first_executor.cancel(first)
        await first_executor.close()
        await second_executor.close()

    asyncio.run(scenario())
    assert not marker.exists()


async def wait_running(executor: LocalExecutor, identifier: str) -> None:
    async with asyncio.timeout(5):
        while (await executor.poll(identifier)).state != State.RUNNING:
            await asyncio.sleep(0.01)
    # Let the child take its place before another executor contends for the lane.
    await asyncio.sleep(0.2)


def test_wall_limit_starts_at_process_start_not_in_the_queue(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_executor = LocalExecutor(tmp_path / "a")
        second_executor = LocalExecutor(tmp_path / "b")
        first = await first_executor.submit(
            job("import time; time.sleep(20)", gpu=True), key="first"
        )
        await wait_running(first_executor, first)
        waiting = await second_executor.submit(
            job(
                "raise SystemExit(0)",
                limits=Limits(wall_seconds=1.0),
                gpu=True,
            ),
            key="waiter",
        )
        # The waiter outlives its wall limit in the queue and must still run.
        await asyncio.sleep(1.3)
        assert (await second_executor.poll(waiting)).state == State.QUEUED
        await first_executor.cancel(first)
        result = await second_executor.collect(waiting)
        assert result.state == State.SUCCEEDED
        assert result.queued_seconds is not None
        assert result.queued_seconds >= 1.2
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds < result.queued_seconds
        await first_executor.close()
        await second_executor.close()

    asyncio.run(scenario())


def test_started_job_times_out_on_wall_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        identifier = await executor.submit(
            job(
                "import time; time.sleep(20)",
                limits=Limits(wall_seconds=0.3, queue_seconds=30),
            ),
            key="wall",
        )
        result = await executor.collect(identifier)
        assert result.state == State.TIMED_OUT
        assert result.started_at is not None
        assert result.queued_seconds is not None
        assert result.elapsed_seconds is not None
        assert 0.25 <= result.elapsed_seconds < 5
        await executor.close()

    asyncio.run(scenario())


def test_queue_limit_already_passed_at_dispatch_expires(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        identifier = await executor.submit(
            job("raise SystemExit(123)", limits=Limits(queue_seconds=0.02)),
            key="delayed",
        )
        # Model a coordinator that cannot dispatch its accepted job immediately.
        time.sleep(0.04)
        result = await executor.collect(identifier)
        assert result.state == State.EXPIRED
        assert result.started_at is None
        assert result.exit_code is None
        assert result.artifacts == ()
        await executor.close()

    asyncio.run(scenario())
