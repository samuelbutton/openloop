"""One slot-scheduling loop, written once against the Executor protocol.

It must complete jobs in the same order on real processes and on the mock's
virtual clock, with no sleep-polling.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from openloop.executors import (
    Executor,
    InvalidJobError,
    LocalExecutor,
    MockExecutor,
    ProcessJob,
    RecordedRun,
    State,
    UnknownJobError,
)

UNIT = 0.3  # Seconds per duration unit; gaps of one unit absorb process start-up.
DURATIONS = {"A": 4, "B": 1, "C": 2, "D": 2}
JOBS = {
    name: ProcessJob(
        (sys.executable, "-c", f"import time; time.sleep({units * UNIT})", name),
        reviewed=True,
    )
    for name, units in DURATIONS.items()
}


async def run_slots(executor: Executor, order: list[str], slots: int) -> list[str]:
    """Keep up to `slots` jobs in flight; return names in completion order."""
    pending: dict[str, str] = {}
    queue = list(order)
    done: list[str] = []
    while queue or pending:
        while queue and len(pending) < slots:
            name = queue.pop(0)
            pending[await executor.submit(JOBS[name], key=name)] = name
        finished = await executor.wait_any(pending)
        assert finished.state is State.SUCCEEDED
        done.append(pending.pop(finished.job_id))
    return done


def test_same_slot_loop_has_identical_completion_order_on_local_and_mock(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        local = LocalExecutor(tmp_path)
        mock = MockExecutor(
            tuple(
                RecordedRun(job.hash, "a" * 64, DURATIONS[name], State.SUCCEEDED)
                for name, job in JOBS.items()
            ),
            workers=2,
            strict=True,
        )
        try:
            expected = ["B", "C", "A", "D"]
            order = list(DURATIONS)
            assert await run_slots(mock, order, 2) == expected
            assert mock.now == 5
            assert await run_slots(local, order, 2) == expected
        finally:
            await local.close()
            await mock.close()

    asyncio.run(scenario())


def test_local_wait_any_validation_ties_and_cancelled_waiters(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = LocalExecutor(tmp_path)
        try:
            with pytest.raises(InvalidJobError):
                await executor.wait_any([])
            with pytest.raises(UnknownJobError):
                await executor.wait_any(["unknown"])
            quick = ProcessJob((sys.executable, "-c", "pass"), reviewed=True)
            slow = ProcessJob(
                (sys.executable, "-c", "import time; time.sleep(30)"), reviewed=True
            )
            fast_id = await executor.submit(quick, key="quick")
            slow_id = await executor.submit(slow, key="slow")
            first = await executor.wait_any([slow_id, fast_id])
            assert first.job_id == fast_id
            # Already terminal: returns at once, and cancelling a waiter keeps jobs.
            assert (await executor.wait_any([fast_id, slow_id])).job_id == fast_id
            waiter = asyncio.create_task(executor.wait_any([slow_id]))
            await asyncio.sleep(0.2)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert (await executor.poll(slow_id)).state in (
                State.QUEUED,
                State.RUNNING,
            )
            assert (await executor.wait_any([slow_id, fast_id])).job_id == fast_id
            assert (await executor.cancel(slow_id)).state is State.CANCELLED
        finally:
            await executor.close()

    asyncio.run(scenario())
