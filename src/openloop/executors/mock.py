"""Finite virtual-time replay. Never execute commands or access their inputs.

Ownership, FIFO capacity, and clock state are in memory. No ledger, artifact,
network, subprocess, or real-time waits are involved. Each receipt is simulated.
"""

from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from heapq import heappop, heappush

from .models import (
    ExecutorError,
    InvalidJobError,
    ProcessJob,
    State,
    SubmissionConflictError,
    UnknownJobError,
)
from .replay import RecordedRun, ReplaySnapshot, duration


@dataclass
class _Replay:
    job: ProcessJob
    recording: RecordedRun
    snapshot: ReplaySnapshot
    submitted: float
    started: float | None = None


class MockExecutor:
    """Replay explicit recordings on identical simulated workers.

    poll never advances time. advance(seconds) processes all events in that
    interval. collect advances the shared clock until that job is terminal,
    so other jobs can complete too. New keys consume recordings in input order;
    repeated keys are idempotent. Exhaustion never samples with replacement.
    """

    def __init__(
        self,
        recordings: tuple[RecordedRun, ...],
        *,
        workers: int = 1,
        epoch: datetime = datetime(2000, 1, 1, tzinfo=UTC),
    ) -> None:
        if type(workers) is not int or workers < 1:
            raise InvalidJobError("workers must be a positive integer")
        if not isinstance(epoch, datetime) or epoch.utcoffset() is None:
            raise InvalidJobError("Simulation epoch must be an aware datetime")
        self._epoch = epoch.astimezone(UTC)
        self._workers = workers
        self._time = 0.0
        self._recordings: dict[str, deque[RecordedRun]] = {}
        for recording in recordings:
            if not isinstance(recording, RecordedRun):
                raise InvalidJobError("Replay requires typed recordings")
            self._timestamp(recording.duration_seconds)
            self._recordings.setdefault(recording.job_hash, deque()).append(recording)
        self._runs: dict[str, _Replay] = {}
        self._keys: dict[str, str] = {}
        self._queue: deque[str] = deque()
        self._events: list[tuple[float, int, int, str]] = []
        self._event_sequence = 0
        self._active = 0
        self._closed = False

    @property
    def now(self) -> float:
        """Simulated seconds since epoch, not host elapsed time."""
        return self._time

    @property
    def workers(self) -> int:
        return self._workers

    def _timestamp(self, seconds: float) -> datetime:
        try:
            return self._epoch + timedelta(seconds=seconds)
        except (OverflowError, ValueError) as error:
            raise InvalidJobError(
                "Simulated time exceeds the datetime range"
            ) from error

    def _event(self, when: float, kind: int, job_id: str) -> None:
        self._event_sequence += 1
        heappush(self._events, (when, kind, self._event_sequence, job_id))

    async def submit(self, job: ProcessJob, *, key: str) -> str:
        if self._closed:
            raise ExecutorError("Executor is closed")
        if (
            not isinstance(job, ProcessJob)
            or not isinstance(key, str)
            or not key.strip()
        ):
            raise InvalidJobError("A typed job and idempotency key are required")
        digest = job.hash
        if key in self._keys:
            job_id = self._keys[key]
            if self._runs[job_id].snapshot.input_hash != digest:
                raise SubmissionConflictError("Key already identifies different inputs")
            return job_id
        available = self._recordings.get(digest)
        if not available:
            raise InvalidJobError("No remaining recording for these exact inputs")
        recording = available[0]
        self._timestamp(
            self._time + min(recording.duration_seconds, job.limits.wall_seconds)
        )
        if job.limits.queue_seconds is not None:
            self._timestamp(self._time + job.limits.queue_seconds)
        job_id = f"mock-{len(self._runs) + 1}"
        snapshot = ReplaySnapshot(
            job_id,
            digest,
            State.QUEUED,
            self._timestamp(self._time),
            recording_hash=recording.hash,
        )
        self._runs[job_id] = _Replay(job, available.popleft(), snapshot, self._time)
        self._keys[key] = job_id
        self._queue.append(job_id)
        if job.limits.queue_seconds is not None:
            self._event(self._time + job.limits.queue_seconds, 0, job_id)
        self._start_ready()
        return job_id

    def _run(self, job_id: str) -> _Replay:
        try:
            return self._runs[job_id]
        except KeyError as error:
            raise UnknownJobError(job_id) from error

    def _start_ready(self) -> None:
        while self._queue and self._active < self._workers:
            job_id = self._queue.popleft()
            run = self._runs[job_id]
            if run.snapshot.state is not State.QUEUED:
                continue
            run.started = self._time
            run.snapshot = replace(
                run.snapshot,
                state=State.RUNNING,
                started_at=self._timestamp(self._time),
                queued_seconds=self._time - run.submitted,
            )
            self._active += 1
            self._event(
                self._time
                + min(run.recording.duration_seconds, run.job.limits.wall_seconds),
                1,
                job_id,
            )

    def _finish(
        self,
        run: _Replay,
        state: State,
        *,
        replayed: bool = False,
        error: str | None = None,
    ) -> None:
        if run.snapshot.state is State.RUNNING:
            self._active -= 1
        run.snapshot = replace(
            run.snapshot,
            state=state,
            finished_at=self._timestamp(self._time),
            elapsed_seconds=None if run.started is None else self._time - run.started,
            recorded_result=run.recording.result if replayed else None,
            error=run.recording.error if replayed else error,
        )

    def advance(self, seconds: float) -> tuple[ReplaySnapshot, ...]:
        """Advance virtual time and return newly terminal receipts in event order.

        Queue expiry wins a tie with a worker release. A recording whose duration
        equals its wall limit completes; longer recordings time out at that limit.
        Zero-duration jobs finish at advance(0), without a real-time sleep.
        """
        target = self._time + duration(seconds, "Clock advance")
        self._timestamp(target)
        finished: list[ReplaySnapshot] = []
        while self._events and self._events[0][0] <= target:
            when, kind, _, job_id = heappop(self._events)
            self._time = when
            run = self._runs[job_id]
            if kind == 0 and run.snapshot.state is State.QUEUED:
                self._finish(run, State.EXPIRED, error="Simulated queue limit exceeded")
            elif kind == 1 and run.snapshot.state is State.RUNNING:
                if run.recording.duration_seconds > run.job.limits.wall_seconds:
                    self._finish(
                        run, State.TIMED_OUT, error="Simulated wall limit exceeded"
                    )
                else:
                    self._finish(run, run.recording.state, replayed=True)
            else:
                continue
            finished.append(run.snapshot)
            self._start_ready()
        self._time = target
        return tuple(finished)

    async def poll(self, job_id: str) -> ReplaySnapshot:
        return self._run(job_id).snapshot

    async def collect(self, job_id: str) -> ReplaySnapshot:
        run = self._run(job_id)
        while run.snapshot.state in (State.QUEUED, State.RUNNING):
            if not self._events:
                raise ExecutorError("Pending replay has no clock event")
            self.advance(self._events[0][0] - self._time)
        return run.snapshot

    async def cancel(self, job_id: str) -> ReplaySnapshot:
        run = self._run(job_id)
        if run.snapshot.state in (State.QUEUED, State.RUNNING):
            self._finish(run, State.CANCELLED)
            self._start_ready()
        return run.snapshot

    async def close(self) -> None:
        self._closed = True
        # Do not start queued work while cancelling the instance.
        self._queue.clear()
        for run in self._runs.values():
            if run.snapshot.state in (State.QUEUED, State.RUNNING):
                self._finish(run, State.CANCELLED)
        self._events.clear()
