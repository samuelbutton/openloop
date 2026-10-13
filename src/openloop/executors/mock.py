"""Finite virtual-time replay. Never execute commands or access their inputs.

Ownership, queue order, lane and worker capacity, recording draws, and clock
state are in memory. No ledger, artifact, network, subprocess, or real-time
waits are involved. Each receipt is simulated.
"""

import hashlib
import random
from collections import deque
from collections.abc import Callable, Iterable
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

_PENDING = (State.QUEUED, State.RUNNING)


@dataclass
class _Replay:
    job: ProcessJob
    recording: RecordedRun
    snapshot: ReplaySnapshot
    submitted: float
    order: int
    started: float | None = None


class MockExecutor:
    """Replay recordings on identical simulated workers and GPU lanes.

    Time. poll never advances time. advance(seconds) processes all events in
    that interval. collect and wait_any advance the shared clock until the
    awaited job (or the first of several) is terminal, so other jobs can
    complete too. Waiting with sleep-and-poll hangs here; use collect/wait_any.

    Capacity. A job needs a free worker. A job with gpu=True also needs a free
    GPU lane (gpu_lanes of them); a waiting GPU job holds no worker, so CPU
    jobs behind it still start. Other order is FIFO. The real Mac has one
    lane; model more GPUs only by setting gpu_lanes deliberately.

    Draws. Recordings are pooled by RecordedRun.pool (declared pool_key, else
    the exact job hash). A submission maps to a pool through pool_of, or by
    its job hash when pool_of is None. By default each new submission key draws
    one recording from its pool with replacement, using an RNG seeded from
    (seed, pool, occurrence index within that pool), so a draw does not depend
    on how submissions of other pools interleave. Resampling reuses recorded
    noise; it cannot create variance beyond the recorded pool, and a reused
    recording is a repeat of an old observation, not a new sample. Receipts
    carry the recording hash, reuse_count, and resampled. With strict=True a
    pool is consumed once, in input order, and exhaustion raises.
    Repeated keys are idempotent and draw nothing.
    """

    def __init__(
        self,
        recordings: tuple[RecordedRun, ...],
        *,
        workers: int = 1,
        gpu_lanes: int = 1,
        seed: int | None = None,
        strict: bool = False,
        pool_of: Callable[[ProcessJob], str] | None = None,
        epoch: datetime = datetime(2000, 1, 1, tzinfo=UTC),
    ) -> None:
        if type(workers) is not int or workers < 1:
            raise InvalidJobError("workers must be a positive integer")
        if type(gpu_lanes) is not int or gpu_lanes < 1:
            raise InvalidJobError("gpu_lanes must be a positive integer")
        if type(strict) is not bool:
            raise InvalidJobError("strict must be a boolean")
        if not strict and (type(seed) is not int):
            raise InvalidJobError("Resampling replay requires an integer seed")
        if not isinstance(epoch, datetime) or epoch.utcoffset() is None:
            raise InvalidJobError("Simulation epoch must be an aware datetime")
        self._epoch = epoch.astimezone(UTC)
        self._workers = workers
        self._gpu_lanes = gpu_lanes
        self._seed = seed
        self._strict = strict
        self._pool_of = pool_of
        self._time = 0.0
        self._pools: dict[str, list[RecordedRun]] = {}
        self._pool_by_job: dict[str, str] = {}
        for recording in recordings:
            if not isinstance(recording, RecordedRun):
                raise InvalidJobError("Replay requires typed recordings")
            self._timestamp(recording.duration_seconds)
            if self._pool_by_job.setdefault(recording.job_hash, recording.pool) != (
                recording.pool
            ):
                raise InvalidJobError("One job hash cannot belong to two pools")
            self._pools.setdefault(recording.pool, []).append(recording)
        self._draws: dict[str, int] = {}
        self._served: dict[str, list[int]] = {
            pool: [0] * len(items) for pool, items in self._pools.items()
        }
        self._runs: dict[str, _Replay] = {}
        self._keys: dict[str, str] = {}
        self._cpu_queue: deque[str] = deque()
        self._gpu_queue: deque[str] = deque()
        self._events: list[tuple[float, int, int, str]] = []
        self._event_sequence = 0
        self._active = 0
        self._active_gpu = 0
        self._closed = False

    @property
    def now(self) -> float:
        """Simulated seconds since epoch, not host elapsed time."""
        return self._time

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def gpu_lanes(self) -> int:
        return self._gpu_lanes

    @property
    def resampled_submissions(self) -> int:
        """Submissions served a recording that an earlier submission already used."""
        return sum(run.snapshot.resampled for run in self._runs.values())

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
        pool = self._pool_for(job)
        index = self._choose(pool)
        recording = self._pools[pool][index]
        self._timestamp(
            self._time + min(recording.duration_seconds, job.limits.wall_seconds)
        )
        if job.limits.queue_seconds is not None:
            self._timestamp(self._time + job.limits.queue_seconds)
        self._draws[pool] = self._draws.get(pool, 0) + 1
        reuse_count = self._served[pool][index]
        self._served[pool][index] += 1
        order = len(self._runs) + 1
        job_id = f"mock-{order}"
        snapshot = ReplaySnapshot(
            job_id,
            digest,
            State.QUEUED,
            self._timestamp(self._time),
            recording_hash=recording.hash,
            pool_key=pool,
            reuse_count=reuse_count,
            resampled=reuse_count > 0,
        )
        self._runs[job_id] = _Replay(job, recording, snapshot, self._time, order)
        self._keys[key] = job_id
        (self._gpu_queue if job.gpu else self._cpu_queue).append(job_id)
        if job.limits.queue_seconds is not None:
            self._event(self._time + job.limits.queue_seconds, 0, job_id)
        self._start_ready()
        return job_id

    def _pool_for(self, job: ProcessJob) -> str:
        if self._pool_of is None:
            pool = self._pool_by_job.get(job.hash)
        else:
            pool = self._pool_of(job)
            if not isinstance(pool, str) or not pool.strip():
                raise InvalidJobError("pool_of must return nonempty text")
        if pool is None or pool not in self._pools:
            raise InvalidJobError("No recording pool for these exact inputs")
        return pool

    def _choose(self, pool: str) -> int:
        """Pick the next recording index without changing any state."""
        occurrence = self._draws.get(pool, 0)
        size = len(self._pools[pool])
        if self._strict:
            if occurrence >= size:
                raise InvalidJobError("No remaining recording for these exact inputs")
            return occurrence
        material = f"{self._seed}\0{pool}\0{occurrence}".encode()
        generator = random.Random(int.from_bytes(hashlib.sha256(material).digest()))
        return generator.randrange(size)

    def _run(self, job_id: str) -> _Replay:
        try:
            return self._runs[job_id]
        except KeyError as error:
            raise UnknownJobError(job_id) from error

    def _next_startable(self) -> str | None:
        """Pop the earliest queued job that has the resources it needs."""
        for queue in (self._cpu_queue, self._gpu_queue):
            while queue and self._runs[queue[0]].snapshot.state is not State.QUEUED:
                queue.popleft()
        cpu = self._cpu_queue[0] if self._cpu_queue else None
        gpu = (
            self._gpu_queue[0]
            if self._gpu_queue and self._active_gpu < self._gpu_lanes
            else None
        )
        if cpu is None and gpu is None:
            return None
        if gpu is None or (
            cpu is not None and self._runs[cpu].order < self._runs[gpu].order
        ):
            return self._cpu_queue.popleft()
        return self._gpu_queue.popleft()

    def _start_ready(self) -> None:
        while self._active < self._workers:
            job_id = self._next_startable()
            if job_id is None:
                return
            run = self._runs[job_id]
            run.started = self._time
            run.snapshot = replace(
                run.snapshot,
                state=State.RUNNING,
                started_at=self._timestamp(self._time),
                queued_seconds=self._time - run.submitted,
            )
            self._active += 1
            self._active_gpu += run.job.gpu
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
            self._active_gpu -= run.job.gpu
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
        return self._wait((self._run(job_id),))

    async def wait_any(self, job_ids: Iterable[str]) -> ReplaySnapshot:
        """Advance virtual time to the next terminal job among job_ids.

        Returns at once if one is terminal; ties go to the smallest job ID.
        """
        ids = sorted(set(job_ids))
        if not ids:
            raise InvalidJobError("wait_any requires at least one job ID")
        return self._wait(tuple(self._run(job_id) for job_id in ids))

    def _wait(self, runs: tuple[_Replay, ...]) -> ReplaySnapshot:
        while True:
            for run in runs:
                if run.snapshot.state not in _PENDING:
                    return run.snapshot
            if not self._events:
                raise ExecutorError("Pending replay has no clock event")
            self.advance(self._events[0][0] - self._time)

    async def cancel(self, job_id: str) -> ReplaySnapshot:
        run = self._run(job_id)
        if run.snapshot.state in _PENDING:
            self._finish(run, State.CANCELLED)
            self._start_ready()
        return run.snapshot

    async def close(self) -> None:
        self._closed = True
        # Do not start queued work while cancelling the instance.
        self._cpu_queue.clear()
        self._gpu_queue.clear()
        for run in self._runs.values():
            if run.snapshot.state in _PENDING:
                self._finish(run, State.CANCELLED)
        self._events.clear()
