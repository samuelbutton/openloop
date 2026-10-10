"""Shared job supervision for the local and container executors.

The supervisor owns admission, idempotency, deadlines, logs, cancellation, and
receipts. A private backend supplies the parts that differ: how to prepare a
run, what command to execute, how to watch it, and how to clean up. This module
never writes ledger records or authoritative scores.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import signal
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4

from .models import (
    Artifact,
    ArtifactRole,
    ExecutorError,
    InvalidJobError,
    ProcessJob,
    Snapshot,
    State,
    SubmissionConflictError,
    UnknownJobError,
)


@dataclass
class Run:
    """One admitted job. Times are event-loop clock readings."""

    job: ProcessJob
    snapshot: Snapshot
    directory: Path
    admitted: float
    started: float | None = None
    task: asyncio.Task[None] | None = None
    observed_exit: int | None = None
    cancel_requested: bool = False
    peak_rss_bytes: int | None = None
    outputs: tuple[Artifact, ...] = ()


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    cwd: Path | None
    environment: Mapping[str, str]


class Backend(Protocol):
    """What differs between executors. The supervisor calls these in order.

    prepare: build inputs. Time spent here counts as queue time.
    command: the process to start. The wall limit starts when it starts.
    monitor: run beside the process; raise ExecutorError to stop the job.
    finish: after the process exits on its own, validate and return outputs.
    cleanup: always runs after prepare, including after failure or cancellation.
    """

    async def prepare(self, run: Run) -> None: ...
    def command(self, run: Run) -> Command: ...
    async def monitor(self, run: Run, process: asyncio.subprocess.Process) -> None: ...
    async def finish(self, run: Run) -> tuple[Artifact, ...]: ...
    async def cleanup(self, run: Run) -> None: ...


def artifact(path: Path, role: ArtifactRole = ArtifactRole.LOG) -> Artifact:
    """Hash a completed file and record its observed size."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return Artifact(path.resolve(), digest, path.stat().st_size, role)


def kill_group(process: asyncio.subprocess.Process) -> None:
    """Stop the owned POSIX group, including descendants of an exited parent."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


async def wait_for_exit(process: asyncio.subprocess.Process, seconds: float) -> None:
    """Sleep up to the given time, but return at once when the process exits."""
    with suppress(TimeoutError):
        async with asyncio.timeout(seconds):
            await process.wait()


@asynccontextmanager
async def gpu_lane(enabled: bool) -> AsyncIterator[None]:
    """Hold a user-wide file lock, shared by every executor on this host."""
    if not enabled:
        yield
        return
    lane = Path("/tmp") / f"openloop-gpu-{os.getuid()}.lock"
    descriptor = os.open(lane, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.02)
        yield


class Supervisor:
    def __init__(self, root: Path, backend: Backend) -> None:
        if os.name != "posix":
            raise ExecutorError("Executors require POSIX process groups and limits")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._backend = backend
        self._runs: dict[str, Run] = {}
        self._keys: dict[str, str] = {}
        self._closed = False

    @property
    def job_ids(self) -> frozenset[str]:
        return frozenset(self._runs)

    def admit(self, job: ProcessJob, *, key: str, input_hash: str) -> str:
        if self._closed:
            raise ExecutorError("Executor is closed")
        if not isinstance(key, str) or not key.strip():
            raise InvalidJobError("An idempotency key is required")
        if key in self._keys:
            run = self._runs[self._keys[key]]
            if run.snapshot.input_hash != input_hash:
                raise SubmissionConflictError("Key already identifies different inputs")
            return run.snapshot.job_id
        job_id = uuid4().hex
        directory = self.root / job_id
        directory.mkdir(mode=0o700)
        run = Run(
            job,
            Snapshot(job_id, input_hash, State.QUEUED, datetime.now(UTC)),
            directory,
            asyncio.get_running_loop().time(),
        )
        self._runs[job_id] = run
        self._keys[key] = job_id
        run.task = asyncio.create_task(self._drive(run))
        return job_id

    def _run(self, job_id: str) -> Run:
        try:
            return self._runs[job_id]
        except KeyError as error:
            raise UnknownJobError(job_id) from error

    async def poll(self, job_id: str) -> Snapshot:
        return self._run(job_id).snapshot

    async def collect(self, job_id: str) -> Snapshot:
        run = self._run(job_id)
        if run.task is not None:
            # Shield: cancelling a collector must not cancel the job.
            await asyncio.shield(run.task)
        return run.snapshot

    async def cancel(self, job_id: str) -> Snapshot:
        run = self._run(job_id)
        if run.task is not None and not run.task.done():
            # Cancel the task once; concurrent callers all wait for the same cleanup.
            if not run.cancel_requested:
                run.cancel_requested = True
                run.task.cancel()
            try:
                # Shield: cancelling this caller must not interrupt cleanup.
                await asyncio.shield(run.task)
            except asyncio.CancelledError:
                if not run.task.done():
                    raise  # This caller was cancelled, not the job.
                # A task cancelled before its first instruction never ran
                # _drive, so nothing published a receipt. Publish it here.
                run.snapshot = replace(
                    run.snapshot, state=State.CANCELLED, finished_at=datetime.now(UTC)
                )
        return run.snapshot

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(*(self.cancel(job_id) for job_id in self._runs))

    async def _drive(self, run: Run) -> None:
        loop = asyncio.get_running_loop()
        limits = run.job.limits
        deadline = (
            None
            if limits.queue_seconds is None
            else run.admitted + limits.queue_seconds
        )
        state = State.FAILED
        reason: str | None = None
        exit_code: int | None = None
        try:
            if deadline is not None and loop.time() >= deadline:
                raise TimeoutError
            async with asyncio.timeout_at(deadline) as queue, gpu_lane(run.job.gpu):
                try:
                    await self._backend.prepare(run)
                    queue.reschedule(None)
                    run.started = loop.time()
                    run.snapshot = replace(
                        run.snapshot,
                        state=State.RUNNING,
                        started_at=datetime.now(UTC),
                        queued_seconds=run.started - run.admitted,
                    )
                    async with asyncio.timeout(limits.wall_seconds):
                        exit_code = await self._process(run)
                        run.outputs = await self._backend.finish(run)
                finally:
                    await self._backend.cleanup(run)
            state = State.SUCCEEDED if exit_code == 0 else State.FAILED
        except TimeoutError:
            if run.started is None:
                state = State.EXPIRED
                reason = "Queue limit exceeded; the job never started"
            else:
                state = State.TIMED_OUT
                reason = "Declared wall limit exceeded"
        except asyncio.CancelledError:
            # Cancellation is a normal outcome: end the task with a CANCELLED
            # receipt so collect() returns it instead of raising.
            state = State.CANCELLED
        except (OSError, ExecutorError) as error:
            reason = str(error)
        except BaseException:
            run.snapshot = replace(
                run.snapshot,
                state=State.FAILED,
                finished_at=datetime.now(UTC),
                error="Executor failed",
            )
            raise
        logs = tuple(
            artifact(path)
            for path in (run.directory / "stdout", run.directory / "stderr")
            if path.exists()
        )
        run.snapshot = replace(
            run.snapshot,
            state=state,
            exit_code=exit_code if exit_code is not None else run.observed_exit,
            finished_at=datetime.now(UTC),
            elapsed_seconds=None if run.started is None else loop.time() - run.started,
            peak_rss_bytes=run.peak_rss_bytes,
            artifacts=logs + run.outputs,
            error=reason,
        )

    async def _process(self, run: Run) -> int:
        command = self._backend.command(run)
        process: asyncio.subprocess.Process | None = None
        with (
            (run.directory / "stdout").open("xb") as stdout,
            (run.directory / "stderr").open("xb") as stderr,
        ):
            try:
                process = await asyncio.create_subprocess_exec(
                    *command.argv,
                    cwd=command.cwd,
                    env=dict(command.environment),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                assert process.stdin is not None
                assert process.stdout is not None
                assert process.stderr is not None

                async def send() -> None:
                    assert process is not None
                    assert process.stdin is not None
                    try:
                        for offset in range(0, len(run.job.stdin), 65536):
                            process.stdin.write(run.job.stdin[offset : offset + 65536])
                            await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        process.stdin.close()

                async def drain(reader: asyncio.StreamReader, stream: BinaryIO) -> None:
                    size = 0
                    while data := await reader.read(65536):
                        remaining = run.job.limits.output_bytes - size
                        stream.write(data[:remaining])
                        stream.flush()
                        size += len(data)
                        if size > run.job.limits.output_bytes:
                            raise ExecutorError("Declared output limit exceeded")

                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(send())
                    tasks.create_task(drain(process.stdout, stdout))
                    tasks.create_task(drain(process.stderr, stderr))
                    tasks.create_task(process.wait())
                    tasks.create_task(self._backend.monitor(run, process))
                assert process.returncode is not None
                return process.returncode
            except ExceptionGroup as errors:
                # TaskGroup preserves simultaneous stream errors. Preserve a
                # domain failure; propagate unexpected failures to the caller.
                if all(
                    isinstance(error, (ExecutorError, OSError))
                    for error in errors.exceptions
                ):
                    raise ExecutorError(str(errors.exceptions[0])) from errors
                raise
            finally:
                if process is not None:
                    # Kill descendants even when their direct parent has exited.
                    kill_group(process)
                    await process.wait()
                    run.observed_exit = process.returncode
