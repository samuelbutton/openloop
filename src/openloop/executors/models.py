"""Immutable executor messages. This module validates values and performs no I/O."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

from openloop.ledger import content_hash


class ExecutorError(Exception):
    """Base class for execution failures."""


class InvalidJobError(ExecutorError, ValueError):
    """A job violates the executor contract."""


class SubmissionConflictError(ExecutorError):
    """An idempotency key was used for different inputs."""


class UnknownJobError(ExecutorError, LookupError):
    """The executor does not own the requested job."""


class State(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    # The job never started: its queue limit passed while it waited.
    EXPIRED = "expired"


class ArtifactRole(StrEnum):
    LOG = "log"
    OUTPUT = "output"


def require_positive(value: object, name: str) -> None:
    """Reject nonnumeric, nonpositive, and nonfinite limits."""
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)  # type: ignore[arg-type]
        or value <= 0  # type: ignore[operator]
    ):
        raise InvalidJobError(f"{name} must be positive and finite")


_SECONDS = ("wall_seconds", "queue_seconds", "cpu_seconds", "cpus")
_INTEGERS = (
    "memory_bytes",
    "processes",
    "output_bytes",
    "file_bytes",
    "outputs_bytes",
    "scratch_bytes",
)
_OPTIONAL = ("queue_seconds", "cpu_seconds", "file_bytes")


@dataclass(frozen=True, kw_only=True)
class Limits:
    """Time limits are in seconds and size limits are in bytes.

    queue_seconds counts from submission to process start.
    wall_seconds counts from process start. None disables an optional limit.
    """

    wall_seconds: float = 60
    queue_seconds: float | None = None
    cpu_seconds: float | None = None
    memory_bytes: int = 1_073_741_824
    cpus: float = 1
    processes: int = 64
    output_bytes: int = 1_048_576
    file_bytes: int | None = None
    outputs_bytes: int = 67_108_864
    scratch_bytes: int = 67_108_864

    def __post_init__(self) -> None:
        for name in _SECONDS + _INTEGERS:
            value = getattr(self, name)
            if value is None and name in _OPTIONAL:
                continue
            require_positive(value, name)
            if name in _INTEGERS and type(value) is not int:
                raise InvalidJobError(f"{name} must be an integer")


@dataclass(frozen=True)
class ProcessJob:
    """Explicit command and byte inputs. Local execution requires reviewed=True.

    Only the local executor uses cwd and environment. Neither executor inherits
    the coordinator's environment. Files are sandbox inputs, mounted read-only.
    """

    argv: tuple[str, ...]
    _: KW_ONLY
    limits: Limits = field(default_factory=Limits)
    stdin: bytes = b""
    cwd: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    files: Mapping[str, bytes] = field(default_factory=dict)
    reviewed: bool = False
    gpu: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.argv, str)
            or not self.argv
            or any(not isinstance(arg, str) or "\0" in arg for arg in self.argv)
        ):
            raise InvalidJobError(
                "argv must contain nonempty, NUL-free command arguments"
            )
        if not self.argv[0]:
            raise InvalidJobError("The executable must not be empty")
        if not isinstance(self.stdin, bytes) or not isinstance(self.limits, Limits):
            raise InvalidJobError("A job needs byte stdin and validated limits")
        if type(self.reviewed) is not bool or type(self.gpu) is not bool:
            raise InvalidJobError("reviewed and gpu must be booleans")
        if self.cwd is not None and not self.cwd.is_absolute():
            raise InvalidJobError("cwd must be absolute")
        for key, value in self.environment.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key)
                or "\0" in value
            ):
                raise InvalidJobError("Invalid environment entry")
        for name, data in self.files.items():
            if not isinstance(name, str) or "\0" in name:
                raise InvalidJobError("Input paths must be NUL-free strings")
            path = PurePosixPath(name)
            if (
                not name
                or not path.parts
                or path.is_absolute()
                or ".." in path.parts
                or str(path) != name
                or not isinstance(data, bytes)
            ):
                raise InvalidJobError(
                    "Input files need normalized relative paths and bytes"
                )
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        names = set(self.files)
        if any(
            str(parent) in names
            for name in names
            for parent in PurePosixPath(name).parents
        ):
            raise InvalidJobError("An input file cannot also be an input directory")

    @property
    def hash(self) -> str:
        return content_hash(
            {
                "argv": self.argv,
                "limits": self.limits,
                "stdin": hashlib.sha256(self.stdin).hexdigest(),
                "cwd": str(self.cwd) if self.cwd is not None else None,
                "environment": dict(self.environment),
                "files": {
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in self.files.items()
                },
                "reviewed": self.reviewed,
                "gpu": self.gpu,
            }
        )


@dataclass(frozen=True)
class Artifact:
    """A coordinator-owned file, identified by its SHA-256 digest.

    Logs are `stdout` and `stderr` in the job directory. Outputs are files that
    a sandboxed candidate wrote to /outputs; they are untrusted bytes.
    """

    path: Path
    sha256: str
    size_bytes: int
    role: ArtifactRole = ArtifactRole.LOG

    def __post_init__(self) -> None:
        if not self.path.is_absolute() or not re.fullmatch(
            r"[0-9a-f]{64}", self.sha256
        ):
            raise InvalidJobError("Artifacts need an absolute path and SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise InvalidJobError("Artifact size must be a nonnegative integer")
        try:
            object.__setattr__(self, "role", ArtifactRole(self.role))
        except ValueError as error:
            raise InvalidJobError("Unknown artifact role") from error


@dataclass(frozen=True)
class Snapshot:
    """Observed lifecycle values. Anything not observed remains absent.

    queued_seconds runs from submission to process start. elapsed_seconds runs
    from process start to finish. A job that never started has neither.
    """

    job_id: str
    input_hash: str
    state: State
    submitted_at: datetime
    _: KW_ONLY
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    queued_seconds: float | None = None
    elapsed_seconds: float | None = None
    peak_rss_bytes: int | None = None
    artifacts: tuple[Artifact, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.job_id or not re.fullmatch(r"[0-9a-f]{64}", self.input_hash):
            raise InvalidJobError("Snapshots need a job ID and input SHA-256")
        try:
            object.__setattr__(self, "state", State(self.state))
        except ValueError as error:
            raise InvalidJobError("Unknown execution state") from error
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        times = [
            time
            for time in (self.submitted_at, self.started_at, self.finished_at)
            if time is not None
        ]
        if any(time.utcoffset() is None for time in times) or times != sorted(times):
            raise InvalidJobError("Lifecycle times must be aware and ordered")
        for seconds in (self.queued_seconds, self.elapsed_seconds):
            if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
                raise InvalidJobError("Durations must be finite and nonnegative")
        if self.peak_rss_bytes is not None and (
            type(self.peak_rss_bytes) is not int or self.peak_rss_bytes < 0
        ):
            raise InvalidJobError("Observed RAM must be a nonnegative integer")


class Executor(Protocol):
    """In-process job ownership. collect waits; poll returns an immutable snapshot."""

    async def submit(self, job: ProcessJob, *, key: str) -> str: ...
    async def poll(self, job_id: str) -> Snapshot: ...
    async def cancel(self, job_id: str) -> Snapshot: ...
    async def collect(self, job_id: str) -> Snapshot: ...
    async def close(self) -> None: ...
