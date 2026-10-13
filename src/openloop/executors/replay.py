"""Immutable recordings and simulated receipts. No clocks, files, or execution.

The caller supplies trusted recorded outcomes. Replaying them never creates
new empirical evidence or authoritative evaluations of candidate code.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from openloop.ledger import EventKind, Result, Run, Stage, Status, content_hash

from .models import InvalidJobError, Snapshot, State


def duration(value: object, name: str) -> float:
    """Require an explicit finite nonnegative duration, including zero."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise InvalidJobError(f"{name} must be finite and nonnegative")
    return float(value)


def _digest(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise InvalidJobError("Recordings require lowercase SHA-256 references")


@dataclass(frozen=True)
class RecordedRun:
    """Bind one source recording to an exact ProcessJob hash for replay.

    duration_seconds is recorded execution time, not queue time. result retains
    original metrics and artifact references, without recreating artifact files.
    Missing duration or outcome must not be replaced with a fabricated default.

    pool_key declares which recordings are interchangeable draws of one
    candidate, for example candidate plus fidelity but not seed. It defaults to
    job_hash, which pools only recordings of identical inputs.
    """

    job_hash: str
    source_hash: str
    duration_seconds: float
    state: State
    result: Result | None = field(default=None, kw_only=True)
    error: str | None = field(default=None, kw_only=True)
    pool_key: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _digest(self.job_hash)
        _digest(self.source_hash)
        object.__setattr__(
            self,
            "duration_seconds",
            duration(self.duration_seconds, "Recorded duration"),
        )
        try:
            state = State(self.state)
        except ValueError as error:
            raise InvalidJobError("Unknown recorded state") from error
        if state in (State.QUEUED, State.RUNNING, State.EXPIRED):
            raise InvalidJobError("Replay requires a terminal outcome of started work")
        if self.result is not None and not isinstance(self.result, Result):
            raise InvalidJobError("Recorded results must be typed")
        if state is not State.SUCCEEDED and self.result is not None:
            raise InvalidJobError("Unsuccessful work cannot supply a result")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise InvalidJobError("A recorded error must be nonempty text")
        if self.pool_key is not None and (
            not isinstance(self.pool_key, str) or not self.pool_key.strip()
        ):
            raise InvalidJobError("A pool key must be nonempty text")
        object.__setattr__(self, "state", state)

    @property
    def pool(self) -> str:
        """The declared pool key, or the exact job hash when none is declared."""
        return self.job_hash if self.pool_key is None else self.pool_key

    @property
    def hash(self) -> str:
        return content_hash({"replay_version": 2, "recording": self})

    @classmethod
    def from_run(
        cls,
        run: Run,
        *,
        job_hash: str,
        duration_metric: str | Stage,
        pool_key: str | None = None,
    ) -> RecordedRun:
        """Capture a completed ledger attempt; reject cache aliases and unknown time.

        duration_metric is required because the choices differ. A string names
        a seconds observation with the execution split, such as T1's
        process_seconds, which times the training process alone.
        Stage.EXECUTION selects the execution-stage duration, which can include
        orchestration overhead around the process. Neither is a default.
        Job binding is a caller assertion; this method cannot reconstruct argv.
        """
        if not isinstance(run, Run) or run.reused_from is not None:
            raise InvalidJobError(
                "Replay needs an original ledger attempt, not a cache alias"
            )
        if run.status not in (Status.SUCCEEDED, Status.FAILED):
            raise InvalidJobError("Replay needs a terminal ledger attempt")
        if run.status is Status.SUCCEEDED and run.result is None:
            raise InvalidJobError("A successful attempt requires a recorded result")
        if not isinstance(duration_metric, Stage):
            metric = (
                run.result.observations.get(duration_metric) if run.result else None
            )
            if (
                metric is None
                or metric.unit != "seconds"
                or metric.split != "execution"
            ):
                raise InvalidJobError("Missing recorded execution duration in seconds")
            seconds = metric.value
        else:
            events = tuple(
                event
                for event in run.events
                if event.stage is Stage.EXECUTION
                and event.kind is EventKind.STAGE_FINISHED
            )
            if len(events) != 1:
                raise InvalidJobError("Missing recorded execution-stage duration")
            seconds = duration(
                events[0].payload.get("duration_seconds"), "Execution duration"
            )
        reason = run.events[-1].payload.get("reason") if run.events else None
        return cls(
            job_hash,
            content_hash(run),
            seconds,
            State.SUCCEEDED if run.status is Status.SUCCEEDED else State.FAILED,
            result=run.result if run.status is Status.SUCCEEDED else None,
            error=reason if isinstance(reason, str) else None,
            pool_key=pool_key,
        )


@dataclass(frozen=True, kw_only=True)
class ReplaySnapshot(Snapshot):
    """Simulated lifecycle times plus a reference to the source recording.

    recorded_result appears only when the recorded outcome completes unchanged.
    It is an old observation, never a new measurement. No logs are fabricated.

    pool_key is the pool the recording was drawn from. reuse_count is how many
    earlier submissions were served this same recording, and resampled is
    reuse_count > 0: the result repeats recorded noise instead of adding a new
    sample. A strict replay always reports reuse_count 0.
    """

    recording_hash: str
    recorded_result: Result | None = None
    pool_key: str | None = None
    reuse_count: int = 0
    resampled: bool = False
    simulated: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _digest(self.recording_hash)
        if (
            type(self.reuse_count) is not int
            or self.reuse_count < 0
            or self.resampled != (self.reuse_count > 0)
        ):
            raise InvalidJobError("resampled must match a positive reuse count")
        if self.recorded_result is not None and (
            not isinstance(self.recorded_result, Result)
            or self.state is not State.SUCCEEDED
        ):
            raise InvalidJobError(
                "Only successful replays can expose a recorded result"
            )
