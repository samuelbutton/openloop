"""Pure value types for runs, measurements, events, and comparisons."""

# Runtime isinstance checks validate untyped callers at the trust boundary.
# pyright: reportUnnecessaryIsInstance=false

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .errors import InvalidInputError
from .identity import ExperimentInputs
from .validation import (
    JSONValue,
    check_digest,
    check_positive,
    check_text,
    coerce_enum,
)


class Purpose(StrEnum):
    RUN = "run"
    RETRY = "retry"
    REPLICATION = "replication"


class Status(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Stage(StrEnum):
    """Attempt stages, in pipeline order."""

    QUEUE = "queue"
    EXECUTION = "execution"
    EVALUATION = "evaluation"


class PreparationStage(StrEnum):
    """Work done before submission, in pipeline order."""

    PROPOSAL = "proposal"
    IMPLEMENTATION = "implementation"


class Direction(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class EventKind(StrEnum):
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class Metric:
    value: float
    unit: str
    direction: Direction = Direction.MINIMIZE
    split: str = "validation"
    sample_count: int = 1

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise InvalidInputError("Metric values must be finite numbers")
        object.__setattr__(self, "value", float(self.value))
        check_text(self.unit)
        check_text(self.split)
        check_positive(self.sample_count)
        object.__setattr__(
            self, "direction", coerce_enum(Direction, self.direction, "direction")
        )


@dataclass(frozen=True)
class Result:
    """Trusted measurements. Decisions over several results are recorded elsewhere.

    Metrics are the result being judged. Observations are operational
    measurements, such as timing and throughput, that vary between identical
    runs; comparisons and decisions ignore them.
    """

    metrics: Mapping[str, Metric]
    artifacts: Mapping[str, str] = field(default_factory=dict[str, str])
    observations: Mapping[str, Metric] = field(default_factory=dict[str, Metric])

    def __post_init__(self) -> None:
        if not (
            isinstance(self.metrics, Mapping)
            and isinstance(self.artifacts, Mapping)
            and isinstance(self.observations, Mapping)
        ):
            raise InvalidInputError(
                "Metrics, artifacts, and observations must be mappings"
            )
        if not self.metrics:
            raise InvalidInputError("A completed result requires at least one metric")
        for name, metric in self.metrics.items():
            check_text(name)
            if not isinstance(metric, Metric):
                raise InvalidInputError("Expected typed metrics")
        for name, metric in self.observations.items():
            check_text(name)
            if not isinstance(metric, Metric):
                raise InvalidInputError("Expected typed observations")
        for role, digest in self.artifacts.items():
            check_text(role)
            check_digest(digest)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        object.__setattr__(
            self, "observations", MappingProxyType(dict(self.observations))
        )


@dataclass(frozen=True)
class Event:
    sequence: int
    kind: EventKind
    timestamp: str
    stage: Stage | None
    payload: Mapping[str, JSONValue]


@dataclass(frozen=True)
class Preparation:
    """Actual UTC times of proposal or implementation work before submission."""

    stage: PreparationStage
    started_at: str
    finished_at: str
    duration_seconds: float


@dataclass(frozen=True)
class Run:
    """One submission; reused submissions reference the same execution attempt."""

    id: str
    attempt_id: str
    inputs: ExperimentInputs
    hypothesis: str
    parents: tuple[str, ...]
    purpose: Purpose
    retry_of: str | None
    created_at: str
    reused_from: str | None
    status: Status
    preparations: tuple[Preparation, ...]
    events: tuple[Event, ...]
    result: Result | None

    @property
    def stages(self) -> Mapping[Stage, tuple[Event, ...]]:
        return MappingProxyType(
            {
                stage: tuple(event for event in self.events if event.stage is stage)
                for stage in Stage
            }
        )


@dataclass(frozen=True)
class ProbeResult:
    id: str
    baseline_id: str
    repeat_id: str
    matches: bool
    differences: tuple[str, ...]
    atol: float
    rtol: float
    invalidates_cache: bool


def validate_tolerances(atol: float, rtol: float) -> None:
    for tolerance in (atol, rtol):
        if (
            type(tolerance) not in (int, float)
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise InvalidInputError("Tolerances must be finite nonnegative numbers")


def compare_results(
    baseline: Result,
    repeat: Result,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> tuple[str, ...]:
    """Compare metrics and artifact hashes, ignoring observations such as timing."""
    validate_tolerances(atol, rtol)
    differences: list[str] = []
    for name in sorted(baseline.metrics.keys() | repeat.metrics.keys()):
        left, right = baseline.metrics.get(name), repeat.metrics.get(name)
        if left is None or right is None:
            differences.append(f"metric {name}: missing or extra")
        elif (left.unit, left.direction, left.split, left.sample_count) != (
            right.unit,
            right.direction,
            right.split,
            right.sample_count,
        ):
            differences.append(f"metric {name}: metadata differs")
        elif abs(left.value - right.value) > atol + rtol * abs(left.value):
            differences.append(f"metric {name}: {left.value!r} -> {right.value!r}")
    differences.extend(
        f"artifact {role}: hash differs"
        for role in sorted(baseline.artifacts.keys() | repeat.artifacts.keys())
        if baseline.artifacts.get(role) != repeat.artifacts.get(role)
    )
    return tuple(differences)
