"""Pure environment-validity rules for wall-clock-budgeted study runs.

No file, process or clock access happens here: callers pass in readings.
Boundary: a run is invalid because of an environment event (machine sleep or
throttling), decided from clock drift and step count alone. A score never
decides validity, so an exclusion cannot depend on an outcome.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from openloop.ledger import ConflictError, InvalidInputError
from openloop.ledger.validation import finite_number

DRIFT_SAMPLE_SECONDS = 5.0
DRIFT_LIMIT_SECONDS = 5.0
DRIFT_QUIET_TOLERANCE_SECONDS = 1.0
SETTLE_SECONDS = 60.0
STEP_FRACTION = 0.85
MAX_ATTEMPTS_PER_SLOT = 3


def validity_policy() -> dict[str, int | float]:
    """The thresholds as recorded in the plan."""
    return {
        "drift_sample_seconds": DRIFT_SAMPLE_SECONDS,
        "drift_limit_seconds": DRIFT_LIMIT_SECONDS,
        "drift_quiet_tolerance_seconds": DRIFT_QUIET_TOLERANCE_SECONDS,
        "settle_seconds": SETTLE_SECONDS,
        "step_fraction": STEP_FRACTION,
        "max_attempts_per_slot": MAX_ATTEMPTS_PER_SLOT,
    }


@dataclass(frozen=True)
class DriftTrip:
    """Cumulative sleep drift that exceeded the limit."""

    drift_seconds: float

    @property
    def reason(self) -> str:
        return f"invalid: machine sleep (drift {self.drift_seconds:.1f} s)"


class SleepDriftMonitor:
    """Accumulate wall-clock time that the monotonic clock did not see.

    On macOS the monotonic clock stops during sleep and the wall clock does
    not, so the difference of their elapsed times is time spent asleep. Only
    positive per-sample differences accumulate, so a backward wall-clock step
    cannot hide earlier sleep.
    """

    def __init__(
        self, wall: float, monotonic: float, *, limit: float = DRIFT_LIMIT_SECONDS
    ):
        self._wall = finite_number(wall, "wall")
        self._monotonic = finite_number(monotonic, "monotonic")
        self._limit = limit
        self.total = 0.0
        self.last_increment = 0.0

    def observe(self, wall: float, monotonic: float) -> DriftTrip | None:
        """Add one sample. Trip when cumulative drift exceeds the limit."""
        self.last_increment = max(
            0.0, (wall - self._wall) - (monotonic - self._monotonic)
        )
        self._wall, self._monotonic = wall, monotonic
        self.total += self.last_increment
        return DriftTrip(self.total) if self.total > self._limit else None


def settle_wait(
    *,
    wall: Callable[[], float],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    settle_seconds: float = SETTLE_SECONDS,
    interval: float = DRIFT_SAMPLE_SECONDS,
    tolerance: float = DRIFT_QUIET_TOLERANCE_SECONDS,
) -> float:
    """Block until the machine shows no drift for settle_seconds; return the wait.

    Drift above the tolerance in any sample restarts the quiet period.
    """
    monitor = SleepDriftMonitor(wall(), monotonic(), limit=float("inf"))
    start = quiet_since = monotonic()
    while monotonic() - quiet_since < settle_seconds:
        sleep(interval)
        monitor.observe(wall(), monotonic())
        if monitor.last_increment > tolerance:
            quiet_since = monotonic()
    return monotonic() - start


def reference_steps(counts_by_arm: Mapping[str, Sequence[int]]) -> dict[str, float]:
    """Median step count per arm from earlier valid runs."""
    result: dict[str, float] = {}
    for arm, counts in counts_by_arm.items():
        if not counts or any(type(c) is not int or c <= 0 for c in counts):
            raise InvalidInputError(
                f"Reference steps for {arm} must be positive integers"
            )
        result[arm] = float(median(counts))
    return result


def step_floor(reference: float, fraction: float = STEP_FRACTION) -> float:
    """The smallest step count that is still valid."""
    return reference * fraction


def steps_invalid_reason(num_steps: int, floor: float) -> str | None:
    """Return the invalidity reason when a run trained too few steps, else None."""
    if num_steps < floor:
        return f"invalid: throttled ({num_steps} steps below floor {floor:.1f})"
    return None


def is_valid(receipt: Mapping[str, object]) -> bool:
    """A receipt counts for analysis only when it carries validity valid true."""
    validity = receipt.get("validity")
    return isinstance(validity, Mapping) and validity.get("valid") is True


def valid_receipts(
    receipts: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Drop invalid attempts. Environment-based, never outcome-based."""
    return [r for r in receipts if is_valid(r)]


class SlotExhaustedError(ConflictError):
    """Every allowed attempt of one slot was invalid. The study must stop."""


@dataclass(frozen=True)
class Attempt:
    """The validity verdict of one attempt. A reason is present when invalid."""

    valid: bool
    reason: str | None = None


async def run_slot(
    attempt_once: Callable[[int], Awaitable[Attempt]],
    settle: Callable[[], Awaitable[object]],
    *,
    max_attempts: int = MAX_ATTEMPTS_PER_SLOT,
) -> tuple[Attempt, ...]:
    """Run one slot until an attempt is valid, settling between attempts.

    Returns all attempts in order; the last one is valid. Raises
    SlotExhaustedError after max_attempts invalid attempts.
    """
    attempts: list[Attempt] = []
    for number in range(1, max_attempts + 1):
        attempt = await attempt_once(number)
        attempts.append(attempt)
        if attempt.valid:
            return tuple(attempts)
        if number < max_attempts:
            await settle()
    reasons = "; ".join(str(a.reason) for a in attempts)
    raise SlotExhaustedError(f"Slot invalid after {max_attempts} attempts: {reasons}")
