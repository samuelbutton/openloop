"""Pure rules for Amendment 4: resume a study from earlier valid receipts.

No file, process or clock access happens here: callers pass in parsed receipts
and readings. Boundary: a retained receipt must already satisfy the validity
rules that were fixed before it ran (Amendment 3). Retention never looks at a
score, and any mismatch between a receipt and the frozen slot refuses the
resume. The module also holds the Amendment 4 memory-guard floor, so the
Amendment 2 and 3 modules stay unchanged and their plans still reproduce.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openloop.ledger import ConflictError
from openloop.studies.claims_audit import (
    FREE_PERCENT_CONSECUTIVE_SAMPLES,
    MEMORY_SAMPLE_SECONDS,
    PRESSURE_CRITICAL,
    GuardTrip,
    MemoryReading,
)
from openloop.studies.heldout import HELDOUT_KEY
from openloop.studies.validity import is_valid

# Amendment 4 lowers the free-memory floor from 15 to 5 percent. On the study
# machine, warning pressure coincides with 14 to 15 percent free, so the old
# floor contradicted the rule that warning alone must not stop the study.
# Critical pressure is the main stop; the floor is only a last backstop.
FREE_PERCENT_FLOOR = 5


class ResumeMismatchError(ConflictError):
    """A retained receipt, or the prior run, does not match the frozen plan."""


def memory_guard_policy() -> dict[str, int | float]:
    """The Amendment 4 thresholds as recorded in the plan."""
    return {
        "sample_seconds": MEMORY_SAMPLE_SECONDS,
        "critical_pressure_level": PRESSURE_CRITICAL,
        "free_percent_floor": FREE_PERCENT_FLOOR,
        "free_percent_consecutive_samples": FREE_PERCENT_CONSECUTIVE_SAMPLES,
    }


class FloorMemoryGuard:
    """Trip on critical pressure, or free memory below the floor repeatedly.

    Stateful across one job's samples. Warning pressure, or a single low free
    reading, never trips. Critical pressure trips immediately.
    """

    def __init__(self, floor: int = FREE_PERCENT_FLOOR) -> None:
        self._floor = floor
        self._low_run = 0
        self._recent: list[MemoryReading] = []

    def observe(self, reading: MemoryReading) -> GuardTrip | None:
        self._recent = [*self._recent, reading][-FREE_PERCENT_CONSECUTIVE_SAMPLES - 1 :]
        if reading.free_percent < self._floor:
            self._low_run += 1
        else:
            self._low_run = 0
        if reading.pressure_level >= PRESSURE_CRITICAL:
            return GuardTrip("critical memory pressure", tuple(self._recent))
        if self._low_run >= FREE_PERCENT_CONSECUTIVE_SAMPLES:
            return GuardTrip(
                f"system free memory below {self._floor}% for "
                f"{FREE_PERCENT_CONSECUTIVE_SAMPLES} consecutive samples",
                tuple(self._recent),
            )
        return None


@dataclass(frozen=True)
class Retained:
    """One receipt kept from the prior run, with the file it came from."""

    file: str
    receipt: Mapping[str, object]

    @property
    def slot(self) -> int:
        index = self.receipt["index"]
        assert isinstance(index, int)
        return index


def _slot_for(
    order: Sequence[Mapping[str, object]], index: object
) -> Mapping[str, object]:
    for item in order:
        if item["index"] == index:
            return item
    raise ResumeMismatchError(f"Receipt names slot {index!r}, which is not in the plan")


def select_retained(
    receipts: Mapping[str, Mapping[str, object]],
    order: Sequence[Mapping[str, object]],
    step_floors: Mapping[str, float],
) -> list[Retained]:
    """Keep receipts that are valid and successful; refuse any that mismatch.

    Invalid attempts, failed attempts and attempts without a summary are not
    retained and are not an error. A kept receipt must match its slot's kind,
    arm, seed and both program hashes, must clear the arm's step floor, and
    must be the only kept receipt for its slot.
    """
    kept: dict[int, Retained] = {}
    for name in sorted(receipts):
        receipt = receipts[name]
        if (
            not is_valid(receipt)
            or not isinstance(receipt.get("summary"), Mapping)
            or "failure" in receipt
        ):
            continue
        slot = _slot_for(order, receipt.get("index"))
        for field in ("kind", "arm", "seed", "base_program_sha256", "program_sha256"):
            if receipt.get(field) != slot[field]:
                raise ResumeMismatchError(
                    f"{name}: {field} {receipt.get(field)!r} differs from the "
                    f"frozen slot {slot['index']} ({slot[field]!r})"
                )
        summary = receipt["summary"]
        assert isinstance(summary, Mapping)
        steps = summary.get("num_steps")
        if type(steps) is not int or steps < step_floors[str(slot["arm"])]:
            raise ResumeMismatchError(f"{name}: step count fails the validity floor")
        if slot["kind"] == "heldout" and HELDOUT_KEY not in summary:
            raise ResumeMismatchError(f"{name}: held-out score is missing")
        index = slot["index"]
        assert isinstance(index, int)
        if index in kept:
            raise ResumeMismatchError(
                f"Slot {index} has two valid receipts: {kept[index].file} and {name}"
            )
        kept[index] = Retained(name, receipt)
    return [kept[i] for i in sorted(kept)]


def missing_slots(
    order: Sequence[Mapping[str, object]], retained: Sequence[Retained]
) -> list[Mapping[str, object]]:
    """The slots still to run, in the frozen order. A retained slot is never listed."""
    done = {r.slot for r in retained}
    return [item for item in order if item["index"] not in done]
