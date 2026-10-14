"""Amendment 4 rules: memory floor and receipt retention; pure, no GPU."""

import pytest

from openloop.studies.claims_audit import MemoryReading
from openloop.studies.resume import (
    FREE_PERCENT_FLOOR,
    FloorMemoryGuard,
    ResumeMismatchError,
    memory_guard_policy,
    missing_slots,
    select_retained,
)

ORDER = [
    {
        "index": i,
        "kind": "heldout",
        "arm": "a",
        "seed": 3000 + i,
        "base_program_sha256": f"b{i}",
        "program_sha256": f"p{i}",
    }
    for i in range(1, 5)
]
FLOORS = {"a": 700.0}


def receipt(index: int, **changes: object) -> dict[str, object]:
    slot = ORDER[index - 1]
    value: dict[str, object] = {
        "index": index,
        "kind": "heldout",
        "arm": "a",
        "seed": slot["seed"],
        "base_program_sha256": slot["base_program_sha256"],
        "program_sha256": slot["program_sha256"],
        "summary": {"num_steps": 800, "val_bpb": 1.4, "heldout_bpb": 1.4},
        "validity": {"valid": True},
    }
    return {**value, **changes}


def reading(free: int, level: int = 2) -> MemoryReading:
    return MemoryReading("t", level, free)


def test_floor_is_five_percent() -> None:
    assert FREE_PERCENT_FLOOR == 5
    assert memory_guard_policy()["free_percent_floor"] == 5


def test_floor_does_not_trip_at_the_old_readings() -> None:
    guard = FloorMemoryGuard()
    for free in (15, 14, 14, 9, 9, 6):
        assert guard.observe(reading(free)) is None


def test_floor_trips_on_two_consecutive_low_samples() -> None:
    guard = FloorMemoryGuard()
    assert guard.observe(reading(4)) is None
    trip = guard.observe(reading(4))
    assert trip is not None
    assert "below 5%" in trip.reason


def test_single_low_sample_or_interrupted_run_does_not_trip() -> None:
    guard = FloorMemoryGuard()
    for free in (4, 30, 4, 30, 4):
        assert guard.observe(reading(free)) is None


def test_exactly_at_the_floor_is_not_low() -> None:
    guard = FloorMemoryGuard()
    assert guard.observe(reading(5)) is None
    assert guard.observe(reading(5)) is None


def test_critical_pressure_trips_immediately() -> None:
    trip = FloorMemoryGuard().observe(reading(80, level=4))
    assert trip is not None
    assert trip.reason == "critical memory pressure"


def test_only_valid_successful_receipts_are_retained() -> None:
    receipts = {
        "r1": receipt(1),
        "r2": receipt(2, validity={"valid": False, "reason": "x"}),
        "r3": receipt(3, summary=None, failure="memory guard: x"),
    }
    kept = select_retained(receipts, ORDER, FLOORS)
    assert [k.slot for k in kept] == [1]
    assert [i["index"] for i in missing_slots(ORDER, kept)] == [2, 3, 4]


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 1},
        {"arm": "z"},
        {"kind": "seed_completion"},
        {"program_sha256": "other"},
        {"base_program_sha256": "other"},
        {"index": 99},
    ],
)
def test_mismatched_receipt_refuses(change: dict[str, object]) -> None:
    with pytest.raises(ResumeMismatchError):
        select_retained({"r": {**receipt(1), **change}}, ORDER, FLOORS)


def test_step_floor_and_missing_score_refuse() -> None:
    low = receipt(1, summary={"num_steps": 10, "val_bpb": 1.4, "heldout_bpb": 1.4})
    with pytest.raises(ResumeMismatchError):
        select_retained({"r": low}, ORDER, FLOORS)
    bare = receipt(1, summary={"num_steps": 800, "val_bpb": 1.4})
    with pytest.raises(ResumeMismatchError):
        select_retained({"r": bare}, ORDER, FLOORS)


def test_two_valid_receipts_for_one_slot_refuse() -> None:
    with pytest.raises(ResumeMismatchError):
        select_retained({"a": receipt(1), "b": receipt(1)}, ORDER, FLOORS)
