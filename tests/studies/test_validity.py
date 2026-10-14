"""Synthetic validity tests: injectable clocks, no sleep, no training."""

import asyncio

import pytest

from openloop.ledger import InvalidInputError
from openloop.studies.validity import (
    MAX_ATTEMPTS_PER_SLOT,
    STEP_FRACTION,
    Attempt,
    SleepDriftMonitor,
    SlotExhaustedError,
    is_valid,
    reference_steps,
    run_slot,
    settle_wait,
    step_floor,
    steps_invalid_reason,
    valid_receipts,
)


def test_drift_does_not_trip_at_or_below_the_limit() -> None:
    monitor = SleepDriftMonitor(1000.0, 50.0, limit=5.0)
    assert monitor.observe(1005.0, 55.0) is None
    assert monitor.observe(1013.0, 60.0) is None  # 3 s of sleep
    assert monitor.observe(1017.0, 62.0) is None  # 5 s in total, not above
    assert monitor.total == pytest.approx(5.0)


def test_drift_trips_above_the_limit_with_the_reason() -> None:
    monitor = SleepDriftMonitor(1000.0, 50.0, limit=5.0)
    assert monitor.observe(1003.0, 51.0) is None
    trip = monitor.observe(1010.0, 53.0)
    assert trip is not None
    assert trip.drift_seconds == pytest.approx(7.0)
    assert trip.reason == "invalid: machine sleep (drift 7.0 s)"


def test_backward_wall_clock_step_does_not_hide_earlier_sleep() -> None:
    monitor = SleepDriftMonitor(1000.0, 50.0, limit=5.0)
    monitor.observe(1010.0, 52.0)  # 8 s of sleep
    assert monitor.total == pytest.approx(8.0)
    monitor.observe(1000.0, 54.0)
    assert monitor.total == pytest.approx(8.0)


def test_step_floor_marks_invalid_below_the_fraction_only() -> None:
    floor = step_floor(844.0)
    assert floor == pytest.approx(844.0 * STEP_FRACTION)
    assert steps_invalid_reason(717, floor) is not None
    assert steps_invalid_reason(718, floor) is None
    assert steps_invalid_reason(359, floor) is not None


def test_reference_is_the_median_of_positive_counts() -> None:
    assert reference_steps({"a": [838, 844, 844, 844, 844], "b": [1, 3]}) == {
        "a": 844.0,
        "b": 2.0,
    }
    with pytest.raises(InvalidInputError):
        reference_steps({"a": []})
    with pytest.raises(InvalidInputError):
        reference_steps({"a": [0]})


def test_settle_waits_until_a_full_quiet_period() -> None:
    state = {"wall": 0.0, "mono": 0.0, "calls": 0}

    def sleep(seconds: float) -> None:
        state["calls"] += 1
        state["wall"] += seconds
        state["mono"] += seconds
        if state["calls"] == 4:  # a sleep event restarts the quiet period
            state["wall"] += 30.0

    waited = settle_wait(
        wall=lambda: state["wall"],
        monotonic=lambda: state["mono"],
        sleep=sleep,
        settle_seconds=60.0,
        interval=5.0,
    )
    assert state["calls"] == 4 + 12
    assert waited == pytest.approx(80.0)


def _slot(verdicts: list[Attempt]) -> tuple[list[int], list[int], object]:
    ran: list[int] = []
    settled: list[int] = []

    async def attempt_once(number: int) -> Attempt:
        ran.append(number)
        return verdicts[number - 1]

    async def settle() -> None:
        settled.append(len(ran))

    return ran, settled, (attempt_once, settle)


def test_invalid_attempts_are_re_executed_until_one_is_valid() -> None:
    bad = Attempt(False, "invalid: machine sleep (drift 9.0 s)")
    ran, settled, (once, settle) = _slot([bad, bad, Attempt(True)])  # type: ignore[misc]
    attempts = asyncio.run(run_slot(once, settle))
    assert ran == [1, 2, 3]
    assert settled == [1, 2]
    assert [a.valid for a in attempts] == [False, False, True]


def test_valid_first_attempt_is_not_repeated() -> None:
    ran, settled, (once, settle) = _slot([Attempt(True)])  # type: ignore[misc]
    asyncio.run(run_slot(once, settle))
    assert ran == [1]
    assert settled == []


def test_slot_stops_the_study_after_the_cap() -> None:
    bad = Attempt(False, "invalid: throttled")
    ran, settled, (once, settle) = _slot([bad] * 5)  # type: ignore[misc]
    with pytest.raises(SlotExhaustedError, match="throttled"):
        asyncio.run(run_slot(once, settle))
    assert ran == list(range(1, MAX_ATTEMPTS_PER_SLOT + 1))
    assert settled == [1, 2]


def test_only_receipts_marked_valid_count() -> None:
    receipts = [
        {"id": 1, "validity": {"valid": True}},
        {"id": 2, "validity": {"valid": False, "reason": "x"}},
        {"id": 3},
        {"id": 4, "validity": {"valid": "yes"}},
    ]
    assert [r["id"] for r in valid_receipts(receipts)] == [1]
    assert not is_valid(receipts[2])
