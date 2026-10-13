"""Record validation and capture through an isolated public ledger."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest

from openloop.executors import (
    InvalidJobError,
    ProcessJob,
    RecordedRun,
    ReplaySnapshot,
    State,
)
from openloop.ledger import Ledger, Metric, Purpose, Result, Run, Stage


def test_capture_preserves_outcomes_and_provenance(
    ledger: Ledger, inputs, result: Result, finish
) -> None:
    run: Run = finish(ledger.submit(inputs), result)
    job = ProcessJob(("unused",), stdin=inputs.hash.encode())
    recording = RecordedRun.from_run(run, job_hash=job.hash)
    assert recording.job_hash == job.hash
    assert recording.result == result
    assert recording.state is State.SUCCEEDED
    assert (
        recording.duration_seconds
        == run.stages[Stage.EXECUTION][-1].payload["duration_seconds"]
    )
    assert recording.hash != replace(recording, duration_seconds=10).hash
    assert (
        recording.hash
        != replace(recording, result=Result({"score": Metric(0, "unit")})).hash
    )
    assert recording.source_hash != recording.hash
    alias = ledger.submit(inputs)
    with pytest.raises(InvalidJobError, match="cache alias"):
        RecordedRun.from_run(alias, job_hash=job.hash)
    with pytest.raises(FrozenInstanceError):
        recording.duration_seconds = 99  # pyright: ignore[reportAttributeAccessIssue]


def test_explicit_observation_duration_does_not_invent_missing_time(
    ledger: Ledger, inputs, finish
) -> None:
    result = Result(
        {"score": Metric(1, "unit")},
        observations={"process_seconds": Metric(12.5, "seconds", split="execution")},
    )
    run = finish(ledger.submit(inputs), result)
    assert (
        RecordedRun.from_run(
            run, job_hash="a" * 64, duration_metric="process_seconds"
        ).duration_seconds
        == 12.5
    )
    with pytest.raises(InvalidJobError, match="Missing"):
        RecordedRun.from_run(run, job_hash="a" * 64, duration_metric="missing")
    with pytest.raises(InvalidJobError, match="stage"):
        RecordedRun.from_run(replace(run, events=()), job_hash="a" * 64)


def test_failed_runs_have_recorded_time_and_no_score(ledger: Ledger, inputs) -> None:
    run = ledger.submit(inputs)
    with pytest.raises(InvalidJobError, match="terminal"):
        RecordedRun.from_run(run, job_hash="a" * 64)
    ledger.start_stage(run.id, "execution")
    failed = ledger.fail(run.id, "worker crashed")
    recording = RecordedRun.from_run(failed, job_hash="a" * 64)
    assert recording.state is State.FAILED
    assert recording.result is None
    assert recording.error == "worker crashed"
    assert recording.duration_seconds >= 0


def test_replications_remain_separate_recordings(
    ledger: Ledger, inputs, result, finish
) -> None:
    first = finish(ledger.submit(inputs), result)
    second = finish(ledger.submit(inputs, purpose=Purpose.REPLICATION), result)
    left = RecordedRun.from_run(first, job_hash="a" * 64)
    right = RecordedRun.from_run(second, job_hash="a" * 64)
    assert left.source_hash != right.source_hash


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "1"])
def test_invalid_recorded_time_is_rejected(value) -> None:
    with pytest.raises(InvalidJobError, match="duration"):
        RecordedRun("a" * 64, "b" * 64, value, State.SUCCEEDED)


@pytest.mark.parametrize(
    "state", [State.QUEUED, State.RUNNING, State.EXPIRED, "unknown"]
)
def test_pending_or_unstarted_recordings_are_rejected(state) -> None:
    with pytest.raises(InvalidJobError):
        RecordedRun("a" * 64, "b" * 64, 1, state)


def test_failure_and_simulated_receipt_cannot_expose_a_successful_score() -> None:
    result = Result({"score": Metric(1, "unit")})
    with pytest.raises(InvalidJobError, match="Unsuccessful"):
        RecordedRun("a" * 64, "b" * 64, 1, State.FAILED, result=result)
    with pytest.raises(InvalidJobError, match="successful"):
        ReplaySnapshot(
            "mock-1",
            "a" * 64,
            State.FAILED,
            datetime(2000, 1, 1, tzinfo=UTC),
            recording_hash="b" * 64,
            recorded_result=result,
        )
