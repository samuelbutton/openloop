"""Check probe orchestration over the public ledger API."""

from dataclasses import replace

import pytest

from openloop.ledger import (
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    Metric,
    NotFoundError,
)
from openloop.probe import run_probe


def test_probe_forces_measurement_and_persists_match(
    ledger, finish, tmp_path, inputs, result
):
    baseline = finish(ledger.submit(inputs), result)
    called = []

    def runner(snapshot):
        called.append(snapshot.hash)
        return result

    report = run_probe(ledger, baseline.id, runner)
    repeat = ledger.get(report.repeat_id)
    assert called == [inputs.hash]
    assert report.matches
    assert not report.differences
    assert not report.invalidates_cache
    assert repeat.attempt_id != baseline.attempt_id
    assert repeat.inputs == baseline.inputs
    assert repeat.purpose == "replication"
    assert repeat.parents == (baseline.id,)
    assert repeat.status == "succeeded"
    with Ledger(tmp_path / "ledger.sqlite3") as reopened:
        assert reopened.probes(baseline.id) == (report,)
        assert len(reopened.query(purpose="replication")) == 1


def test_probe_drift_disables_cache_for_deterministic_inputs(
    ledger, finish, inputs, result
):
    baseline = finish(ledger.submit(inputs), result)
    changed = replace(result, metrics={"val_bpb": Metric(1.8, "BPB")})
    report = run_probe(ledger, baseline.id, lambda _: changed)
    assert not report.matches
    assert report.invalidates_cache
    assert report.differences == ("metric val_bpb: 1.75 -> 1.8",)
    fresh = ledger.submit(inputs)
    assert fresh.reused_from is None
    assert fresh.attempt_id not in (
        baseline.attempt_id,
        ledger.get(report.repeat_id).attempt_id,
    )
    assert ledger.submit(inputs).attempt_id == fresh.attempt_id


def test_probe_drift_keeps_cache_for_noisy_inputs(ledger, finish, inputs, result):
    noisy = replace(inputs, reproducibility="noisy")
    baseline = finish(ledger.submit(noisy), result)
    changed = replace(result, metrics={"val_bpb": Metric(1.8, "BPB")})
    report = run_probe(ledger, baseline.id, lambda _: changed)
    assert not report.matches
    assert not report.invalidates_cache
    assert ledger.probes(baseline.id) == (report,)
    cached = ledger.submit(noisy)
    assert cached.reused_from == baseline.id
    assert cached.result == result


def test_probe_failure_is_recorded(ledger, finish, inputs, result):
    baseline = finish(ledger.submit(inputs), result)

    def crash(_):
        raise RuntimeError("Runner failed")

    with pytest.raises(RuntimeError, match="Runner failed"):
        run_probe(ledger, baseline.id, crash)
    failed = ledger.query(status="failed", purpose="replication")
    assert len(failed) == 1
    assert failed[0].events[-1].payload["reason"] == "RuntimeError: Runner failed"
    assert ledger.probes(baseline.id) == ()
    assert ledger.get(baseline.id).result == result


def test_probe_interruption_is_recorded(ledger, finish, inputs, result):
    baseline = finish(ledger.submit(inputs), result)

    def interrupt(_):
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        run_probe(ledger, baseline.id, interrupt)
    (failed,) = ledger.query(status="failed", purpose="replication")
    assert failed.events[-1].payload["reason"] == "KeyboardInterrupt: stop"


def test_non_result_return_fails_the_replication(ledger, finish, inputs, result):
    baseline = finish(ledger.submit(inputs), result)
    # The runner deliberately returns the wrong type to test validation.
    with pytest.raises(InvalidInputError):
        run_probe(
            ledger,
            baseline.id,
            lambda _: {"val_bpb": 1.75},  # pyright: ignore[reportArgumentType]
        )
    (failed,) = ledger.query(status="failed", purpose="replication")
    assert "InvalidInputError" in failed.events[-1].payload["reason"]


def test_invalid_probe_does_not_admit_work(ledger, finish, inputs, result):
    baseline = finish(ledger.submit(inputs), result)
    with pytest.raises(InvalidInputError):
        run_probe(ledger, baseline.id, lambda _: result, atol=float("nan"))
    queued = ledger.submit(replace(inputs, seed=1))
    with pytest.raises(InvalidTransitionError):
        run_probe(ledger, queued.id, lambda _: result)
    with pytest.raises(NotFoundError):
        run_probe(ledger, "missing", lambda _: result)
    assert ledger.query(purpose="replication") == ()
