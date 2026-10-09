"""Check persisted provenance, admission races, cache identity, and measured drift."""

import signal
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from openloop.ledger import (
    ExperimentInputs,
    Ledger,
    Metric,
    Result,
    canonical_json,
    compare_results,
    content_hash,
)


@pytest.fixture
def inputs():
    return ExperimentInputs(
        source_hash="1" * 64,
        dependencies_hash="2" * 64,
        data_hash="3" * 64,
        environment_hash="4" * 64,
        evaluator_hash="5" * 64,
        loop_hash="6" * 64,
        seed=42,
        fidelity="screen",
        budget_unit="tokens",
        budget_amount=1024,
        config={"depth": 4, "layers": [1, 2]},
        execution={"device": "cpu"},
    )


@pytest.fixture
def result():
    return Result({"val_bpb": Metric(1.75, "BPB")}, {"checkpoint": "7" * 64})


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.sqlite3") as store:
        yield store


def finish(ledger, run, result):
    ledger.start_stage(run.id, "execution")
    ledger.start_stage(run.id, "evaluation")
    ledger.start_stage(run.id, "decision")
    return ledger.complete(run.id, result)


def test_canonical_identity_and_frozen_snapshots(inputs):
    assert canonical_json({"z": 1, "a": [2]}) == '{"a":[2],"z":1}'
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    config = {"layers": [1, {"width": 32}]}
    snapshot = replace(inputs, config=config)
    identity = snapshot.hash
    config["layers"][1]["width"] = 64
    assert snapshot.hash == identity
    with pytest.raises(TypeError):
        snapshot.config["new"] = 1
    with pytest.raises(TypeError):
        snapshot.config["layers"][1]["width"] = 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_hash", "a" * 64),
        ("dependencies_hash", "a" * 64),
        ("data_hash", "a" * 64),
        ("environment_hash", "a" * 64),
        ("evaluator_hash", "a" * 64),
        ("loop_hash", "a" * 64),
        ("tokenizer_hash", "a" * 64),
        ("seed", 43),
        ("fidelity", "confirm"),
        ("budget_unit", "epochs"),
        ("budget_amount", 2048),
        ("config", {"depth": 5}),
        ("execution", {"device": "gpu"}),
        ("stage", "held_out"),
    ],
)
def test_every_declared_input_changes_identity(inputs, field, value):
    assert replace(inputs, **{field: value}).hash != inputs.hash


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_inputs_and_results_are_rejected(inputs, value):
    with pytest.raises(ValueError):
        replace(inputs, config={"value": value})
    with pytest.raises(ValueError):
        Metric(value, "BPB")


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": True},
        {"seed": -1},
        {"budget_amount": 0},
        {"source_hash": "unversioned"},
        {"config": {1: "bad key"}},
        {"execution": ["not an object"]},
    ],
)
def test_invalid_inputs_fail_before_admission(inputs, changes):
    with pytest.raises(ValueError):
        replace(inputs, **changes)


def test_stage_timestamps_survive_reopen(ledger, tmp_path, inputs, result):
    run = ledger.submit(inputs, hypothesis="Increase depth")
    end = datetime.now(UTC) - timedelta(seconds=1)
    ledger.record_preparation(
        run.id,
        "proposal",
        started_at=end - timedelta(seconds=2),
        finished_at=end,
    )
    ledger.record_preparation(
        run.id,
        "implementation",
        started_at=end,
        finished_at=end,
    )
    assert ledger.get(run.id).status == "queued"
    completed = finish(ledger, run, result)
    assert completed.status == "succeeded"
    with Ledger(tmp_path / "ledger.sqlite3") as reopened:
        restored = reopened.get(run.id)
        assert restored == completed
        assert restored.result == result
        assert set(restored.stages) == {
            "proposal",
            "implementation",
            "queue",
            "execution",
            "evaluation",
            "decision",
        }
        for stage, events in restored.stages.items():
            assert events
            for event in events:
                assert datetime.fromisoformat(event.timestamp).utcoffset() == timedelta(
                    0
                )
            if stage not in ("proposal", "implementation"):
                assert [event.kind for event in events] == [
                    "stage_started",
                    "stage_finished",
                ]
                assert events[-1].payload["duration_seconds"] >= 0
                assert events[-1].payload["duration_clock"] == "monotonic"


def test_cache_reuses_result_without_duplicate_samples(ledger, inputs, result):
    baseline = finish(ledger, ledger.submit(inputs), result)
    parent = ledger.submit(replace(inputs, seed=43))
    cached = ledger.submit(inputs, hypothesis="Same inputs", parents=(parent.id,))
    assert cached.id != baseline.id
    assert cached.attempt_id == baseline.attempt_id
    assert cached.reused_from == baseline.id
    assert cached.parents == (parent.id,)
    assert cached.hypothesis == "Same inputs"
    assert cached.result == baseline.result
    assert len(ledger.query(status="succeeded")) == 1
    assert len(ledger.query(status="succeeded", include_reused=True)) == 2
    assert {run.id for run in ledger.lineage(cached.id)} == {
        baseline.id,
        parent.id,
        cached.id,
    }
    with pytest.raises(ValueError, match="Reused"):
        ledger.fail(cached.id, "Cannot edit a cache alias")


def test_atomic_admission_between_connections(tmp_path, inputs):
    path = tmp_path / "concurrent.sqlite3"
    barrier = Barrier(2)

    def submit():
        barrier.wait(timeout=10)
        with Ledger(path) as ledger:
            return ledger.submit(inputs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        runs = [future.result(timeout=20) for future in futures]
    assert runs[0].attempt_id == runs[1].attempt_id
    assert sum(run.reused_from is None for run in runs) == 1
    with Ledger(path) as ledger:
        assert len(ledger.query()) == 1
        assert len(ledger.query(include_reused=True)) == 2


def test_request_key_is_idempotent_and_rejects_conflicts(ledger, inputs):
    run = ledger.submit(inputs, request_key="job-1")
    assert ledger.submit(inputs, request_key="job-1") == run
    with pytest.raises(ValueError, match="different inputs"):
        ledger.submit(replace(inputs, seed=43), request_key="job-1")
    with pytest.raises(ValueError, match="different inputs"):
        ledger.submit(inputs, hypothesis="New provenance", request_key="job-1")
    assert len(ledger.query(include_reused=True)) == 1


def test_retry_preserves_failed_attempt_and_cache_key(ledger, inputs, result):
    original = ledger.submit(inputs)
    ledger.fail(original.id, "Worker was interrupted")
    retry = ledger.submit(inputs, purpose="retry", retry_of=original.id)
    assert retry.attempt_id != original.attempt_id
    assert retry.retry_of == original.attempt_id
    assert retry.parents == (original.id,)
    assert retry.inputs.hash == original.inputs.hash
    finish(ledger, retry, result)
    cache = ledger.submit(inputs, request_key="from-retry")
    assert cache.reused_from == retry.id
    assert cache.purpose == "run"
    assert cache.retry_of is None
    assert ledger.submit(inputs, request_key="from-retry") == cache
    assert (
        ledger.get(original.id).events[-1].payload["reason"] == "Worker was interrupted"
    )


def test_failed_runs_and_replications_do_not_reuse_pending_work(ledger, inputs):
    original = ledger.submit(inputs)
    replication = ledger.submit(inputs, purpose="replication")
    assert replication.attempt_id != original.attempt_id
    ledger.fail(original.id, "Crash")
    next_run = ledger.submit(inputs)
    assert next_run.attempt_id not in (original.attempt_id, replication.attempt_id)
    with pytest.raises(ValueError, match="failed run"):
        ledger.submit(inputs, purpose="retry", retry_of=next_run.id)


def test_lineage_preserves_diamond_and_combined_parents(ledger, inputs):
    root = ledger.submit(inputs)
    left = ledger.submit(replace(inputs, seed=1), parents=(root.id,))
    right = ledger.submit(replace(inputs, seed=2), parents=(root.id,))
    child = ledger.submit(replace(inputs, seed=3), parents=(left.id, right.id))
    ancestors = ledger.lineage(child.id)
    positions = {run.id: index for index, run in enumerate(ancestors)}
    assert len(ancestors) == 4
    assert ancestors[0].id == root.id
    assert ancestors[-1].id == child.id
    for run in ancestors:
        assert all(positions[parent] < positions[run.id] for parent in run.parents)
    with pytest.raises(KeyError):
        ledger.submit(inputs, parents=("missing",))
    assert len(ledger.query()) == 4


def test_query_filters_and_pagination(ledger, inputs, result):
    queued = ledger.submit(inputs)
    failed = ledger.submit(replace(inputs, seed=1))
    ledger.fail(failed.id, "Crash")
    succeeded = finish(ledger, ledger.submit(replace(inputs, seed=2)), result)
    assert ledger.query(status="queued") == (ledger.get(queued.id),)
    assert ledger.query(status="failed") == (ledger.get(failed.id),)
    assert ledger.query(experiment_hash=succeeded.inputs.hash) == (succeeded,)
    assert ledger.query(limit=1, offset=1)[0].id == failed.id
    assert ledger.query(status="succeeded", purpose="replication") == ()


def test_invalid_transitions_do_not_change_evidence(ledger, inputs, result):
    run = ledger.submit(inputs)
    with pytest.raises(ValueError, match="required"):
        ledger.complete(run.id, result)
    assert ledger.get(run.id) == run
    ledger.start_stage(run.id, "execution")
    with pytest.raises(ValueError, match="order"):
        ledger.start_stage(run.id, "queue")
    ledger.fail(run.id, "Execution failed")
    with pytest.raises(ValueError, match="terminal"):
        ledger.start_stage(run.id, "evaluation")


def test_preparation_rejects_unknown_times_and_overlap(ledger, inputs):
    run = ledger.submit(inputs)
    end = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(ValueError, match="timezone"):
        ledger.record_preparation(
            run.id,
            "proposal",
            started_at=end.replace(tzinfo=None),
            finished_at=end,
        )
    ledger.record_preparation(
        run.id,
        "proposal",
        started_at=end - timedelta(seconds=2),
        finished_at=end,
    )
    with pytest.raises(ValueError, match="overlap"):
        ledger.record_preparation(
            run.id,
            "implementation",
            started_at=end - timedelta(seconds=1),
            finished_at=end,
        )
    assert ledger.get(run.id).status == "queued"


def test_probe_forces_measurement_and_persists_match(ledger, tmp_path, inputs, result):
    baseline = finish(ledger, ledger.submit(inputs), result)
    called = []

    def runner(snapshot):
        called.append(snapshot.hash)
        return result

    report = ledger.probe(baseline.id, runner)
    repeat = ledger.get(report.repeat_id)
    assert called == [inputs.hash]
    assert report.matches and not report.differences
    assert repeat.attempt_id != baseline.attempt_id
    assert repeat.inputs == baseline.inputs
    assert repeat.purpose == "replication"
    assert repeat.parents == (baseline.id,)
    with Ledger(tmp_path / "ledger.sqlite3") as reopened:
        assert reopened.probes(baseline.id) == (report,)
        assert len(reopened.query(purpose="replication")) == 1


def test_probe_drift_disables_cache(ledger, inputs, result):
    baseline = finish(ledger, ledger.submit(inputs), result)
    changed = replace(result, metrics={"val_bpb": Metric(1.8, "BPB")})
    report = ledger.probe(baseline.id, lambda _: changed)
    assert not report.matches
    assert report.differences == ("metric val_bpb: 1.75 -> 1.8",)
    fresh = ledger.submit(inputs)
    assert fresh.reused_from is None
    assert fresh.attempt_id not in (
        baseline.attempt_id,
        ledger.get(report.repeat_id).attempt_id,
    )
    assert ledger.submit(inputs).attempt_id == fresh.attempt_id


def test_probe_failure_is_recorded(ledger, inputs, result):
    baseline = finish(ledger, ledger.submit(inputs), result)

    def crash(_):
        raise RuntimeError("Runner failed")

    with pytest.raises(RuntimeError, match="Runner failed"):
        ledger.probe(baseline.id, crash)
    failed = ledger.query(status="failed", purpose="replication")
    assert len(failed) == 1
    assert failed[0].events[-1].payload["reason"] == "RuntimeError: Runner failed"
    assert ledger.probes(baseline.id) == ()
    assert ledger.get(baseline.id).result == result


def test_invalid_probe_tolerance_does_not_admit_work(ledger, inputs, result):
    baseline = finish(ledger, ledger.submit(inputs), result)
    with pytest.raises(ValueError):
        ledger.probe(baseline.id, lambda _: result, atol=float("nan"))
    assert ledger.query(purpose="replication") == ()


def test_probe_completion_and_comparison_are_atomic(ledger, tmp_path, inputs, result):
    baseline = finish(ledger, ledger.submit(inputs), result)
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as db:
        db.execute("""
            CREATE TRIGGER reject_probe BEFORE INSERT ON probe
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END
        """)
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        ledger.probe(baseline.id, lambda _: result)
    (repeat,) = ledger.query(purpose="replication")
    assert repeat.status == "failed"
    assert all(event.kind != "succeeded" for event in repeat.events)
    assert ledger.probes(baseline.id) == ()


def test_comparison_tolerances_and_metadata(result):
    slightly_changed = replace(result, metrics={"val_bpb": Metric(1.751, "BPB")})
    assert compare_results(result, slightly_changed, atol=0.002) == ()
    assert compare_results(result, slightly_changed, rtol=0.001) == ()
    assert compare_results(result, slightly_changed)
    metadata_changed = replace(result, metrics={"val_bpb": Metric(1.75, "nats")})
    assert compare_results(result, metadata_changed) == (
        "metric val_bpb: metadata differs",
    )
    artifact_changed = replace(result, artifacts={"checkpoint": "8" * 64})
    assert compare_results(result, artifact_changed) == (
        "artifact checkpoint: hash differs",
    )
    missing_metric = Result({"accuracy": Metric(0.5, "fraction", "maximize")})
    assert "metric val_bpb: missing or extra" in compare_results(result, missing_metric)
    for tolerance in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            compare_results(result, result, atol=tolerance)


def test_append_only_records_and_schema_version(tmp_path, inputs):
    path = tmp_path / "protected.sqlite3"
    with Ledger(path) as ledger:
        ledger.submit(inputs)
    with sqlite3.connect(path) as db:
        for table in (
            "experiment",
            "attempt",
            "submission",
            "parent",
            "event",
            "probe",
        ):
            triggers = db.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = ?",
                (table,),
            ).fetchone()[0]
            assert triggers == 2
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM event")
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="Unsupported"):
        Ledger(path)


def test_killed_transaction_leaves_no_partial_submission(tmp_path, inputs):
    path = tmp_path / "killed.sqlite3"
    with Ledger(path):
        pass
    script = """
import json, os, signal, sys
from openloop.ledger import ExperimentInputs, Ledger
class KilledLedger(Ledger):
    def _event(self, *args):
        super()._event(*args)
        os.kill(os.getpid(), signal.SIGKILL)
with KilledLedger(sys.argv[1]) as ledger:
    ledger.submit(ExperimentInputs(**json.loads(sys.argv[2])))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path), canonical_json(inputs)],
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr.decode()
    with Ledger(path) as ledger:
        assert ledger.query() == ()
        assert ledger.submit(inputs).status == "queued"
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_interrupted_run_stays_visible_and_can_be_retried(tmp_path, inputs, result):
    path = tmp_path / "restart.sqlite3"
    with Ledger(path) as ledger:
        run = ledger.submit(inputs)
        ledger.start_stage(run.id, "execution")
    with Ledger(path) as ledger:
        assert ledger.get(run.id).status == "running"
        assert ledger.submit(inputs).attempt_id == run.attempt_id
        failed = ledger.fail(run.id, "Worker lost after coordinator restart")
        assert failed.stages["execution"][-1].payload["duration_clock"] == "wall"
        retry = ledger.submit(inputs, purpose="retry", retry_of=run.id)
        finish(ledger, retry, result)
        assert len(ledger.query()) == 2


def test_killed_completion_does_not_publish_a_partial_result(tmp_path, inputs, result):
    path = tmp_path / "killed-result.sqlite3"
    with Ledger(path) as ledger:
        run = ledger.submit(inputs)
        ledger.start_stage(run.id, "execution")
        before = ledger.start_stage(run.id, "evaluation")
    script = """
import os, signal, sys
from openloop.ledger import Ledger, Metric, Result
class KilledLedger(Ledger):
    def _event(self, attempt_id, kind, payload):
        super()._event(attempt_id, kind, payload)
        if kind == 'succeeded':
            os.kill(os.getpid(), signal.SIGKILL)
with KilledLedger(sys.argv[1]) as ledger:
    ledger.complete(sys.argv[2], Result({'score': Metric(0.5, 'fraction')}))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path), run.id],
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr.decode()
    with Ledger(path) as ledger:
        assert ledger.get(run.id) == before
        assert ledger.query(status="succeeded") == ()
        completed = ledger.complete(run.id, result)
        assert completed.result == result
