"""Check persisted provenance, admission races, cache identity, and recovery."""

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
    SCHEMA_VERSION,
    ConflictError,
    EventKind,
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    LedgerError,
    Metric,
    NotFoundError,
    Result,
    SchemaError,
    Stage,
    canonical_json,
)


def test_stage_timestamps_survive_reopen(ledger, finish, tmp_path, inputs, result):
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
    completed = finish(run, result)
    assert completed.status == "succeeded"
    with Ledger(tmp_path / "ledger.sqlite3") as reopened:
        restored = reopened.get(run.id)
        assert restored == completed
        assert restored.result == result
        assert [item.stage for item in restored.preparations] == [
            "proposal",
            "implementation",
        ]
        assert restored.preparations[0].duration_seconds == 2
        assert set(restored.stages) == set(Stage)
        for events in restored.stages.values():
            assert [event.kind for event in events] == [
                EventKind.STAGE_STARTED,
                EventKind.STAGE_FINISHED,
            ]
            for event in events:
                assert datetime.fromisoformat(event.timestamp).utcoffset() == timedelta(
                    0
                )
            duration = events[-1].payload["duration_seconds"]
            assert isinstance(duration, int | float)
            assert duration >= 0
            assert events[-1].payload["duration_clock"] == "monotonic"


def test_ticks_do_not_accumulate(ledger, finish, inputs, result):
    finish(ledger.submit(inputs), result)
    failed = ledger.submit(replace(inputs, seed=1))
    ledger.start_stage(failed.id, "execution")
    ledger.fail(failed.id, "Crash")
    assert ledger._ticks == {}


def test_cache_reuses_result_without_duplicate_samples(ledger, finish, inputs, result):
    baseline = finish(ledger.submit(inputs), result)
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
    with pytest.raises(InvalidTransitionError, match="Reused"):
        ledger.fail(cached.id, "Cannot edit a cache alias")
    with pytest.raises(InvalidTransitionError, match="Reused"):
        ledger.start_stage(cached.id, "execution")


def test_preparation_belongs_to_each_submission(ledger, finish, inputs, result):
    baseline = finish(ledger.submit(inputs, hypothesis="First"), result)
    cached = ledger.submit(inputs, hypothesis="Second")
    assert cached.reused_from == baseline.id
    end = datetime.now(UTC) - timedelta(seconds=1)
    updated = ledger.record_preparation(
        cached.id, "proposal", started_at=end - timedelta(seconds=5), finished_at=end
    )
    assert [item.stage for item in updated.preparations] == ["proposal"]
    assert updated.status == "succeeded"
    assert ledger.get(baseline.id).preparations == ()
    ledger.record_preparation(
        baseline.id, "proposal", started_at=end - timedelta(seconds=9), finished_at=end
    )
    assert ledger.get(baseline.id).preparations[0].duration_seconds == 9
    assert ledger.get(cached.id).preparations[0].duration_seconds == 5


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
    with pytest.raises(ConflictError, match="different inputs"):
        ledger.submit(replace(inputs, seed=43), request_key="job-1")
    with pytest.raises(ConflictError, match="different inputs"):
        ledger.submit(inputs, hypothesis="New provenance", request_key="job-1")
    assert len(ledger.query(include_reused=True)) == 1


def test_retry_preserves_failed_attempt_and_cache_key(ledger, finish, inputs, result):
    original = ledger.submit(inputs)
    ledger.fail(original.id, "Worker was interrupted")
    retry = ledger.submit(inputs, purpose="retry", retry_of=original.id)
    assert retry.attempt_id != original.attempt_id
    assert retry.retry_of == original.attempt_id
    assert retry.parents == (original.id,)
    assert retry.inputs.hash == original.inputs.hash
    finish(retry, result)
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
    with pytest.raises(InvalidTransitionError, match="failed run"):
        ledger.submit(inputs, purpose="retry", retry_of=next_run.id)
    with pytest.raises(InvalidInputError, match="identical inputs"):
        ledger.submit(replace(inputs, seed=1), purpose="retry", retry_of=original.id)
    with pytest.raises(InvalidInputError, match="retry_of"):
        ledger.submit(inputs, retry_of=original.id)


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
    with pytest.raises(NotFoundError):
        ledger.submit(inputs, parents=("missing",))
    assert len(ledger.query()) == 4


def test_query_filters_and_pagination(ledger, finish, inputs, result):
    queued = ledger.submit(inputs)
    failed = ledger.submit(replace(inputs, seed=1))
    ledger.fail(failed.id, "Crash")
    succeeded = finish(ledger.submit(replace(inputs, seed=2)), result)
    assert ledger.query(status="queued") == (ledger.get(queued.id),)
    assert ledger.query(status="failed") == (ledger.get(failed.id),)
    assert ledger.query(experiment_hash=succeeded.inputs.hash) == (succeeded,)
    assert ledger.query(limit=1, offset=1)[0].id == failed.id
    assert ledger.query(status="succeeded", purpose="replication") == ()
    with pytest.raises(InvalidInputError):
        ledger.query(status="finished")
    with pytest.raises(InvalidInputError):
        ledger.query(purpose="sample")


def test_query_groups_seeds_by_candidate(ledger, inputs):
    seeds = [ledger.submit(replace(inputs, seed=seed)) for seed in (1, 2, 3)]
    other = ledger.submit(replace(inputs, config={"depth": 8}))
    grouped = ledger.query(candidate_hash=inputs.candidate_hash)
    assert [run.id for run in grouped] == [run.id for run in seeds]
    assert ledger.query(candidate_hash=other.inputs.candidate_hash) == (other,)
    with pytest.raises(InvalidInputError):
        ledger.query(candidate_hash="not a digest")


def test_invalid_transitions_do_not_change_evidence(ledger, inputs, result):
    run = ledger.submit(inputs)
    with pytest.raises(InvalidTransitionError, match="required"):
        ledger.complete(run.id, result)
    assert ledger.get(run.id) == run
    ledger.start_stage(run.id, "execution")
    with pytest.raises(InvalidTransitionError, match="order"):
        ledger.start_stage(run.id, "queue")
    with pytest.raises(InvalidInputError):
        ledger.start_stage(run.id, "decision")
    ledger.fail(run.id, "Execution failed")
    with pytest.raises(InvalidTransitionError, match="terminal"):
        ledger.start_stage(run.id, "evaluation")


def test_unknown_runs_raise_not_found(ledger, result):
    for call in (
        lambda: ledger.get("missing"),
        lambda: ledger.start_stage("missing", "execution"),
        lambda: ledger.fail("missing", "reason"),
        lambda: ledger.complete("missing", result),
        lambda: ledger.probes("missing"),
        lambda: ledger.lineage("missing"),
    ):
        with pytest.raises(NotFoundError):
            call()
    assert issubclass(NotFoundError, (LedgerError, LookupError))
    assert issubclass(InvalidInputError, (LedgerError, ValueError))


def test_preparation_rejects_unknown_times_and_overlap(ledger, inputs):
    run = ledger.submit(inputs)
    end = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(InvalidInputError, match="timezone"):
        ledger.record_preparation(
            run.id,
            "proposal",
            started_at=end.replace(tzinfo=None),
            finished_at=end,
        )
    with pytest.raises(InvalidInputError, match="before submission"):
        ledger.record_preparation(
            run.id,
            "proposal",
            started_at=end,
            finished_at=datetime.now(UTC) + timedelta(seconds=60),
        )
    with pytest.raises(InvalidInputError, match="preparation stage"):
        ledger.record_preparation(run.id, "queue", started_at=end, finished_at=end)
    ledger.record_preparation(
        run.id,
        "proposal",
        started_at=end - timedelta(seconds=2),
        finished_at=end,
    )
    with pytest.raises(ConflictError, match="already"):
        ledger.record_preparation(run.id, "proposal", started_at=end, finished_at=end)
    with pytest.raises(InvalidInputError, match="overlap"):
        ledger.record_preparation(
            run.id,
            "implementation",
            started_at=end - timedelta(seconds=1),
            finished_at=end,
        )
    assert ledger.get(run.id).status == "queued"


def test_record_probe_validates_the_replication(ledger, finish, inputs, result):
    baseline = finish(ledger.submit(inputs), result)
    queued = ledger.submit(replace(inputs, seed=1))
    unrelated = ledger.submit(inputs, purpose="replication")
    other_inputs = ledger.submit(
        replace(inputs, seed=2), purpose="replication", parents=(baseline.id,)
    )
    plain_run = ledger.submit(replace(inputs, seed=3), parents=(baseline.id,))
    for repeat, error in (
        (unrelated, InvalidInputError),
        (other_inputs, InvalidInputError),
        (plain_run, InvalidInputError),
    ):
        with pytest.raises(error):
            ledger.record_probe(baseline.id, repeat.id, result)
    with pytest.raises(InvalidTransitionError, match="baseline"):
        ledger.record_probe(queued.id, unrelated.id, result)
    with pytest.raises(NotFoundError):
        ledger.record_probe(baseline.id, "missing", result)
    ready = ledger.submit(inputs, purpose="replication", parents=(baseline.id,))
    with pytest.raises(InvalidTransitionError, match="required"):
        ledger.record_probe(baseline.id, ready.id, result)
    ledger.start_stage(ready.id, "execution")
    ledger.start_stage(ready.id, "evaluation")
    with pytest.raises(InvalidInputError):
        ledger.record_probe(baseline.id, ready.id, result, atol=-1)
    assert ledger.get(ready.id).status == "running"
    report = ledger.record_probe(baseline.id, ready.id, result)
    assert report.matches
    assert not report.invalidates_cache
    with pytest.raises(InvalidTransitionError, match="terminal"):
        ledger.record_probe(baseline.id, ready.id, result)
    assert ledger.probes(baseline.id) == (report,)


def test_probe_completion_and_comparison_are_atomic(
    ledger, finish, tmp_path, inputs, result
):
    baseline = finish(ledger.submit(inputs), result)
    repeat = ledger.submit(inputs, purpose="replication", parents=(baseline.id,))
    ledger.start_stage(repeat.id, "execution")
    ledger.start_stage(repeat.id, "evaluation")
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as db:
        db.execute("""
            CREATE TRIGGER reject_probe BEFORE INSERT ON probe
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END
        """)
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        ledger.record_probe(baseline.id, repeat.id, result)
    after = ledger.get(repeat.id)
    assert after.status == "running"
    assert all(event.kind is not EventKind.SUCCEEDED for event in after.events)
    assert ledger.probes(baseline.id) == ()


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
            "preparation",
        ):
            triggers = db.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = ?",
                (table,),
            ).fetchone()[0]
            assert triggers == 2
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM event")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE event SET stage = 'bogus'")
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(SchemaError, match="Unsupported"):
        Ledger(path)


def test_event_stage_values_are_constrained(ledger, tmp_path, inputs):
    ledger.submit(inputs)
    with (
        sqlite3.connect(tmp_path / "ledger.sqlite3") as db,
        pytest.raises(sqlite3.IntegrityError, match="CHECK"),
    ):
        db.execute(
            "INSERT INTO event(attempt_id, kind, stage, timestamp, payload) "
            "SELECT id, 'stage_started', 'decision', '', '{}' FROM attempt"
        )


def test_v1_and_unrelated_databases_are_rejected(tmp_path):
    old = tmp_path / "v1.sqlite3"
    with sqlite3.connect(old) as db:
        db.execute("CREATE TABLE experiment (hash TEXT PRIMARY KEY, manifest TEXT)")
        db.execute("PRAGMA user_version = 1")
    with pytest.raises(SchemaError, match="version 1"):
        Ledger(old)
    other = tmp_path / "other.sqlite3"
    with sqlite3.connect(other) as db:
        db.execute("CREATE TABLE notes (body TEXT)")
    with pytest.raises(SchemaError, match="unrelated"):
        Ledger(other)


def test_killed_transaction_leaves_no_partial_submission(tmp_path, inputs):
    path = tmp_path / "killed.sqlite3"
    with Ledger(path):
        pass
    script = """
import json, os, signal, sys
from openloop.ledger import ExperimentInputs, Ledger
class KilledLedger(Ledger):
    def _event(self, *args, **kwargs):
        super()._event(*args, **kwargs)
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
        assert failed.stages[Stage.EXECUTION][-1].payload["duration_clock"] == "wall"
        retry = ledger.submit(inputs, purpose="retry", retry_of=run.id)
        ledger.start_stage(retry.id, "execution")
        ledger.start_stage(retry.id, "evaluation")
        ledger.complete(retry.id, result)
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
    def _event(self, attempt_id, kind, payload, **kwargs):
        super()._event(attempt_id, kind, payload, **kwargs)
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
        assert isinstance(completed.result, Result)
        assert completed.result.metrics["val_bpb"] == Metric(1.75, "BPB")


def test_observations_round_trip_and_legacy_payloads_default_empty(
    ledger, inputs, result
):
    timed = replace(result, observations={"seconds": Metric(2.5, "seconds")})
    run = ledger.submit(inputs)
    ledger.start_stage(run.id, "execution")
    ledger.start_stage(run.id, "evaluation")
    completed = ledger.complete(run.id, timed)
    assert completed.result is not None
    assert completed.result.observations["seconds"] == Metric(2.5, "seconds")
    assert ledger.get(run.id).result == timed

    legacy = ledger.submit(replace(inputs, seed=inputs.seed + 1))
    ledger.start_stage(legacy.id, "execution")
    ledger.start_stage(legacy.id, "evaluation")
    with ledger._transaction():  # pyright: ignore[reportPrivateUsage]
        ledger._event(  # pyright: ignore[reportPrivateUsage]
            legacy.attempt_id,
            EventKind.SUCCEEDED,
            {
                "metrics": {
                    "val_bpb": {
                        "value": 1.0,
                        "unit": "BPB",
                        "direction": "minimize",
                        "split": "validation",
                        "sample_count": 1,
                    }
                },
                "artifacts": {},
            },
        )
    old = ledger.get(legacy.id)
    assert old.result is not None
    assert old.result.observations == {}
