"""Atomic operator admission and migration, with fresh temporary ledgers only."""

import hashlib
import os
import selectors
import signal
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openloop.ledger import (
    SCHEMA_VERSION,
    ActivityKind,
    ConflictError,
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    ReadOnlyError,
    SchemaError,
)

SUBJECT = "a" * 64


def hypothesis(ledger: Ledger) -> None:
    ledger.record_activity(ActivityKind.HYPOTHESIS, SUBJECT, {"card": "fixture"})


def test_activity_idempotency_and_immutability(ledger: Ledger) -> None:
    payload = {"nested": {"value": 1}}
    first = ledger.record_activity(
        ActivityKind.HYPOTHESIS, SUBJECT, payload, key="card"
    )
    payload["nested"]["value"] = 2
    assert (
        ledger.record_activity(
            ActivityKind.HYPOTHESIS, SUBJECT, {"nested": {"value": 1}}, key="card"
        )
        == first
    )
    before = ledger.activities()
    with pytest.raises(ConflictError):
        ledger.record_activity(ActivityKind.HYPOTHESIS, SUBJECT, payload, key="card")
    assert ledger.activities() == before
    assert ledger.query(limit=100) == ()


def test_retry_cap_survives_reopening_without_idea_failures(tmp_path: Path) -> None:
    path = tmp_path / "operators.sqlite3"
    with Ledger(path) as ledger:
        hypothesis(ledger)
        for number in range(1, 4):
            start = ledger.record_activity(
                ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {"plan": "fixed"}
            )
            assert start.payload["attempt"] == number
            ledger.record_activity(
                ActivityKind.IMPLEMENTATION_FAILURE,
                SUBJECT,
                {"reason": "fixture failure"},
                parent_sequence=start.sequence,
            )
    with Ledger(path) as ledger:
        before = ledger.activities()
        with pytest.raises(InvalidTransitionError, match="limit"):
            ledger.record_activity(
                ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {"plan": "fixed"}
            )
        assert ledger.activities() == before
        assert all(event.kind is not ActivityKind.IDEA_FAILURE for event in before)


def test_pending_attempt_blocks_admission_and_ends_once(ledger: Ledger) -> None:
    hypothesis(ledger)
    start = ledger.record_activity(ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {})
    before = ledger.activities()
    with pytest.raises(InvalidTransitionError, match="unresolved"):
        ledger.record_activity(ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {})
    assert ledger.activities() == before
    ledger.record_activity(
        ActivityKind.IMPLEMENTATION_SUCCESS, SUBJECT, {}, parent_sequence=start.sequence
    )
    with pytest.raises(InvalidTransitionError, match="once"):
        ledger.record_activity(
            ActivityKind.IMPLEMENTATION_FAILURE,
            SUBJECT,
            {},
            parent_sequence=start.sequence,
        )
    with pytest.raises(InvalidTransitionError, match="already implemented"):
        ledger.record_activity(ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {})


def test_retry_cannot_change_source_or_smoke_policy(ledger: Ledger) -> None:
    hypothesis(ledger)
    start = ledger.record_activity(
        ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {"plan": "first"}, key="attempt"
    )
    assert (
        ledger.record_activity(
            ActivityKind.IMPLEMENTATION_STARTED,
            SUBJECT,
            {"plan": "first"},
            key="attempt",
        )
        == start
    )
    ledger.record_activity(
        ActivityKind.IMPLEMENTATION_FAILURE, SUBJECT, {}, parent_sequence=start.sequence
    )
    with pytest.raises(ConflictError, match="frozen"):
        ledger.record_activity(
            ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {"plan": "changed"}
        )


def test_concurrent_admission_has_one_owner(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite3"
    with Ledger(path) as ledger:
        hypothesis(ledger)

    def admit(_: int) -> bool:
        with Ledger(path) as ledger:
            try:
                ledger.record_activity(ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {})
            except InvalidTransitionError:
                return False
            return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(admit, range(2))) == [False, True]


def test_v2_migration_preserves_identity_and_cache(
    tmp_path: Path, inputs, result
) -> None:
    path = tmp_path / "v2.sqlite3"
    with Ledger(path) as ledger:
        run = ledger.submit(inputs)
        ledger.start_stage(run.id, "execution")
        ledger.start_stage(run.id, "evaluation")
        original = ledger.complete(run.id, result)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE activity")
        db.execute("PRAGMA user_version = 2")
    with Ledger(path, migrate=True) as ledger:
        assert ledger.get(original.id) == original
        assert ledger.submit(inputs).attempt_id == original.attempt_id
        hypothesis(ledger)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM activity")


def make_v2(path: Path, inputs, result) -> str:
    with Ledger(path) as ledger:
        run = ledger.submit(inputs)
        ledger.start_stage(run.id, "execution")
        ledger.start_stage(run.id, "evaluation")
        ledger.complete(run.id, result)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE activity")
        db.execute("PRAGMA user_version = 2")
        db.execute("PRAGMA journal_mode = DELETE")
    return run.id


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schema_of(path: Path) -> tuple[int, list[tuple[str, str]]]:
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        return (
            db.execute("PRAGMA user_version").fetchone()[0],
            db.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall(),
        )
    finally:
        db.close()


def test_default_open_leaves_v2_unchanged_and_refuses_activity(
    tmp_path: Path, inputs, result
) -> None:
    path = tmp_path / "v2.sqlite3"
    run_id = make_v2(path, inputs, result)
    before = schema_of(path)
    with Ledger(path) as ledger:
        assert ledger.get(run_id).status == "succeeded"
        assert ledger.submit(inputs).attempt_id == ledger.get(run_id).attempt_id
        with pytest.raises(SchemaError, match="migrate=True"):
            ledger.record_activity(ActivityKind.HYPOTHESIS, SUBJECT, {})
        with pytest.raises(SchemaError, match="migrate=True"):
            ledger.activities()
    assert schema_of(path) == before
    with (
        Ledger(path, read_only=True) as ledger,
        pytest.raises(SchemaError, match="migrate=True"),
    ):
        ledger.activities()
    assert schema_of(path) == before


def test_read_only_is_byte_identical_and_rejects_writes(
    tmp_path: Path, inputs, result
) -> None:
    v2 = tmp_path / "v2.sqlite3"
    run_id = make_v2(v2, inputs, result)
    v3 = tmp_path / "v3.sqlite3"
    with Ledger(v3) as ledger:
        hypothesis(ledger)
    for path in (v2, v3):
        before = digest(path)
        with Ledger(path, read_only=True) as ledger:
            assert ledger.query_all() == ledger.query_all()
            with pytest.raises(ReadOnlyError):
                ledger.submit(inputs, request_key="new")
            with pytest.raises(ReadOnlyError):
                ledger.fail(run_id, "no")
        assert digest(path) == before
    with Ledger(v3, read_only=True) as ledger:
        assert len(ledger.activities()) == 1
        with pytest.raises(ReadOnlyError):
            hypothesis(ledger)
    with pytest.raises(InvalidInputError):
        Ledger(v3, read_only=True, migrate=True)
    with pytest.raises(sqlite3.OperationalError):
        Ledger(tmp_path / "missing.sqlite3", read_only=True)
    assert not (tmp_path / "missing.sqlite3").exists()


def test_migrate_preserves_records_and_index_serves_subject_reads(
    tmp_path: Path, inputs, result
) -> None:
    path = tmp_path / "v2.sqlite3"
    run_id = make_v2(path, inputs, result)
    with Ledger(path) as ledger:
        original = ledger.get(run_id)
    with Ledger(path, migrate=True) as ledger:
        assert ledger.get(run_id) == original
        assert ledger.activities() == ()
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        plan = db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM activity WHERE subject_hash = ?",
            (SUBJECT,),
        ).fetchall()
        assert any("activity_subject" in row[3] for row in plan)
    fresh = tmp_path / "fresh.sqlite3"
    with Ledger(fresh):
        pass
    with sqlite3.connect(fresh) as db:
        assert db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'activity_subject'"
        ).fetchone()


def test_hypothesis_outcome_links_card_to_decision_once(ledger: Ledger) -> None:
    card = ledger.record_activity(ActivityKind.HYPOTHESIS, SUBJECT, {"card": "x"})
    decision = {"outcome": "verified", "decision_hash": "d" * 64}
    for bad in (
        {"outcome": "bogus", "decision_hash": "d" * 64},
        {"outcome": "verified"},
        {"outcome": "discarded", "decision_hash": "nope"},
    ):
        with pytest.raises(InvalidInputError):
            ledger.record_activity(
                ActivityKind.HYPOTHESIS_OUTCOME,
                SUBJECT,
                bad,
                parent_sequence=card.sequence,
            )
    with pytest.raises(InvalidTransitionError, match="hypothesis record"):
        ledger.record_activity(ActivityKind.HYPOTHESIS_OUTCOME, SUBJECT, decision)
    done = ledger.record_activity(
        ActivityKind.HYPOTHESIS_OUTCOME,
        SUBJECT,
        decision,
        key="outcome",
        parent_sequence=card.sequence,
    )
    assert (
        ledger.record_activity(
            ActivityKind.HYPOTHESIS_OUTCOME,
            SUBJECT,
            decision,
            key="outcome",
            parent_sequence=card.sequence,
        )
        == done
    )
    with pytest.raises(InvalidTransitionError, match="one recorded outcome"):
        ledger.record_activity(
            ActivityKind.HYPOTHESIS_OUTCOME,
            SUBJECT,
            {"outcome": "screen_dropped"},
            parent_sequence=card.sequence,
        )
    other = ledger.record_activity(ActivityKind.HYPOTHESIS, SUBJECT, {"card": "y"})
    ledger.record_activity(
        ActivityKind.HYPOTHESIS_OUTCOME,
        SUBJECT,
        {"outcome": "screen_dropped"},
        parent_sequence=other.sequence,
    )


def test_spend_settlement_references_one_reservation_once(ledger: Ledger) -> None:
    with pytest.raises(InvalidTransitionError, match="reservation"):
        ledger.record_activity(ActivityKind.SPEND_SETTLEMENT, SUBJECT, {})
    hold = ledger.record_activity(ActivityKind.SPEND_RESERVATION, SUBJECT, {"usd": 1})
    with pytest.raises(InvalidTransitionError, match="no parent"):
        ledger.record_activity(
            ActivityKind.SPEND_RESERVATION, SUBJECT, {}, parent_sequence=hold.sequence
        )
    note = ledger.record_activity(ActivityKind.HYPOTHESIS, SUBJECT, {})
    with pytest.raises(InvalidTransitionError, match="reservation"):
        ledger.record_activity(
            ActivityKind.SPEND_SETTLEMENT, SUBJECT, {}, parent_sequence=note.sequence
        )
    settled = ledger.record_activity(
        ActivityKind.SPEND_SETTLEMENT,
        SUBJECT,
        {"usd": 0.5},
        key="settle",
        parent_sequence=hold.sequence,
    )
    assert (
        ledger.record_activity(
            ActivityKind.SPEND_SETTLEMENT,
            SUBJECT,
            {"usd": 0.5},
            key="settle",
            parent_sequence=hold.sequence,
        )
        == settled
    )
    with pytest.raises(InvalidTransitionError, match="settled once"):
        ledger.record_activity(
            ActivityKind.SPEND_SETTLEMENT, SUBJECT, {}, parent_sequence=hold.sequence
        )


@pytest.mark.parametrize("committed", [False, True])
def test_sigkill_keeps_activity_admission_atomic(
    tmp_path: Path, committed: bool
) -> None:
    path = tmp_path / "crash.sqlite3"
    script = """
import signal, sys
import openloop.ledger.store as store
from openloop.ledger import Ledger, ActivityKind
with Ledger(sys.argv[1]) as ledger:
    ledger.record_activity(ActivityKind.HYPOTHESIS, 'a' * 64, {})
    if sys.argv[2] == 'False':
        append = store.append_activity
        def held(*args, **kwargs):
            result = append(*args, **kwargs)
            print('ready', flush=True)
            signal.pause()
            return result
        store.append_activity = held
    ledger.record_activity(ActivityKind.IMPLEMENTATION_STARTED, 'a' * 64, {})
    print('ready', flush=True)
    signal.pause()
"""
    root = Path(__file__).parents[2]
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(committed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": os.defpath, "PYTHONPATH": str(root / "src")},
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=10), "Child did not reach the transaction"
            assert process.stdout.readline() == b"ready\n"
        process.send_signal(signal.SIGKILL)
        process.communicate(timeout=5)
        with Ledger(path) as ledger:
            history = ledger.activities(subject_hash=SUBJECT)
            assert len(history) == (2 if committed else 1)
            if committed:
                with pytest.raises(InvalidTransitionError, match="unresolved"):
                    ledger.record_activity(
                        ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {}
                    )
            else:
                assert (
                    ledger.record_activity(
                        ActivityKind.IMPLEMENTATION_STARTED, SUBJECT, {}
                    ).payload["attempt"]
                    == 1
                )
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
