"""Plugin reports use ledger transactions; tests write fresh temporary DBs only."""

import os
import selectors
import signal
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.ledger import (
    SCHEMA_VERSION,
    ActivityKind,
    Derivation,
    Ledger,
    ReadOnlyError,
    SchemaError,
)


@given(st.dictionaries(st.text(min_size=1), st.integers(), max_size=10))
def test_report_hash_ignores_time_and_object_key_order(payload) -> None:
    first = Derivation("research_memory.v0", "a" * 64, payload, "before")
    second = replace(
        first, payload=dict(reversed(list(payload.items()))), timestamp="after"
    )
    assert first.hash == second.hash
    assert replace(first, subject_hash="b" * 64).hash != first.hash
    assert replace(first, namespace="different.v0").hash != first.hash


def test_report_publication_is_idempotent_and_append_only(tmp_path: Path) -> None:
    path = tmp_path / "reports.sqlite3"
    payload = {"runs": ["measured"]}
    with Ledger(path) as ledger:
        first = ledger.record_derivation("memory.v0", "a" * 64, payload)
        payload["runs"].append("changed")
        assert first.payload["runs"] == ("measured",)
        assert ledger.record_derivation("memory.v0", "a" * 64, first.payload) == first
        other = ledger.record_derivation("memory.v0", "b" * 64, payload)
        assert ledger.derivations("memory.v0") == (first, other)
        assert ledger.derivations("memory.v0", subject_hash="a" * 64) == (first,)
    with Ledger(path, read_only=True) as ledger:
        assert ledger.derivations("memory.v0") == (first, other)
        with pytest.raises(ReadOnlyError):
            ledger.record_derivation("memory.v0", "a" * 64, {})
    with sqlite3.connect(path) as db:
        for statement in (
            "DELETE FROM derivation",
            "UPDATE derivation SET payload='{}'",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute(statement)


@pytest.mark.parametrize("version", [2, 3])
def test_derivations_require_explicit_migration_and_preserve_runs(
    tmp_path: Path, inputs, result, version: int
) -> None:
    path = tmp_path / "old.sqlite3"
    with Ledger(path) as ledger:
        run = ledger.submit(inputs)
        ledger.start_stage(run.id, "execution")
        ledger.start_stage(run.id, "evaluation")
        original = ledger.complete(run.id, result)
        activity = ledger.record_activity(ActivityKind.HYPOTHESIS, "a" * 64, {})
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE derivation")
        if version == 2:
            db.execute("DROP TABLE activity")
        db.execute(f"PRAGMA user_version = {version}")
        db.execute("PRAGMA journal_mode = DELETE")
    before = path.read_bytes()
    for read_only in (False, True):
        with Ledger(path, read_only=read_only) as ledger:
            assert ledger.get(run.id) == original
            with pytest.raises(SchemaError, match="migrate=True"):
                ledger.derivations("memory.v0")
            with pytest.raises(SchemaError, match="migrate=True"):
                ledger.record_derivation("memory.v0", "a" * 64, {})
        assert path.read_bytes() == before
    with Ledger(path, migrate=True) as ledger:
        assert ledger.get(run.id) == original
        assert ledger.submit(inputs).attempt_id == original.attempt_id
        assert ledger.activities() == (() if version == 2 else (activity,))
        assert ledger.derivations("memory.v0") == ()
        ledger.record_derivation("memory.v0", "a" * 64, {})
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_concurrent_duplicate_reports_publish_once(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite3"
    with Ledger(path):
        pass

    def publish(_: int) -> Derivation:
        with Ledger(path) as ledger:
            return ledger.record_derivation("memory.v0", "a" * 64, {"refs": [1, 2]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(publish, range(2))
    assert first == second
    with Ledger(path) as ledger:
        assert ledger.derivations("memory.v0") == (first,)


@pytest.mark.parametrize("committed", [False, True])
def test_sigkill_leaves_one_complete_report_or_none(
    tmp_path: Path, committed: bool
) -> None:
    path = tmp_path / "crash.sqlite3"
    script = """
import signal, sys
import openloop.ledger.store as store
from openloop.ledger import Ledger
with Ledger(sys.argv[1]) as ledger:
    if sys.argv[2] == 'False':
        append = store.append_derivation
        def held(*args, **kwargs):
            result = append(*args, **kwargs)
            print('ready', flush=True)
            signal.pause()
            return result
        store.append_derivation = held
    ledger.record_derivation('memory.v0', 'a' * 64, {'refs': [1, 2]})
    print('ready', flush=True)
    signal.pause()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(committed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": os.defpath, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
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
            records = ledger.derivations("memory.v0")
            assert len(records) == int(committed)
            complete = ledger.record_derivation("memory.v0", "a" * 64, {"refs": [1, 2]})
            assert ledger.derivations("memory.v0") == (complete,)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
