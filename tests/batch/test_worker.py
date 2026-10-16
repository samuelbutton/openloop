"""Exercise real reviewed CPU commands, return failures, and restart boundaries."""

import asyncio
import hashlib
import json
import signal
import subprocess
import sys
import time

import pytest

from openloop.batch.plan import Batch, BatchJob
from openloop.batch.receive import receive
from openloop.batch.transfer import BatchError, publish, sync_bundle
from openloop.batch.worker import drain
from openloop.executors import InvalidJobError
from openloop.ledger import Ledger, canonical_json

IMAGE = "sha256:" + "a" * 64


def job(name: str = "smoke", *, code: str | None = None) -> BatchJob:
    argv = (sys.executable, "-m", "openloop.batch.smoke")
    if code is not None:
        argv = (sys.executable, "-c", code)
    return BatchJob(name, argv, 15, 512 * 1024**2)


def test_drain_and_resume_preserve_rows_and_return_hashes(tmp_path):
    batch = Batch((job(), job("second")), IMAGE)
    state, returned = tmp_path / "state", tmp_path / "return"
    assert asyncio.run(drain(batch, state, returned)) == 0
    first = returned / batch.hash / "smoke"
    rows = (first / "ledger-rows.json").read_bytes()
    data = json.loads(rows)
    assert len(data["tables"]["submission"]) == 1
    assert (
        max(data["tables"]["event"], key=lambda event: event["sequence"])["kind"]
        == "succeeded"
    )
    manifest = json.loads((first / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    ledger_path = state / batch.hash / "smoke" / "ledger.sqlite3"
    before = ledger_path.read_bytes()
    assert asyncio.run(drain(batch, state, returned)) == 0
    assert ledger_path.read_bytes() == before
    assert (first / "ledger-rows.json").read_bytes() == rows
    accepted = receive(returned / batch.hash, tmp_path / "accepted")
    assert (accepted / "smoke" / "ledger-rows.json").read_bytes() == rows
    assert receive(returned / batch.hash, tmp_path / "accepted") == accepted


@pytest.mark.parametrize("code", ["raise SystemExit(2)", "print('no ledger')"])
def test_command_failure_drains_remaining_jobs(tmp_path, code):
    batch = Batch((job("broken", code=code), job("good")), IMAGE)
    assert asyncio.run(drain(batch, tmp_path / "state", tmp_path / "return")) == 1
    returned = tmp_path / "return" / batch.hash
    failed = json.loads((returned / "broken" / "receipt.json").read_text())
    assert not failed["succeeded"]
    assert failed["ledger_error"]
    assert (returned / "good" / "manifest.json").exists()
    assert not json.loads((returned / "drained.json").read_text())["succeeded"]


def test_partial_running_ledger_is_preserved(tmp_path):
    code = """
import asyncio, os
from openloop.batch.smoke import run
from openloop.ledger import Ledger
asyncio.run(run())
with Ledger(os.environ['OPENLOOP_LEDGER']) as ledger:
    previous = ledger.query_all()[0]
    ledger.submit(previous.inputs, purpose='replication')
"""
    batch = Batch((job(code=code),), IMAGE)
    assert asyncio.run(drain(batch, tmp_path / "state", tmp_path / "return")) == 1
    rows = json.loads(
        (tmp_path / "return" / batch.hash / "smoke" / "ledger-rows.json").read_text()
    )
    assert len(rows["tables"]["submission"]) == 2


def test_transfer_conflict_stops_admission_then_resume_only_syncs(tmp_path):
    batch = Batch((job(), job("second")), IMAGE)
    state, returned = tmp_path / "state", tmp_path / "return"
    conflict = returned / batch.hash / "smoke" / "ledger-rows.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("conflict")
    with pytest.raises(BatchError, match="conflicts"):
        asyncio.run(drain(batch, state, returned))
    assert not (state / batch.hash / "second").exists()
    source = state / batch.hash / "smoke" / "bundle" / "receipt.json"
    before = source.read_bytes()
    conflict.unlink()  # Repair only the test transport fixture, not ledger evidence.
    assert asyncio.run(drain(batch, state, returned)) == 0
    assert source.read_bytes() == before


def test_receiver_checks_hashes_before_publishing(tmp_path):
    batch = Batch((job(),), IMAGE)
    asyncio.run(drain(batch, tmp_path / "state", tmp_path / "return"))
    returned = tmp_path / "return" / batch.hash
    (returned / "smoke" / "ledger-rows.json").write_text("truncated")
    with pytest.raises(BatchError, match="hash mismatch"):
        receive(returned, tmp_path / "accepted")
    assert not (tmp_path / "accepted").exists()


def test_atomic_publish_and_bundle_replay(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    publish(source / "receipt.json", b"{}")
    sync_bundle(source, target)
    sync_bundle(source, target)
    with pytest.raises(BatchError, match="conflicts"):
        publish(target / "receipt.json", b"changed")
    assert (target / "receipt.json").read_bytes() == b"{}"
    (source / "link").symlink_to(source / "receipt.json")
    with pytest.raises(BatchError, match="symlinks"):
        sync_bundle(source, tmp_path / "unsafe")


@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM])
def test_worker_interruption_never_replays_a_started_job(tmp_path, signum):
    code = """
import asyncio, os, time
from pathlib import Path
from openloop.batch.smoke import run
from openloop.ledger import Ledger
asyncio.run(run())
with Ledger(os.environ['OPENLOOP_LEDGER']) as ledger:
    run = ledger.submit(ledger.query_all()[0].inputs, purpose='replication')
    ledger.start_stage(run.id, 'execution')
parent = os.getppid()
Path('ready').write_text(str(os.getpid()))
while os.getppid() == parent:
    time.sleep(.02)
"""
    batch = Batch((job(code=code), job("later")), IMAGE)
    state, returned = tmp_path / "state", tmp_path / "return"
    queue = tmp_path / "queue.json"
    queue.write_text(
        canonical_json({"version": 1, "image": batch.image, "jobs": batch.jobs})
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openloop.batch.worker",
            "--image-id",
            IMAGE,
            "--queue",
            str(queue),
            "--state",
            str(state),
            "--sync-to",
            str(returned),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    directory = state / batch.hash / "smoke"
    try:
        deadline = time.monotonic() + 10
        while not (directory / "ready").exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.02)
        # A second coordinator must not admit any job in the same batch.
        with pytest.raises(BatchError, match="already has a worker"):
            asyncio.run(drain(batch, state, returned))
        process.send_signal(signum)
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    with Ledger(directory / "ledger.sqlite3", read_only=True) as ledger:
        before = ledger.export_rows()
        assert ledger.query_all()[-1].status == "running"
    with pytest.raises(BatchError, match="Reconcile interrupted"):
        asyncio.run(drain(batch, state, returned))
    with Ledger(directory / "ledger.sqlite3", read_only=True) as ledger:
        assert ledger.export_rows() == before
    assert not (state / batch.hash / "later").exists()
    assert not (returned / batch.hash / "drained.json").exists()


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"version": True, "jobs": []},
        {"version": 1, "jobs": [{"name": "../bad"}]},
    ],
)
def test_invalid_queue_rejected_before_work(value):
    with pytest.raises(InvalidJobError):
        Batch.parse(value)


def test_queue_identity_and_validation():
    batch = Batch((job(),), IMAGE)
    assert Batch(batch.jobs, "sha256:" + "b" * 64).hash != batch.hash
    assert (
        Batch.parse(
            json.loads(
                canonical_json({"version": 1, "image": batch.image, "jobs": batch.jobs})
            )
        )
        == batch
    )
    with pytest.raises(InvalidJobError, match="Duplicate"):
        Batch((job(), job()), IMAGE)
    with pytest.raises(InvalidJobError):
        BatchJob("../escape", ("python",), 1, 1)
    with pytest.raises(InvalidJobError):
        BatchJob("bad", ("python",), float("inf"), 1)
