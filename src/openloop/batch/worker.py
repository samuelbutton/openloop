"""Trusted batch coordinator. Reviewed commands share its container trust boundary.

No shell or inherited secrets. Container policy supplies isolation and disables
networking. LocalExecutor supplies process-group cancellation and resource limits.
A started job without a receipt is ambiguous and requires human reconciliation.
"""

import argparse
import asyncio
import fcntl
import json
import os
import signal
import sqlite3
import sys
from pathlib import Path

from openloop.executors import LocalExecutor, ProcessJob, State
from openloop.ledger import LedgerError, canonical_json

from .plan import Batch, BatchJob
from .transfer import BatchError, export_ledger, publish, sync_bundle


async def _execute(job: BatchJob, directory: Path) -> bool:
    bundle = directory / "bundle"
    bundle.mkdir()
    executor = LocalExecutor(directory / "executor")
    ended = False
    ledger_error: str | None = None
    try:
        process = ProcessJob(
            job.argv,
            cwd=directory,
            limits=job.limits,
            reviewed=True,
            environment={
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
                "HOME": str(directory),
                "TMPDIR": str(directory),
                "OPENLOOP_LEDGER": str(directory / "ledger.sqlite3"),
                "OPENLOOP_OUTPUT": str(directory),
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(Path(__file__).parents[2]),
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
        )
        job_id = await executor.submit(process, key=job.name)
        try:
            snapshot = await executor.collect(job_id)
        except asyncio.CancelledError:
            snapshot = await executor.cancel(job_id)
            raise
    finally:
        await executor.close()
        # The executor has joined the child before the read-only export.
        for path in (directory / "executor").rglob("*"):
            if path.is_file() and not path.is_symlink():
                publish(
                    bundle / "executor" / path.relative_to(directory / "executor"),
                    path.read_bytes(),
                )
        ledger_path = directory / "ledger.sqlite3"
        try:
            ended = export_ledger(ledger_path, bundle / "ledger-rows.json")
        except (BatchError, LedgerError, sqlite3.Error) as error:
            ledger_error = str(error)
            publish(
                bundle / "ledger-error.json",
                canonical_json({"error": ledger_error}).encode(),
            )
    succeeded = snapshot.state is State.SUCCEEDED and ended
    publish(
        bundle / "receipt.json",
        canonical_json(
            {
                "job": job,
                "execution": {
                    "job_id": snapshot.job_id,
                    "input_hash": snapshot.input_hash,
                    "state": snapshot.state,
                    "exit_code": snapshot.exit_code,
                    "submitted_at": snapshot.submitted_at.isoformat(),
                    "started_at": snapshot.started_at.isoformat()
                    if snapshot.started_at
                    else None,
                    "finished_at": snapshot.finished_at.isoformat()
                    if snapshot.finished_at
                    else None,
                    "elapsed_seconds": snapshot.elapsed_seconds,
                    "queued_seconds": snapshot.queued_seconds,
                    "peak_rss_bytes": snapshot.peak_rss_bytes,
                    "error": snapshot.error,
                },
                "complete_ledger": ended,
                "ledger_error": ledger_error,
                "succeeded": succeeded,
            }
        ).encode(),
    )
    return succeeded


async def drain(batch: Batch, state: Path, destination: Path) -> int:
    """Drain a fixed queue sequentially and sync each receipt before proceeding.

    Exit 0 means all commands succeeded and all ledger attempts ended. Exit 1
    means at least one command failed. Transfer errors stop admission. Interrupted
    jobs are never automatically re-executed or marked as failed ledger samples.
    """
    root = state.resolve() / batch.hash
    target = destination.resolve() / batch.hash
    if root == target or root in target.parents or target in root.parents:
        raise BatchError("State and return directories must be separate")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BatchError("This batch already has a worker") from None
        publish(
            root / "queue.json",
            canonical_json(
                {"version": 1, "image": batch.image, "jobs": batch.jobs}
            ).encode(),
        )
        publish(target / "queue.json", (root / "queue.json").read_bytes())
        failed = False
        for job in batch.jobs:
            directory = root / job.name
            bundle = directory / "bundle"
            receipt = bundle / "receipt.json"
            if receipt.exists():
                sync_bundle(bundle, target / job.name)
                failed |= not json.loads(receipt.read_text())["succeeded"]
                continue
            if directory.exists():
                # Preserve any partial evidence, including a running ledger attempt.
                ledger = directory / "ledger.sqlite3"
                if ledger.exists() and not (bundle / "ledger-rows.json").exists():
                    export_ledger(ledger, bundle / "ledger-rows.json")
                if bundle.exists():
                    sync_bundle(bundle, target / job.name / "interrupted")
                raise BatchError(
                    f"Reconcile interrupted job before resuming: {directory}"
                )
            directory.mkdir()
            publish(directory / "started.json", canonical_json({"job": job}).encode())
            try:
                failed |= not await _execute(job, directory)
            finally:
                if bundle.exists():
                    # Cancellation/error evidence is distinct from a finished bundle.
                    returned = target / job.name
                    if not receipt.exists():
                        returned /= "interrupted"
                    sync_bundle(bundle, returned)
        publish(
            target / "drained.json",
            canonical_json(
                {
                    "batch_hash": batch.hash,
                    "succeeded": not failed,
                    "jobs": len(batch.jobs),
                }
            ).encode(),
        )
        return int(failed)


async def _main(args: argparse.Namespace) -> int:
    batch = Batch.parse(json.loads(args.queue.read_text()))
    if args.image_id != batch.image:
        raise BatchError("Queue image does not match the launcher image ID")
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, task.cancel)
    try:
        return await drain(batch, args.state, args.sync_to)
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)


def main() -> None:
    """Run a reviewed queue; terminate the container once the batch is drained."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--sync-to", type=Path, required=True)
    args = parser.parse_args()
    try:
        code = asyncio.run(_main(args))
    except asyncio.CancelledError:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
