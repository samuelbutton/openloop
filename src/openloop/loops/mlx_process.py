"""Shared trusted MLX process boundary; the coordinator owns executor evidence."""

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path

from openloop.executors import (
    ArtifactRole,
    Executor,
    Limits,
    ProcessJob,
    Snapshot,
    State,
)
from openloop.ledger import JSONValue
from openloop.ledger.validation import freeze_object, plain

from .models import ContractError, JobOutput, RunContext

WALL_LIMIT_SECONDS = 3600
MEMORY_LIMIT_BYTES = 16 * 1024**3
WORKER_LOG_LIMIT_BYTES = 16 * 1024**2
LOG_TAIL_CHARACTERS = 4_000
EXECUTION_KEY = "execution"


def _tail(data: bytes) -> str:
    """Return a bounded end of a worker stream, so failure reasons stay small."""
    return data.decode("utf-8", errors="replace")[-LOG_TAIL_CHARACTERS:]


def _execution_evidence(snapshot: Snapshot) -> Mapping[str, JSONValue]:
    """Coordinator-observed executor values. Absent values stay absent."""
    evidence: dict[str, JSONValue] = {
        "job_id": snapshot.job_id,
        "state": str(snapshot.state),
        "logs": {
            artifact.path.name: {
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in snapshot.artifacts
            if artifact.role == ArtifactRole.LOG
        },
    }
    for name in ("exit_code", "queued_seconds", "elapsed_seconds", "peak_rss_bytes"):
        value = getattr(snapshot, name)
        if value is not None:
            evidence[name] = value
    return evidence


def _log_digests(snapshot: Snapshot) -> str:
    return ", ".join(
        f"{artifact.path.name} sha256={artifact.sha256}"
        for artifact in snapshot.artifacts
        if artifact.role == ArtifactRole.LOG
    )


async def run_worker(
    payload: bytes,
    python: Path,
    executor: Executor,
    context: RunContext,
    *,
    module: str,
) -> JobOutput:
    """Execute a reviewed worker, stop on cancellation, and attach observed evidence."""
    request = ProcessJob(
        (str(python), "-m", module),
        stdin=payload,
        environment={
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(__file__).parents[2]),
        },
        reviewed=True,
        gpu=True,
        limits=Limits(
            wall_seconds=WALL_LIMIT_SECONDS,
            memory_bytes=MEMORY_LIMIT_BYTES,
            output_bytes=WORKER_LOG_LIMIT_BYTES,
        ),
    )
    identifier = await executor.submit(request, key=context.attempt_id)
    try:
        snapshot = await executor.collect(identifier)
    except asyncio.CancelledError:
        # collect() is shielded, so a cancelled step must stop its own job;
        # otherwise training keeps running and holds the GPU lane.
        await executor.cancel(identifier)
        raise
    logs = {
        artifact.path.name: artifact.path.read_bytes()
        for artifact in snapshot.artifacts
        if artifact.role == ArtifactRole.LOG
    }
    stdout, stderr = logs.get("stdout", b""), logs.get("stderr", b"")
    if snapshot.state != State.SUCCEEDED:
        raise ContractError(
            f"Training worker failed with exit code {snapshot.exit_code} "
            f"({snapshot.state}): {snapshot.error or ''}"
            f"\nexecutor job {snapshot.job_id}; log digests: "
            f"{_log_digests(snapshot)}"
            f"\n--- stdout tail ---\n{_tail(stdout)}"
            f"\n--- stderr tail ---\n{_tail(stderr)}"
        )
    records = stdout.decode("utf-8").splitlines()
    if not records or not records[-1].startswith("OPENLOOP_RESULT "):
        raise ContractError("Worker omitted its trusted result record")
    output = plain(json.loads(records[-1].removeprefix("OPENLOOP_RESULT ")))
    if not isinstance(output, Mapping):
        raise ContractError("Worker result must be a JSON object")
    if EXECUTION_KEY in output:
        raise ContractError(f"Worker receipt must not contain {EXECUTION_KEY!r}")
    return freeze_object({**output, EXECUTION_KEY: _execution_evidence(snapshot)})
