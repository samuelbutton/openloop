"""Simulated replay of recorded S1 receipts on the mock executor. Read-only.

Boundary: this adapter reads receipt files under one explicit source directory
and writes one JSON report to an explicit path. It never changes receipts,
starts processes, or touches the ledger. Every number it emits is simulated:
outcomes are historical observations resampled from recordings, and time comes
from a virtual clock. Output is deterministic for the same receipts and flags.

    python -m openloop.studies.mock_replay_smoke --check
"""

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from openloop.executors import (
    ExecutorError,
    InvalidJobError,
    Limits,
    MockExecutor,
    ProcessJob,
    RecordedRun,
    State,
)
from openloop.ledger import Metric, Result, content_hash

DEFAULT_SOURCE = Path("research/s1/run-2026-10-11")
DEFAULT_OUTPUT = Path("docs/mock-replay-smoke.json")
DEFAULT_WORKERS = (1, 10, 100, 10_000)
DEFAULT_SEED = 0
DURATION_METRIC = "process_seconds"
SCHEMA_VERSION = 2
# Replay must never time out: the default 60 s wall limit would hide every outcome.
WALL_SECONDS = 86_400


def _result(raw: dict[str, dict[str, dict[str, object]]]) -> Result:
    def metrics(group: str) -> dict[str, Metric]:
        return {
            name: Metric(**fields)  # type: ignore[arg-type]  # JSON fields match Metric.
            for name, fields in raw.get(group, {}).items()
        }

    return Result(
        metrics(group="metrics"),
        dict(raw.get("artifacts", {})),  # type: ignore[arg-type]  # Digest strings.
        metrics(group="observations"),
    )


def load_recordings(source: Path) -> tuple[tuple[ProcessJob, RecordedRun], ...]:
    """Read every succeeded run-*.json receipt into a never-executed job and recording.

    Recordings are pooled by candidate (hash of its config) and fidelity, not
    seed. The duration is the T1 process_seconds observation, not the execution
    stage, which can include orchestration overhead.
    """
    pairs: list[tuple[ProcessJob, RecordedRun]] = []
    for path in sorted(source.glob("run-*.json")):
        data = path.read_bytes()
        run = json.loads(data)["run"]
        if run["status"] != "succeeded" or run.get("result") is None:
            raise InvalidJobError(f"{path.name} is not a succeeded run")
        result = _result(run["result"])
        metric = result.observations.get(DURATION_METRIC)
        if metric is None or metric.unit != "seconds" or metric.split != "execution":
            raise InvalidJobError(f"{path.name} lacks execution {DURATION_METRIC}")
        inputs = run["inputs"]
        pool = f"{content_hash(inputs['config'])[:16]}:{inputs['fidelity']}"
        job = ProcessJob(
            ("replay-only-not-executed", pool, path.name),
            limits=Limits(wall_seconds=WALL_SECONDS),
            gpu=True,
        )
        pairs.append(
            (
                job,
                RecordedRun(
                    job.hash,
                    hashlib.sha256(data).hexdigest(),
                    metric.value,
                    State.SUCCEEDED,
                    result=result,
                    pool_key=pool,
                ),
            )
        )
    if not pairs:
        raise InvalidJobError("No run-*.json receipts found")
    return tuple(pairs)


def _pool(job: ProcessJob) -> str:
    return job.argv[1]


async def _scenario(
    pairs: tuple[tuple[ProcessJob, RecordedRun], ...],
    *,
    workers: int,
    gpu_lanes: int,
    seed: int,
    strict: bool,
) -> dict[str, object]:
    executor = MockExecutor(
        tuple(recording for _, recording in pairs),
        workers=workers,
        gpu_lanes=gpu_lanes,
        seed=seed,
        strict=strict,
        pool_of=_pool,
    )
    try:
        pending = {await executor.submit(job, key=job.argv[2]) for job, _ in pairs}
        completed: list[object] = []
        while pending:
            receipt = await executor.wait_any(pending)
            pending.discard(receipt.job_id)
            completed.append(receipt)
        receipts = [await executor.poll(f"mock-{n}") for n in range(1, len(pairs) + 1)]
        return {
            "workers": workers,
            "gpu_lanes": gpu_lanes,
            "completed_replays": sum(r.state is State.SUCCEEDED for r in receipts),
            "simulated_elapsed_seconds": executor.now,
            "resampled_submissions": executor.resampled_submissions,
            "distinct_recordings_served": len({r.recording_hash for r in receipts}),
        }
    finally:
        await executor.close()


async def build_report(
    source: Path,
    *,
    workers: Sequence[int] = DEFAULT_WORKERS,
    seed: int = DEFAULT_SEED,
    strict: bool = False,
    gpu_lanes: int | None = None,
) -> dict[str, object]:
    """Replay every receipt once per worker count. gpu_lanes defaults to workers."""
    pairs = load_recordings(source)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.glob("run-*.json"))
    }
    combined = "".join(f"{name} {digest}\n" for name, digest in hashes.items())
    scenarios = [
        await _scenario(
            pairs,
            workers=count,
            gpu_lanes=count if gpu_lanes is None else gpu_lanes,
            seed=seed,
            strict=strict,
        )
        for count in workers
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "simulated": True,
        "generator": "python -m openloop.studies.mock_replay_smoke",
        "source_root": source.as_posix(),
        "source_receipts_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "source_files_sha256": hashes,
        "recipe": "legacy S1; not the shared starting recipe",
        "replay": {
            "mode": "strict" if strict else "resample",
            "seed": seed,
            "pool": "candidate config hash and fidelity; seed is not part of the pool",
            "duration_metric": DURATION_METRIC,
        },
        "recording_count": len(pairs),
        "pool_count": len({recording.pool for _, recording in pairs}),
        "total_recorded_process_seconds": sum(
            recording.duration_seconds for _, recording in pairs
        ),
        "scenarios": scenarios,
        "limits": (
            "Simulated. Identical workers; gpu_lanes per scenario as listed; no "
            "transfer, contention, preparation, or evaluation overhead. Resampling "
            "reuses recorded noise and cannot create variance beyond the recorded "
            "pool. Outcomes are historical observations, not new measurements."
        ),
    }


def render(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, nargs="+", default=list(DEFAULT_WORKERS))
    parser.add_argument("--replay-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--strict", action="store_true", help="refuse recording reuse")
    parser.add_argument("--gpu-lanes", type=int, help="default: one per worker")
    parser.add_argument(
        "--check", action="store_true", help="fail if --output differs; write nothing"
    )
    arguments = parser.parse_args(argv)
    try:
        text = render(
            asyncio.run(
                build_report(
                    arguments.source,
                    workers=arguments.workers,
                    seed=arguments.replay_seed,
                    strict=arguments.strict,
                    gpu_lanes=arguments.gpu_lanes,
                )
            )
        )
    except ExecutorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if arguments.check:
        if not arguments.output.exists() or arguments.output.read_text() != text:
            print(f"{arguments.output} differs from a fresh report", file=sys.stderr)
            return 1
        return 0
    arguments.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
