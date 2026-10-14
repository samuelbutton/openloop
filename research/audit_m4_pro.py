"""Prepare or explicitly execute S1 part 2: the M4 Pro public-claims audit.

Default mode reads pinned upstream bytes, checks the data cache and worker
interpreter, compiles every transformed program, and writes a frozen plan. It
imports no MLX and starts no training.

With --execute-approved and a NEW --output directory, it runs the frozen 30
local GPU jobs through LocalExecutor and records each in a new ledger. A memory
guard samples system pressure during every job and stops the whole study if it
trips. No upstream file, cache file, or existing evidence changes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from openloop.executors import (
    ExecutorError,
    Limits,
    LocalExecutor,
    ProcessJob,
    Snapshot,
    State,
)
from openloop.ledger import (
    ConflictError,
    ExperimentInputs,
    Ledger,
    Metric,
    Purpose,
    Reproducibility,
    Result,
    Stage,
    content_hash,
)
from openloop.ledger.validation import plain
from openloop.studies.claims_audit import (
    MEMORY_SAMPLE_SECONDS,
    ClaimedOutcome,
    Comparison,
    GuardTrip,
    MemoryGuard,
    MemoryReading,
    Observation,
    analyze_audit,
    inject_seed,
    interleaved_order,
    memory_guard_policy,
    parse_summary,
    substitute_once,
)

REPO = Path(__file__).resolve().parents[1]
FORK_URL = "https://github.com/elementalcollision/autoresearch"
WIKI_REVISION = "3337ce96ae8c5410fa79088a4273804d7f33bdd8"
WIKI_SHA256 = "c25ecb0521459beb1095a1b74ae88db5622ed8680edfc9110676ffe0275e8061"
FILES = ("train_mlx.py", "prepare.py")
BACKEND_FILES = ("backends/__init__.py", "backends/muon_mlx.py", "backends/muon_mps.py")
SEEDS = (2000, 2001, 2002, 2003, 2004)
SHUFFLE_SEED = 20261013
WALL_SECONDS = 1800
STUDY_WALL_SECONDS = 12 * 3600
MEMORY_BYTES = 23 * 1024**3
MAX_CONSECUTIVE_FAILURES = 3
PACKAGES = ("mlx", "mlx-metal", "numpy", "pyarrow", "regex", "requests", "rustbpe")
PACKAGES += ("tiktoken",)
MANIFEST = REPO / "research/artifacts/autoresearch-mlx/data_manifest.json"

# The amendment of October 13, 2026 dropped the baseline arm (see PROTOCOL.md).
# Its commit is kept only so the history check still verifies the lineage.
DROPPED_BASELINE_COMMIT = "5d903d0027c39cb37b7dd9d9b70151c64756262a"

# Arms: name -> (commit or None, published bpb, published status, description,
# reconstruction). The four discarded experiment commits are not in the public
# repository, so they are rebuilt from 1d83776 with one documented constant edit.
INCUMBENT = "1d83776"
ARMS: dict[str, dict[str, object]] = {
    "bc5cdc1": {
        "commit": "bc5cdc17092555a96a4a3dfd6eb985c512d67018",
        "bpb": 1.439990,
        "status": "keep",
        "log": "#1 MLP 1.5x + batch 16K/8 + warmdown 0.7",
    },
    "1d83776": {
        "commit": "1d8377683d9a5b1af14f5933e6f5130c54e94460",
        "bpb": 1.429396,
        "status": "keep",
        "log": "#4 Batch 8K/4",
    },
    "27b9d9f": {
        "bpb": 1.429607,
        "status": "discard (tied)",
        "log": "#7 MLP_RATIO=1.0",
        "edit": ("MLP_RATIO = 1.5", "MLP_RATIO = 1.0"),
    },
    "000e04d": {
        "bpb": 1.430230,
        "status": "discard (tied)",
        "log": "#11 Softcap 12",
        "edit": ("softcap = 15", "softcap = 12"),
    },
    "8f00506": {
        "bpb": 1.432641,
        "status": "discard (slightly worse)",
        "log": "#9b WARMDOWN_RATIO=0.65",
        "edit": ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.65"),
    },
    "cc51630": {
        "bpb": 1.432399,
        "status": "discard (slightly worse)",
        "log": "#10a Window SL",
        "edit": ('WINDOW_PATTERN = "SSSL"', 'WINDOW_PATTERN = "SL"'),
    },
}
COMPARISONS = (
    ("1d83776", "bc5cdc1", ClaimedOutcome.WIN),
    ("27b9d9f", INCUMBENT, ClaimedOutcome.NOT_WIN),
    ("000e04d", INCUMBENT, ClaimedOutcome.NOT_WIN),
    ("8f00506", INCUMBENT, ClaimedOutcome.NOT_WIN),
    ("cc51630", INCUMBENT, ClaimedOutcome.NOT_WIN),
)
RUN_COUNT = len(ARMS) * len(SEEDS)
HELD_OUT = {
    "status": "requires_owner_approval_not_prepared",
    "shard": "shard_05000.parquet",
    "url": (
        "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/"
        "resolve/main/shard_05000.parquet"
    ),
    "size_bytes": 91420038,
    "size_source": "HTTP HEAD x-linked-size, 2026-10-12; nothing downloaded",
}


def save(path: Path, value: object) -> None:
    """Write one new evidence record, refusing all overwrites."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(source), *args])


def arm_files(source: Path, name: str) -> dict[str, bytes]:
    """Return the exact bytes an arm runs: git objects, then any documented edit."""
    arm = ARMS[name]
    commit = str(arm.get("commit") or ARMS[INCUMBENT]["commit"])
    files = {path: git(source, "show", f"{commit}:{path}") for path in FILES}
    for path in BACKEND_FILES:
        files[path] = git(source, "show", f"{commit}:{path}")
    edit = arm.get("edit")
    if isinstance(edit, tuple):
        old, new = edit
        text = substitute_once(files["train_mlx.py"].decode(), old, new)
        files["train_mlx.py"] = text.encode()
    return files


def history(source: Path) -> list[dict[str, object]]:
    """Check what the fork's history shows; record every discrepancy."""
    checks: list[dict[str, object]] = []
    for name in ARMS:
        commit = ARMS[name].get("commit")
        probe = str(commit or name)
        found = (
            subprocess.run(
                ["git", "-C", str(source), "cat-file", "-e", f"{probe}^{{commit}}"],
                capture_output=True,
            ).returncode
            == 0
        )
        checks.append({"arm": name, "ref": probe, "commit_exists": found})
        if commit and not found:
            raise ConflictError(f"Pinned commit is missing: {name}")
    parent = git(source, "rev-parse", f"{ARMS['bc5cdc1']['commit']}^").decode().strip()
    checks.append({"fact": "parent of bc5cdc1", "value": parent})
    checks.append(
        {
            "fact": "dropped baseline commit is that parent",
            "value": parent == DROPPED_BASELINE_COMMIT,
        }
    )
    parent = git(source, "rev-parse", f"{ARMS[INCUMBENT]['commit']}^").decode().strip()
    checks.append(
        {
            "fact": "parent of 1d83776 is bc5cdc1",
            "value": parent == ARMS["bc5cdc1"]["commit"],
        }
    )
    return checks


def verify_cache() -> list[dict[str, object]]:
    cache = Path.home() / ".cache/autoresearch"
    manifest = json.loads(MANIFEST.read_text())
    expected = {
        Path(item["path"]).name
        for item in manifest
        if item["path"].endswith(".parquet")
    }
    if {p.name for p in (cache / "data").glob("*.parquet")} != expected:
        raise ConflictError(
            "Cached shard membership differs from the recorded snapshot"
        )
    for item in manifest:
        if digest(cache / item["path"]) != item["sha256"]:
            raise ConflictError(f"Cached data changed: {item['path']}")
    return manifest


def probe_runtime(python: Path) -> dict[str, object]:
    code = (
        "import json,platform,sys\nfrom importlib.metadata import version\n"
        f"print(json.dumps({{'python':platform.python_version(),'machine':platform.machine(),"
        f"'packages':{{n:version(n) for n in {PACKAGES!r}}}}}))"
    )
    return json.loads(
        subprocess.check_output([str(python), "-I", "-c", code], text=True)
    )


def hardware() -> dict[str, object]:
    display = json.loads(
        subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
    )["SPDisplaysDataType"][0]
    return {
        "chip": display["sppci_model"],
        "gpu_cores": int(display["sppci_cores"]),
        "ram_bytes": int(
            subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True)
        ),
    }


def preflight(source: Path, python: Path) -> dict[str, object]:
    """Build the frozen plan. Starts no training and imports no MLX."""
    checks = history(source)
    arms: dict[str, object] = {}
    reference: dict[str, str] | None = None
    for name, spec in ARMS.items():
        files = arm_files(source, name)
        hashes = {path: sha(data) for path, data in files.items()}
        shared = {p: h for p, h in hashes.items() if p != "train_mlx.py"}
        reference = reference or shared
        if shared != reference:
            raise ConflictError(
                f"{name} differs from the other arms outside train_mlx.py"
            )
        program = files["train_mlx.py"].decode()
        if "TIME_BUDGET = 300" not in files["prepare.py"].decode():
            raise ConflictError("Time budget is not 300 seconds")
        for seed in SEEDS:
            compile(inject_seed(program, seed), f"{name}/train_mlx.py", "exec")
        arms[name] = {
            "commit": spec.get("commit"),
            "source": "git commit" if spec.get("commit") else "reconstructed",
            "reconstruction": (
                {"base_commit": ARMS[INCUMBENT]["commit"], "replace": list(edit)}
                if isinstance(edit := spec.get("edit"), tuple)
                else None
            ),
            "published_val_bpb": spec["bpb"],
            "published_status": spec["status"],
            "published_log_entry": spec["log"],
            "file_sha256": hashes,
        }
    comparisons = [
        {
            "child": c,
            "parent": p,
            "claimed": claim.value,
            "published_child_bpb": ARMS[c]["bpb"],
            "published_parent_bpb": ARMS[p]["bpb"],
        }
        for c, p, claim in COMPARISONS
    ]
    order = interleaved_order(list(ARMS), SEEDS, SHUFFLE_SEED)
    frozen = {
        "study": "S1 part 2: M4 Pro (24 GB) Mar 14, 2026 lineage replication",
        "repository": FORK_URL,
        "wiki_revision": WIKI_REVISION,
        "wiki_page_sha256": WIKI_SHA256,
        "history_checks": checks,
        "arms": arms,
        "comparisons": comparisons,
        "seeds": list(SEEDS),
        "seed_transform": (
            "one exact replacement: mx.random.seed(42) -> mx.random.seed(N)"
        ),
        "shuffle_seed": SHUFFLE_SEED,
        "order": [
            {"index": i + 1, "arm": a, "seed": s} for i, (a, s) in enumerate(order)
        ],
        "limits": {
            "wall_seconds_per_job": WALL_SECONDS,
            "study_wall_seconds": STUDY_WALL_SECONDS,
            "memory_bytes": MEMORY_BYTES,
            "memory_sample_seconds": 5,
            "memory_guard": memory_guard_policy(),
            "gpu_lanes": 1,
            "max_consecutive_failures": MAX_CONSECUTIVE_FAILURES,
        },
        "environment": {
            **hardware(),
            "runtime": probe_runtime(python),
            "cache_manifest_sha256": digest(MANIFEST),
            "cache_files_verified": len(verify_cache()),
        },
        "code": {
            "audit_script_sha256": digest(Path(__file__)),
            "analysis_module_sha256": digest(
                REPO / "src/openloop/studies/claims_audit.py"
            ),
        },
        "estimated_runtime": {
            "per_run_seconds": 390,
            "basis": "October 10 reproduction: 377 s total script time, "
            "plus ~13 s executor and import overhead",
            "expected_hours": round(RUN_COUNT * 390 / 3600, 1),
            "cap_hours": STUDY_WALL_SECONDS // 3600,
        },
        "phase_b_held_out": HELD_OUT,
    }
    return {"plan_hash": content_hash(frozen), "plan": frozen}


def receipt(snap: Snapshot, output: Path) -> dict[str, object]:
    return {
        "job_id": snap.job_id,
        "input_hash": snap.input_hash,
        "state": snap.state,
        "exit_code": snap.exit_code,
        "started_at": snap.started_at.isoformat() if snap.started_at else None,
        "finished_at": snap.finished_at.isoformat() if snap.finished_at else None,
        "elapsed_seconds": snap.elapsed_seconds,
        "peak_rss_bytes": snap.peak_rss_bytes,
        "error": snap.error,
        "artifacts": [
            {
                "path": str(a.path.relative_to(output)),
                "sha256": a.sha256,
                "size_bytes": a.size_bytes,
            }
            for a in snap.artifacts
        ],
    }


def materialize(files: dict[str, bytes], seed: int, directory: Path) -> str:
    """Write a run directory; only train_mlx.py differs, by the seed call."""
    program = inject_seed(files["train_mlx.py"].decode(), seed).encode()
    for path, data in {**files, "train_mlx.py": program}.items():
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return sha(program)


def summarize(report_dir: Path, plan: dict[str, object]) -> None:
    observations = []
    for path in sorted(report_dir.glob("run-*.json")):
        record = json.loads(path.read_text())
        if record.get("summary"):
            summary = record["summary"]
            observations.append(
                Observation(
                    record["arm"],
                    record["seed"],
                    summary["val_bpb"],
                    summary["num_steps"],
                )
            )
    comparisons = [
        Comparison(
            f"{c['child']} vs {c['parent']}",
            c["child"],
            c["parent"],
            ClaimedOutcome(c["claimed"]),
            c["published_child_bpb"],
            c["published_parent_bpb"],
        )
        for c in plan["comparisons"]  # type: ignore[attr-defined]
    ]
    save(report_dir / "report.json", analyze_audit(comparisons, observations))


class MemoryGuardTripped(ConflictError):
    """The memory guard stopped the study. It is never retried automatically."""


type PressureReader = Callable[[], MemoryReading]


def read_memory() -> MemoryReading:
    """Read macOS pressure level and system-wide free percentage (about 10 ms)."""
    level = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
    )
    report = subprocess.check_output(["/usr/bin/memory_pressure"], text=True)
    marker = "System-wide memory free percentage:"
    line = next(line for line in report.splitlines() if line.startswith(marker))
    return MemoryReading(
        datetime.now(UTC).isoformat(),
        int(level.strip()),
        int(line.removeprefix(marker).strip().removesuffix("%")),
    )


async def run_guarded(
    executor: LocalExecutor,
    job: ProcessJob,
    *,
    key: str,
    read: PressureReader,
    interval: float,
) -> tuple[Snapshot, GuardTrip | None]:
    """Run one job, sampling memory every interval seconds until it finishes.

    On a trip the job is cancelled through the executor, which kills its
    process group. A reader that fails also trips: an unobserved machine is not
    known to be safe.
    """
    guard = MemoryGuard()
    job_id = await executor.submit(job, key=key)
    collector = asyncio.ensure_future(executor.collect(job_id))
    try:
        while True:
            try:
                trip = guard.observe(await asyncio.to_thread(read))
            except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
                trip = GuardTrip("memory pressure reader failed", ())
            if trip is not None:
                return await executor.cancel(job_id), trip
            done, _ = await asyncio.wait({collector}, timeout=interval)
            if done:
                return collector.result(), None
    finally:
        collector.cancel()


def stop_for_memory(
    output: Path,
    receipt_path: Path,
    ledger: Ledger,
    run_id: str,
    record: dict[str, object],
    trip: GuardTrip,
) -> NoReturn:
    """Record the tripped run as failed, write the stop record, and stop the study."""
    ledger.fail(run_id, f"memory guard: {trip.reason}")
    record["failure"] = f"memory guard: {trip.reason}"
    save(receipt_path, record)
    save(
        output / "memory-guard-stop.json",
        {
            "observed_at": datetime.now(UTC).isoformat(),
            "reason": trip.reason,
            "run": {
                "index": record["index"],
                "arm": record["arm"],
                "seed": record["seed"],
                "ledger_run_id": run_id,
            },
            "readings": trip.readings,
            "threshold": memory_guard_policy(),
            "automatic_retry": False,
        },
    )
    raise MemoryGuardTripped(f"Memory guard tripped: {trip.reason}")


async def execute(
    source: Path,
    python: Path,
    prepared: dict[str, object],
    output: Path,
    *,
    read: PressureReader = read_memory,
) -> None:
    plan = prepared["plan"]
    plan_hash = str(prepared["plan_hash"])
    manifest = json.loads(MANIFEST.read_text())
    env = plan["environment"]  # type: ignore[index]
    read()  # Refuse to start unguarded.
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", prepared)
    save(
        output / "started.json",
        {"started_at": datetime.now(UTC).isoformat(), "macos": platform.mac_ver()[0]},
    )
    executor = LocalExecutor(output / "executor", memory_sample_seconds=5)
    results = 0
    failures = 0
    try:
        with Ledger(output / "ledger.sqlite3") as ledger:
            async with asyncio.timeout(STUDY_WALL_SECONDS):
                for item in plan["order"]:  # type: ignore[index]
                    index, arm, seed = item["index"], item["arm"], item["seed"]
                    print(f"S1p2 {index}/{RUN_COUNT}: {arm} seed {seed}", flush=True)
                    files = arm_files(source, arm)
                    hashes = plan["arms"][arm]["file_sha256"]  # type: ignore[index]
                    if {p: sha(d) for p, d in files.items()} != hashes:
                        raise ConflictError(f"Source changed for {arm}")
                    directory = output / "workdirs" / f"{index:02d}-{arm}-s{seed}"
                    program_hash = materialize(files, seed, directory)
                    inputs = ExperimentInputs(
                        source_hash=content_hash(hashes),
                        dependencies_hash=content_hash(env["runtime"]),
                        data_hash=content_hash(manifest),
                        environment_hash=content_hash(
                            {k: env[k] for k in ("chip", "gpu_cores", "ram_bytes")}
                        ),
                        evaluator_hash=content_hash(
                            {"prepare.py": hashes["prepare.py"]}
                        ),
                        loop_hash=content_hash(
                            {"study": "s1-part2", "plan": plan_hash}
                        ),
                        seed=seed,
                        fidelity="300s",
                        budget_unit="training_seconds",
                        budget_amount=300,
                        reproducibility=Reproducibility.NOISY,
                        config={"arm": arm, "commit": plan["arms"][arm]["commit"]},  # type: ignore[index]
                        execution={"program_sha256": program_hash, "slot": index},
                    )
                    run = ledger.submit(
                        inputs,
                        hypothesis=f"S1p2 replication of {arm} (published "
                        f"{ARMS[arm]['bpb']} BPB) on this machine",
                        request_key=f"s1p2:{plan_hash}:{arm}:{seed}",
                        purpose=Purpose.REPLICATION,
                    )
                    ledger.start_stage(run.id, Stage.EXECUTION)
                    snapshot, trip = await run_guarded(
                        executor,
                        ProcessJob(
                            (str(python), "-u", "train_mlx.py"),
                            cwd=directory,
                            environment={
                                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                                "HOME": str(Path.home()),
                            },
                            reviewed=True,
                            gpu=True,
                            limits=Limits(
                                wall_seconds=WALL_SECONDS,
                                memory_bytes=MEMORY_BYTES,
                                output_bytes=16 * 1024**2,
                            ),
                        ),
                        key=f"s1p2:{plan_hash}:{arm}:{seed}",
                        read=read,
                        interval=MEMORY_SAMPLE_SECONDS,
                    )
                    record: dict[str, object] = {
                        "index": index,
                        "arm": arm,
                        "seed": seed,
                        "program_sha256": program_hash,
                        "ledger_run_id": run.id,
                        "execution": receipt(snapshot, output),
                        "summary": None,
                    }
                    if trip is not None:
                        stop_for_memory(
                            output,
                            output / f"run-{index:02d}-{arm}-s{seed}.json",
                            ledger,
                            run.id,
                            record,
                            trip,
                        )
                    try:
                        if snapshot.state != State.SUCCEEDED:
                            raise ExecutorError(f"{snapshot.state}: {snapshot.error}")
                        stdout = next(
                            a.path
                            for a in snapshot.artifacts
                            if a.path.name == "stdout"
                        )
                        text = stdout.read_text()
                        summary = parse_summary(text)
                        header = [
                            line
                            for line in text.splitlines()
                            if line.startswith(
                                ("Model config", "Gradient accumulation", "Time budget")
                            )
                        ]
                        ledger.start_stage(run.id, Stage.EVALUATION)
                        ledger.complete(
                            run.id,
                            Result(
                                metrics={
                                    "val_bpb": Metric(
                                        float(summary["val_bpb"]), "bits/byte"
                                    )
                                },
                                observations={
                                    "num_steps": Metric(
                                        int(summary["num_steps"]), "steps"
                                    ),
                                    "training_seconds": Metric(
                                        float(summary["training_seconds"]), "s"
                                    ),
                                },
                                artifacts={
                                    a.path.name: a.sha256 for a in snapshot.artifacts
                                },
                            ),
                        )
                        record["summary"] = summary
                        record["header_lines"] = header
                        results += 1
                        failures = 0
                    except (
                        ExecutorError,
                        KeyError,
                        StopIteration,
                        ValueError,
                    ) as error:
                        ledger.fail(run.id, f"{type(error).__name__}: {error}")
                        record["failure"] = f"{type(error).__name__}: {error}"
                        failures += 1
                    save(output / f"run-{index:02d}-{arm}-s{seed}.json", record)
                    verify_cache()
                    if any(
                        digest(directory / p) != h
                        for p, h in hashes.items()
                        if p != "train_mlx.py"
                    ):
                        raise ConflictError("Run inputs changed during the study")
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        raise ConflictError("Stopped after consecutive failed runs")
            summarize(output, plan)
    except BaseException as error:
        # A job still running stays running in the ledger until reconciled.
        save(
            output / "interrupted.json",
            {
                "observed_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "reason": str(error),
                "successful_runs": results,
            },
        )
        raise
    finally:
        await executor.close()
    save(output / "finished.json", {"finished_at": datetime.now(UTC).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="fork clone")
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--summarize", type=Path, help="analyse a finished output")
    args = parser.parse_args()
    if args.summarize:
        plan = json.loads((args.summarize / "plan.json").read_text())["plan"]
        summarize(args.summarize, plan)
        return
    source, python = args.source.resolve(), args.worker_python.absolute()
    prepared = preflight(source, python)
    if not args.execute_approved:
        if args.plan is None:
            parser.error("preparing requires --plan")
        save(
            args.plan,
            {
                **prepared,
                "prepared_at": datetime.now(UTC).isoformat(),
                "training_runs": RUN_COUNT,
                "native_training_executed": False,
            },
        )
        print(
            f"Prepared {RUN_COUNT} runs, plan hash {prepared['plan_hash']}. "
            "No training executed."
        )
        return
    if args.output is None or args.plan is None:
        parser.error("--execute-approved requires --plan and a new --output directory")
    frozen = json.loads(args.plan.read_text(encoding="utf-8"))
    if frozen["plan_hash"] != prepared["plan_hash"] or frozen["plan"] != plain(
        prepared["plan"]
    ):
        raise ConflictError("Prepared plan or environment changed; prepare again")
    asyncio.run(execute(source, python, prepared, args.output.resolve()))


if __name__ == "__main__":
    main()
