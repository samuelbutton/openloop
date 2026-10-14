"""Prepare or explicitly execute S1 part 2 follow-ups (Amendment 2, October 13, 2026).

Slot A re-executes the one environment-failed run of the October 13 audit
(arm 8f00506, seed 2004) under the identical frozen protocol. Phase B runs the
published winner 1d83776 and its parent bc5cdc1 at fresh seeds, each with an
evaluation-only addition that also scores a held-out ClimbMix shard.

Default mode writes a frozen plan and starts no training. With
--execute-approved and a NEW --output directory it runs the 11 local GPU jobs
through LocalExecutor behind the same memory guard as the audit. Other modes
only read finished evidence and write new report files. No existing evidence,
upstream file, cache file or shard changes.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from openloop.executors import ExecutorError, Limits, LocalExecutor, ProcessJob, State
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
    analyze_audit,
    inject_seed,
    interleaved_order,
    memory_guard_policy,
    parse_summary,
)
from openloop.studies.heldout import (
    HELDOUT_DIR_ENV,
    HELDOUT_EVAL,
    HELDOUT_KEY,
    HELDOUT_SHARD,
    analyze_heldout,
    combined_observations,
    heldout_run_observations,
    inject_heldout_eval,
    parse_heldout_bpb,
)

REPO = Path(__file__).resolve().parents[1]
_audit_spec = importlib.util.spec_from_file_location(
    "audit_m4_pro", Path(__file__).with_name("audit_m4_pro.py")
)
assert _audit_spec is not None and _audit_spec.loader is not None
audit: ModuleType = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)

PARENT_PLAN = REPO / "research/s1-part2/plan-2026-10-13.json"
PARENT_RUN = REPO / "research/s1-part2/run-2026-10-13"
FAILED_RECEIPT = PARENT_RUN / "run-30-8f00506-s2004.json"
HELDOUT_DIR = REPO / "data/climbmix-heldout"
HELDOUT_MANIFEST = HELDOUT_DIR / "manifest.json"
KIND_COMPLETION = "seed_completion"
KIND_HELDOUT = "heldout"
COMPLETION_ARM, COMPLETION_SEED = "8f00506", 2004
CHILD, PARENT = "1d83776", "bc5cdc1"
HELDOUT_SEEDS = (3000, 3001, 3002, 3003, 3004)
SHUFFLE_SEED = 20261014
WALL_SECONDS = 1800
STUDY_WALL_SECONDS = 4 * 3600
MEMORY_BYTES = 23 * 1024**3
MAX_CONSECUTIVE_FAILURES = 3
JOB_COUNT = 1 + 2 * len(HELDOUT_SEEDS)
SEED_COMPLETION_REPORT = "report-seed-completion.json"
HELDOUT_REPORT = "heldout-report.json"


def verify_heldout_shard(expected_sha256: str, expected_size: int) -> None:
    """Fail unless the held-out shard is exactly the recorded file."""
    shard = HELDOUT_DIR / HELDOUT_SHARD
    if shard.stat().st_size != expected_size or audit.digest(shard) != expected_sha256:
        raise ConflictError("Held-out shard changed")


def job_program(files: dict[str, bytes], kind: str, seed: int) -> tuple[str, str]:
    """Return the seeded program text and the text a job actually runs."""
    seeded = inject_seed(files["train_mlx.py"].decode(), seed)
    return seeded, inject_heldout_eval(seeded) if kind == KIND_HELDOUT else seeded


def write_job_files(
    files: dict[str, bytes], kind: str, seed: int, directory: Path
) -> tuple[str, str]:
    """Write a run directory; return the base (seeded) and final program hashes."""
    seeded, final = job_program(files, kind, seed)
    for path, data in {**files, "train_mlx.py": final.encode()}.items():
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return audit.sha(seeded.encode()), audit.sha(final.encode())


def preflight(source: Path, python: Path) -> dict[str, object]:
    """Build the frozen plan. Starts no training and imports no MLX."""
    parent = json.loads(PARENT_PLAN.read_text(encoding="utf-8"))
    failed = json.loads(FAILED_RECEIPT.read_text(encoding="utf-8"))
    manifest = json.loads(HELDOUT_MANIFEST.read_text(encoding="utf-8"))
    audit.verify_cache()
    verify_heldout_shard(manifest["sha256"], manifest["size_bytes"])
    shard_name = manifest["file"]
    if shard_name != HELDOUT_SHARD or shard_name in {
        Path(item["path"]).name for item in json.loads(audit.MANIFEST.read_text())
    }:
        raise ConflictError("Held-out shard must not be a training or validation shard")
    environment = {
        **audit.hardware(),
        "runtime": audit.probe_runtime(python),
        "cache_manifest_sha256": audit.digest(audit.MANIFEST),
    }
    previous = parent["plan"]["environment"]
    if any(environment[k] != previous[k] for k in environment):
        raise ConflictError(
            "Environment differs from the October 13 plan; "
            "the protocol is not identical"
        )
    arms: dict[str, object] = {}
    files_by_arm: dict[str, dict[str, bytes]] = {}
    for name in (CHILD, PARENT, COMPLETION_ARM):
        files_by_arm[name] = audit.arm_files(source, name)
        hashes = {p: audit.sha(d) for p, d in files_by_arm[name].items()}
        if hashes != parent["plan"]["arms"][name]["file_sha256"]:
            raise ConflictError(f"{name} sources differ from the October 13 plan")
        arms[name] = {"file_sha256": hashes}
    jobs: list[dict[str, object]] = []
    order = [(KIND_COMPLETION, COMPLETION_ARM, COMPLETION_SEED)] + [
        (KIND_HELDOUT, arm, seed)
        for arm, seed in interleaved_order([CHILD, PARENT], HELDOUT_SEEDS, SHUFFLE_SEED)
    ]
    for index, (kind, arm, seed) in enumerate(order, start=1):
        base, final = (
            audit.sha(text.encode())
            for text in job_program(files_by_arm[arm], kind, seed)
        )
        jobs.append(
            {
                "index": index,
                "kind": kind,
                "arm": arm,
                "seed": seed,
                "base_program_sha256": base,
                "program_sha256": final,
            }
        )
    if jobs[0]["base_program_sha256"] != failed["program_sha256"]:
        raise ConflictError(
            "Re-execution program differs from the failed run's program"
        )
    frozen = {
        "study": "S1 part 2, Amendment 2: seed completion and held-out phase",
        "parent_plan": {
            "path": str(PARENT_PLAN.relative_to(REPO)),
            "plan_hash": parent["plan_hash"],
            "file_sha256": audit.digest(PARENT_PLAN),
        },
        "seed_completion": {
            "arm": COMPLETION_ARM,
            "seed": COMPLETION_SEED,
            "reexecution_of": {
                "receipt": str(FAILED_RECEIPT.relative_to(REPO)),
                "receipt_sha256": audit.digest(FAILED_RECEIPT),
                "program_sha256": failed["program_sha256"],
                "failure": failed["failure"],
                "elapsed_seconds": failed["execution"]["elapsed_seconds"],
                "reason": "Python install under /tmp was purged by macOS; "
                "environment failure, not an experiment result",
            },
            "transform": "none; identical to the October 13 protocol",
        },
        "heldout_phase": {
            "claim": f"{CHILD} improves on {PARENT}",
            "child": CHILD,
            "parent": PARENT,
            "seeds": list(HELDOUT_SEEDS),
            "confidence": 0.95,
            "shard": {
                "file": shard_name,
                "url": manifest["url"],
                "size_bytes": manifest["size_bytes"],
                "sha256": manifest["sha256"],
                "manifest_sha256": audit.digest(HELDOUT_MANIFEST),
                "directory_env": HELDOUT_DIR_ENV,
            },
            "transform": {
                "anchor": "the published final evaluate_bpb line, exactly once",
                "appended_code": HELDOUT_EVAL,
                "appended_code_sha256": audit.sha(HELDOUT_EVAL.encode()),
            },
        },
        "arms": arms,
        "shuffle_seed": SHUFFLE_SEED,
        "order": jobs,
        "limits": {
            "wall_seconds_per_job": WALL_SECONDS,
            "study_wall_seconds": STUDY_WALL_SECONDS,
            "memory_bytes": MEMORY_BYTES,
            "memory_sample_seconds": 5,
            "memory_guard": memory_guard_policy(),
            "gpu_lanes": 1,
            "max_consecutive_failures": MAX_CONSECUTIVE_FAILURES,
        },
        "environment": environment,
        "code": {
            "audit_heldout_script_sha256": audit.digest(Path(__file__)),
            "heldout_module_sha256": audit.digest(
                REPO / "src/openloop/studies/heldout.py"
            ),
            "audit_script_sha256": audit.digest(Path(audit.__file__)),
            "analysis_module_sha256": audit.digest(
                REPO / "src/openloop/studies/claims_audit.py"
            ),
        },
        "estimated_runtime": {
            "per_run_seconds": 420,
            "basis": "October 13 runs took about 390 s each; the extra evaluation "
            "of 5,242,880 tokens is assumed to add under 30 s",
            "expected_hours": round(JOB_COUNT * 420 / 3600, 1),
            "cap_hours": STUDY_WALL_SECONDS // 3600,
        },
    }
    return {"plan_hash": content_hash(frozen), "plan": frozen}


def load_receipts(directory: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("run-*.json"))
    ]


def write_reports(output: Path, plan: dict[str, object]) -> None:
    """Write the two new analysis files. Neither touches any October 13 file."""
    receipts = load_receipts(output)
    parent = json.loads((PARENT_RUN / "plan.json").read_text(encoding="utf-8"))["plan"]
    comparisons = [
        Comparison(
            f"{c['child']} vs {c['parent']}",
            c["child"],
            c["parent"],
            ClaimedOutcome(c["claimed"]),
            c["published_child_bpb"],
            c["published_parent_bpb"],
        )
        for c in parent["comparisons"]
    ]
    completion = [r for r in receipts if r.get("kind") == KIND_COMPLETION]
    audit.save(
        output / SEED_COMPLETION_REPORT,
        analyze_audit(
            comparisons,
            combined_observations(load_receipts(PARENT_RUN), completion),
        ),
    )
    audit.save(
        output / HELDOUT_REPORT,
        analyze_heldout(heldout_run_observations(receipts), child=CHILD, parent=PARENT),
    )


async def execute(
    source: Path,
    python: Path,
    prepared: dict[str, object],
    output: Path,
    *,
    read: audit.PressureReader = audit.read_memory,
) -> None:
    plan = prepared["plan"]
    plan_hash = str(prepared["plan_hash"])
    manifest = json.loads(audit.MANIFEST.read_text())
    shard = plan["heldout_phase"]["shard"]  # type: ignore[index]
    env = plan["environment"]  # type: ignore[index]
    read()  # Refuse to start unguarded.
    output.mkdir(parents=True, exist_ok=False)
    audit.save(output / "plan.json", prepared)
    audit.save(
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
                    index, kind = item["index"], item["kind"]
                    arm, seed = item["arm"], item["seed"]
                    held = kind == KIND_HELDOUT
                    print(
                        f"S1p2-A2 {index}/{JOB_COUNT}: {kind} {arm} seed {seed}",
                        flush=True,
                    )
                    files = audit.arm_files(source, arm)
                    hashes = plan["arms"][arm]["file_sha256"]  # type: ignore[index]
                    if {p: audit.sha(d) for p, d in files.items()} != hashes:
                        raise ConflictError(f"Source changed for {arm}")
                    directory = (
                        output / "workdirs" / f"{index:02d}-{kind}-{arm}-s{seed}"
                    )
                    base_hash, program_hash = write_job_files(
                        files, kind, seed, directory
                    )
                    if (base_hash, program_hash) != (
                        item["base_program_sha256"],
                        item["program_sha256"],
                    ):
                        raise ConflictError(f"Program changed for {arm} seed {seed}")
                    key = f"s1p2a2:{plan_hash}:{kind}:{arm}:{seed}"
                    data = {"train+val": manifest}
                    evaluator = {"prepare.py": hashes["prepare.py"]}
                    if held:
                        data["heldout"] = shard["sha256"]
                        evaluator["transform"] = plan["heldout_phase"]["transform"][  # type: ignore[index]
                            "appended_code_sha256"
                        ]
                    run = ledger.submit(
                        ExperimentInputs(
                            source_hash=content_hash(
                                {**hashes, "program": program_hash}
                            ),
                            dependencies_hash=content_hash(env["runtime"]),
                            data_hash=content_hash(data),
                            environment_hash=content_hash(
                                {k: env[k] for k in ("chip", "gpu_cores", "ram_bytes")}
                            ),
                            evaluator_hash=content_hash(evaluator),
                            loop_hash=content_hash(
                                {"study": "s1-part2-amendment-2", "plan": plan_hash}
                            ),
                            seed=seed,
                            fidelity="300s",
                            budget_unit="training_seconds",
                            budget_amount=300,
                            reproducibility=Reproducibility.NOISY,
                            config={"arm": arm, "kind": kind},
                            execution={"program_sha256": program_hash, "slot": index},
                        ),
                        hypothesis=f"S1p2 Amendment 2 {kind} of {arm} on this machine",
                        request_key=key,
                        purpose=Purpose.REPLICATION,
                    )
                    ledger.start_stage(run.id, Stage.EXECUTION)
                    environment = {
                        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                        "HOME": str(Path.home()),
                    }
                    if held:
                        environment[HELDOUT_DIR_ENV] = str(HELDOUT_DIR)
                    snapshot, trip = await audit.run_guarded(
                        executor,
                        ProcessJob(
                            (str(python), "-u", "train_mlx.py"),
                            cwd=directory,
                            environment=environment,
                            reviewed=True,
                            gpu=True,
                            limits=Limits(
                                wall_seconds=WALL_SECONDS,
                                memory_bytes=MEMORY_BYTES,
                                output_bytes=16 * 1024**2,
                            ),
                        ),
                        key=key,
                        read=read,
                        interval=MEMORY_SAMPLE_SECONDS,
                    )
                    record: dict[str, object] = {
                        "index": index,
                        "kind": kind,
                        "arm": arm,
                        "seed": seed,
                        "base_program_sha256": base_hash,
                        "program_sha256": program_hash,
                        "ledger_run_id": run.id,
                        "execution": audit.receipt(snapshot, output),
                        "summary": None,
                    }
                    if kind == KIND_COMPLETION:
                        record["reexecution_of"] = plan["seed_completion"][
                            "reexecution_of"
                        ][  # type: ignore[index]
                            "receipt"
                        ]
                    name = f"run-{index:02d}-{kind}-{arm}-s{seed}.json"
                    if trip is not None:
                        audit.stop_for_memory(
                            output, output / name, ledger, run.id, record, trip
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
                        if held:
                            summary[HELDOUT_KEY] = parse_heldout_bpb(text)
                        elif HELDOUT_KEY in text:
                            raise ExecutorError("Unexpected held-out output")
                        ledger.start_stage(run.id, Stage.EVALUATION)
                        metrics = {
                            "val_bpb": Metric(float(summary["val_bpb"]), "bits/byte")
                        }
                        if held:
                            metrics[HELDOUT_KEY] = Metric(
                                float(summary[HELDOUT_KEY]), "bits/byte"
                            )
                        ledger.complete(
                            run.id,
                            Result(
                                metrics=metrics,
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
                        results += 1
                        failures = 0
                    except (
                        ExecutorError,
                        KeyError,
                        StopIteration,
                        ValueError,
                    ) as error:
                        # InvalidInputError is a ValueError subclass in this package.
                        ledger.fail(run.id, f"{type(error).__name__}: {error}")
                        record["failure"] = f"{type(error).__name__}: {error}"
                        failures += 1
                    audit.save(output / name, record)
                    audit.verify_cache()
                    verify_heldout_shard(shard["sha256"], shard["size_bytes"])
                    if any(
                        audit.digest(directory / p) != h
                        for p, h in hashes.items()
                        if p != "train_mlx.py"
                    ):
                        raise ConflictError("Run inputs changed during the study")
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        raise ConflictError("Stopped after consecutive failed runs")
            write_reports(output, plan)  # type: ignore[arg-type]
    except BaseException as error:
        audit.save(
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
    audit.save(output / "finished.json", {"finished_at": datetime.now(UTC).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="fork clone")
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        help="write new report files for a finished or interrupted output",
    )
    args = parser.parse_args()
    if args.report:
        plan = json.loads((args.report / "plan.json").read_text())["plan"]
        write_reports(args.report.resolve(), plan)
        return
    if args.source is None or args.worker_python is None:
        parser.error("--source and --worker-python are required")
    source, python = args.source.resolve(), args.worker_python.absolute()
    prepared = preflight(source, python)
    if not args.execute_approved:
        if args.plan is None:
            parser.error("preparing requires --plan")
        audit.save(
            args.plan,
            {
                **prepared,
                "prepared_at": datetime.now(UTC).isoformat(),
                "training_runs": JOB_COUNT,
                "native_training_executed": False,
            },
        )
        print(
            f"Prepared {JOB_COUNT} runs, plan hash {prepared['plan_hash']}. "
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
