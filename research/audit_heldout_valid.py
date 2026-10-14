"""Prepare or execute the Amendment 3 re-run of the Amendment 2 jobs.

Same 11 slots, seeds, order, programs and analysis as `audit_heldout.py`
(imported unchanged). Adds a sleep guard, a step-count validity check and
automatic re-execution of invalid attempts for the same slot. Default mode
writes a frozen plan and starts no training. With --execute-approved and a NEW
--output directory it runs the jobs through LocalExecutor behind the same
memory guard. No existing evidence changes.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

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
    analyze_audit,
    parse_summary,
)
from openloop.studies.heldout import (
    HELDOUT_DIR_ENV,
    HELDOUT_KEY,
    analyze_heldout,
    combined_observations,
    heldout_run_observations,
    parse_heldout_bpb,
)
from openloop.studies.validity import (
    Attempt,
    DriftTrip,
    SleepDriftMonitor,
    reference_steps,
    run_slot,
    settle_wait,
    step_floor,
    steps_invalid_reason,
    valid_receipts,
    validity_policy,
)

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audit_heldout", Path(__file__).with_name("audit_heldout.py")
)
assert _spec is not None and _spec.loader is not None
base: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
audit: ModuleType = base.audit

AMENDMENT_2_PLAN = REPO / "research/s1-part2/plan-heldout-2026-10-13.json"
INVALIDATED_RUN = REPO / "research/s1-part2/run-heldout-2026-10-13"
STUDY_WALL_SECONDS = 8 * 3600
ATTEMPTS_REPORT = "attempts-report.json"


def preflight(source: Path, python: Path) -> dict[str, object]:
    """Build the frozen plan: the Amendment 2 plan plus the validity rules."""
    prepared = base.preflight(source, python)
    plan = dict(prepared["plan"])  # type: ignore[call-overload]
    amendment_2 = json.loads(AMENDMENT_2_PLAN.read_text(encoding="utf-8"))
    if amendment_2["plan"]["order"] != plan["order"]:
        raise ConflictError("Slots differ from the Amendment 2 plan")
    counts: dict[str, list[int]] = {}
    for path in sorted(base.PARENT_RUN.glob("run-*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            item["arm"] in (base.CHILD, base.PARENT, base.COMPLETION_ARM)
            and item["summary"]
        ):
            counts.setdefault(item["arm"], []).append(int(item["summary"]["num_steps"]))
    medians = reference_steps(counts)
    policy = validity_policy()
    plan["study"] = "S1 part 2, Amendment 3: valid re-run of the Amendment 2 jobs"
    plan["amendment_2"] = {
        "plan_path": str(AMENDMENT_2_PLAN.relative_to(REPO)),
        "plan_hash": amendment_2["plan_hash"],
        "file_sha256": audit.digest(AMENDMENT_2_PLAN),
        "status": "invalidated by an environment event; no receipt is used",
        "invalidated_files": {
            path.name: audit.digest(path)
            for path in sorted(INVALIDATED_RUN.glob("*.json"))
            if path.name != "plan.json"
        },
    }
    plan["validity"] = {
        **policy,
        "rule": "An attempt is invalid when cumulative wall-minus-monotonic drift "
        "exceeds drift_limit_seconds, or when num_steps is below step_fraction "
        "times the arm's reference median. Scores never decide validity.",
        "reference_steps": {
            arm: {
                "run_2026_10_13_steps": sorted(counts[arm]),
                "median": medians[arm],
                "floor": step_floor(medians[arm]),
            }
            for arm in sorted(medians)
        },
    }
    plan["limits"] = {**plan["limits"], "study_wall_seconds": STUDY_WALL_SECONDS}  # type: ignore[call-overload]
    plan["code"] = {
        **plan["code"],  # type: ignore[call-overload]
        "audit_heldout_valid_script_sha256": audit.digest(Path(__file__)),
        "validity_module_sha256": audit.digest(
            REPO / "src/openloop/studies/validity.py"
        ),
    }
    plan["estimated_runtime"] = {
        **plan["estimated_runtime"],  # type: ignore[call-overload]
        "cap_hours": STUDY_WALL_SECONDS // 3600,
        "note": "Re-executions of invalid attempts add up to 2 more attempts per slot.",
    }
    return {"plan_hash": content_hash(plan), "plan": plan}


class Watched(NamedTuple):
    """Outcome of one watched job: the snapshot and any guard that tripped."""

    snapshot: Snapshot
    memory_trip: GuardTrip | None
    drift_trip: DriftTrip | None
    drift_seconds: float


async def run_watched(
    executor: LocalExecutor,
    job: ProcessJob,
    *,
    key: str,
    read: audit.PressureReader,
    interval: float,
    wall: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
) -> Watched:
    """Run one job, sampling memory and sleep drift every interval seconds.

    The memory guard is `MemoryGuard`, unchanged, and a failing reader trips it.
    Either trip cancels the job through the executor, which kills its process
    group. A final drift sample after completion catches sleep in the last
    interval.
    """
    guard = MemoryGuard()
    drift = SleepDriftMonitor(wall(), monotonic())
    job_id = await executor.submit(job, key=key)
    collector = asyncio.ensure_future(executor.collect(job_id))
    try:
        while True:
            try:
                memory_trip = guard.observe(await asyncio.to_thread(read))
            except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
                memory_trip = GuardTrip("memory pressure reader failed", ())
            if memory_trip is not None:
                snapshot = await executor.cancel(job_id)
                return Watched(snapshot, memory_trip, None, drift.total)
            drift_trip = drift.observe(wall(), monotonic())
            if drift_trip is not None:
                snapshot = await executor.cancel(job_id)
                return Watched(snapshot, None, drift_trip, drift.total)
            done, _ = await asyncio.wait({collector}, timeout=interval)
            if done:
                drift_trip = drift.observe(wall(), monotonic())
                return Watched(collector.result(), None, drift_trip, drift.total)
    finally:
        collector.cancel()


def load_receipts(directory: Path) -> list[dict[str, object]]:
    return base.load_receipts(directory)


def write_reports(output: Path, plan: dict[str, object]) -> None:
    """Write new analysis files from valid attempts only."""
    receipts = load_receipts(output)
    valid = valid_receipts(receipts)
    parent = json.loads((base.PARENT_RUN / "plan.json").read_text(encoding="utf-8"))[
        "plan"
    ]
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
    completion = [r for r in valid if r.get("kind") == base.KIND_COMPLETION]
    audit.save(
        output / base.SEED_COMPLETION_REPORT,
        analyze_audit(
            comparisons,
            combined_observations(load_receipts(base.PARENT_RUN), completion),
        ),
    )
    audit.save(
        output / base.HELDOUT_REPORT,
        analyze_heldout(
            heldout_run_observations(valid), child=base.CHILD, parent=base.PARENT
        ),
    )
    audit.save(
        output / ATTEMPTS_REPORT,
        {
            "attempts": [
                {
                    "slot": r.get("index"),
                    "arm": r.get("arm"),
                    "seed": r.get("seed"),
                    "attempt": r.get("attempt"),
                    "validity": r.get("validity"),
                }
                for r in receipts
            ],
            "excluded_invalid": sum(1 for r in receipts if r not in valid),
        },
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
    reference = plan["validity"]["reference_steps"]  # type: ignore[index]
    cap = int(plan["validity"]["max_attempts_per_slot"])  # type: ignore[index]
    total = len(plan["order"])  # type: ignore[arg-type]
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
                    held = kind == base.KIND_HELDOUT
                    files = audit.arm_files(source, arm)
                    hashes = plan["arms"][arm]["file_sha256"]  # type: ignore[index]

                    async def attempt_once(
                        number: int,
                        index: int = index,
                        kind: str = kind,
                        arm: str = arm,
                        seed: int = seed,
                        held: bool = held,
                        files: dict[str, bytes] = files,
                        hashes: dict[str, str] = hashes,
                        item: dict[str, object] = item,
                    ) -> Attempt:
                        nonlocal results, failures
                        print(
                            f"S1p2-A3 {index}/{total}: {kind} {arm} seed {seed} "
                            f"attempt {number}/{cap}",
                            flush=True,
                        )
                        if {p: audit.sha(d) for p, d in files.items()} != hashes:
                            raise ConflictError(f"Source changed for {arm}")
                        directory = (
                            output
                            / "workdirs"
                            / f"{index:02d}-{kind}-{arm}-s{seed}-a{number}"
                        )
                        base_hash, program_hash = base.write_job_files(
                            files, kind, seed, directory
                        )
                        if (base_hash, program_hash) != (
                            item["base_program_sha256"],
                            item["program_sha256"],
                        ):
                            raise ConflictError(
                                f"Program changed for {arm} seed {seed}"
                            )
                        key = f"s1p2a3:{plan_hash}:{kind}:{arm}:{seed}:a{number}"
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
                                    {
                                        k: env[k]
                                        for k in ("chip", "gpu_cores", "ram_bytes")
                                    }
                                ),
                                evaluator_hash=content_hash(evaluator),
                                loop_hash=content_hash(
                                    {"study": "s1-part2-amendment-3", "plan": plan_hash}
                                ),
                                seed=seed,
                                fidelity="300s",
                                budget_unit="training_seconds",
                                budget_amount=300,
                                reproducibility=Reproducibility.NOISY,
                                config={"arm": arm, "kind": kind},
                                execution={
                                    "program_sha256": program_hash,
                                    "slot": index,
                                    "attempt": number,
                                },
                            ),
                            hypothesis=(
                                f"S1p2 Amendment 3 {kind} of {arm} on this machine"
                            ),
                            request_key=key,
                            purpose=Purpose.REPLICATION,
                        )
                        ledger.start_stage(run.id, Stage.EXECUTION)
                        environment = {
                            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                            "HOME": str(Path.home()),
                        }
                        if held:
                            environment[HELDOUT_DIR_ENV] = str(base.HELDOUT_DIR)
                        watched = await run_watched(
                            executor,
                            ProcessJob(
                                (str(python), "-u", "train_mlx.py"),
                                cwd=directory,
                                environment=environment,
                                reviewed=True,
                                gpu=True,
                                limits=Limits(
                                    wall_seconds=base.WALL_SECONDS,
                                    memory_bytes=base.MEMORY_BYTES,
                                    output_bytes=16 * 1024**2,
                                ),
                            ),
                            key=key,
                            read=read,
                            interval=MEMORY_SAMPLE_SECONDS,
                        )
                        snapshot = watched.snapshot
                        record: dict[str, object] = {
                            "index": index,
                            "kind": kind,
                            "arm": arm,
                            "seed": seed,
                            "attempt": number,
                            "base_program_sha256": base_hash,
                            "program_sha256": program_hash,
                            "ledger_run_id": run.id,
                            "execution": audit.receipt(snapshot, output),
                            "summary": None,
                            "validity": {
                                "valid": True,
                                "drift_seconds": watched.drift_seconds,
                            },
                        }
                        if kind == base.KIND_COMPLETION:
                            record["reexecution_of"] = plan["seed_completion"][  # type: ignore[index]
                                "reexecution_of"
                            ]["receipt"]
                        name = f"run-{index:02d}-{kind}-{arm}-s{seed}-a{number}.json"
                        if watched.memory_trip is not None:
                            audit.stop_for_memory(
                                output,
                                output / name,
                                ledger,
                                run.id,
                                record,
                                watched.memory_trip,
                            )
                        verdict = Attempt(True)
                        if watched.drift_trip is not None:
                            reason = watched.drift_trip.reason
                            verdict = Attempt(False, reason)
                            ledger.fail(run.id, reason)
                            record["failure"] = reason
                        else:
                            try:
                                if snapshot.state != State.SUCCEEDED:
                                    raise ExecutorError(
                                        f"{snapshot.state}: {snapshot.error}"
                                    )
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
                                reason = steps_invalid_reason(
                                    int(summary["num_steps"]), reference[arm]["floor"]
                                )
                                if reason is not None:
                                    verdict = Attempt(False, reason)
                                    ledger.fail(run.id, reason)
                                    record["failure"] = reason
                                    record["observed_summary"] = summary
                                else:
                                    ledger.start_stage(run.id, Stage.EVALUATION)
                                    metrics = {
                                        "val_bpb": Metric(
                                            float(summary["val_bpb"]), "bits/byte"
                                        )
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
                                                    float(summary["training_seconds"]),
                                                    "s",
                                                ),
                                            },
                                            artifacts={
                                                a.path.name: a.sha256
                                                for a in snapshot.artifacts
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
                                # InvalidInputError is a ValueError subclass.
                                ledger.fail(run.id, f"{type(error).__name__}: {error}")
                                record["failure"] = f"{type(error).__name__}: {error}"
                                failures += 1
                        if not verdict.valid:
                            record["validity"] = {
                                "valid": False,
                                "reason": verdict.reason,
                                "drift_seconds": watched.drift_seconds,
                            }
                        audit.save(output / name, record)
                        audit.verify_cache()
                        base.verify_heldout_shard(shard["sha256"], shard["size_bytes"])
                        if any(
                            audit.digest(directory / p) != h
                            for p, h in hashes.items()
                            if p != "train_mlx.py"
                        ):
                            raise ConflictError("Run inputs changed during the study")
                        if failures >= base.MAX_CONSECUTIVE_FAILURES:
                            raise ConflictError("Stopped after consecutive failed runs")
                        return verdict

                    async def settle() -> None:
                        await asyncio.to_thread(
                            settle_wait,
                            wall=time.time,
                            monotonic=time.monotonic,
                            sleep=time.sleep,
                        )

                    await run_slot(attempt_once, settle, max_attempts=cap)
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
                "training_runs": base.JOB_COUNT,
                "native_training_executed": False,
            },
        )
        print(
            f"Prepared {base.JOB_COUNT} runs, plan hash {prepared['plan_hash']}. "
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
