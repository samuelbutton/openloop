"""Prepare or execute the Amendment 4 resume of the Amendment 3 run.

The Amendment 3 run stopped on the memory guard in slot 5 of 11. Amendment 4
keeps its valid receipts (slots 1 to 4), runs the remaining slots in a NEW
output directory with a 10 percent free-memory floor, and writes combined
reports. `audit_heldout_valid.py` and every older module are imported
unchanged, so earlier plans still reproduce. Default mode writes a frozen plan
and starts no training. With --execute-approved and a NEW --output directory
it runs only the missing slots through LocalExecutor. No existing evidence
changes; a retained slot is never run again.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import platform
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import NoReturn

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
    GuardTrip,
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
from openloop.studies.resume import (
    FloorMemoryGuard,
    ResumeMismatchError,
    Retained,
    memory_guard_policy,
    missing_slots,
    select_retained,
)
from openloop.studies.validity import (
    DriftTrip,
    SleepDriftMonitor,
    settle_wait,
    steps_invalid_reason,
    valid_receipts,
)

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audit_heldout_valid", Path(__file__).with_name("audit_heldout_valid.py")
)
assert _spec is not None and _spec.loader is not None
valid: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(valid)
base: ModuleType = valid.base
audit: ModuleType = base.audit

PRIOR_PLAN = REPO / "research/s1-part2/plan-heldout-2026-10-13b.json"
STUDY_WALL_SECONDS = valid.STUDY_WALL_SECONDS
ATTEMPTS_REPORT = valid.ATTEMPTS_REPORT
SOURCE_KEY = "source_directory"
STOP_FILES = ("memory-guard-stop.json", "interrupted.json")


def display(path: Path) -> str:
    """A path relative to the repository when inside it, else as given."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def load_receipt_files(directory: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("run-*.json"))
    }


def why_not_retained(receipt: Mapping[str, object]) -> str:
    validity = receipt.get("validity")
    if isinstance(validity, Mapping) and validity.get("valid") is not True:
        return f"invalid attempt: {validity.get('reason')}"
    if "failure" in receipt:
        return f"failed attempt: {receipt['failure']}"
    return "no summary"


def verify_retained_evidence(
    resume_from: Path, retained: Sequence[Retained], floors: Mapping[str, float]
) -> list[dict[str, object]]:
    """Check each kept receipt against its stdout artifact; return plan entries."""
    entries: list[dict[str, object]] = []
    for item in retained:
        receipt = item.receipt
        summary = receipt["summary"]
        assert isinstance(summary, Mapping)
        execution = receipt["execution"]
        assert isinstance(execution, Mapping)
        stdout = None
        for artifact in execution["artifacts"]:
            path = resume_from / artifact["path"]
            if not path.is_file() or audit.digest(path) != artifact["sha256"]:
                raise ResumeMismatchError(
                    f"{item.file}: artifact {artifact['path']} changed or is missing"
                )
            if path.name == "stdout":
                stdout = path
        if stdout is None:
            raise ResumeMismatchError(f"{item.file}: no stdout artifact")
        text = stdout.read_text()
        reparsed = parse_summary(text)
        if receipt["kind"] == base.KIND_HELDOUT:
            reparsed[HELDOUT_KEY] = parse_heldout_bpb(text)
        if reparsed != dict(summary):
            raise ResumeMismatchError(f"{item.file}: summary differs from its stdout")
        entries.append(
            {
                "slot": item.slot,
                "kind": receipt["kind"],
                "arm": receipt["arm"],
                "seed": receipt["seed"],
                "attempt": receipt["attempt"],
                "receipt": item.file,
                "receipt_sha256": audit.digest(resume_from / item.file),
                "base_program_sha256": receipt["base_program_sha256"],
                "program_sha256": receipt["program_sha256"],
                "num_steps": summary["num_steps"],
                "step_floor": floors[str(receipt["arm"])],
            }
        )
    return entries


def preflight(
    source: Path,
    python: Path,
    resume_from: Path,
    *,
    prior_plan: Path = PRIOR_PLAN,
) -> dict[str, object]:
    """Build the frozen Amendment 4 plan; refuse if the prior run does not match."""
    resume_from = resume_from.resolve()
    prior = json.loads(prior_plan.read_text(encoding="utf-8"))
    fresh = valid.preflight(source, python)
    if (
        prior["plan_hash"] != content_hash(prior["plan"])
        or prior["plan_hash"] != fresh["plan_hash"]
        or prior["plan"] != plain(fresh["plan"])
    ):
        raise ResumeMismatchError(
            "Amendment 3 plan does not reproduce from the current sources"
        )
    run_plan = json.loads((resume_from / "plan.json").read_text(encoding="utf-8"))
    if run_plan != {"plan_hash": prior["plan_hash"], "plan": prior["plan"]}:
        raise ResumeMismatchError("Prior run's plan differs from the frozen plan")
    plan = dict(fresh["plan"])  # type: ignore[call-overload]
    order = plan["order"]
    floors = {
        arm: item["floor"]
        for arm, item in plan["validity"]["reference_steps"].items()  # type: ignore[index]
    }
    receipts = load_receipt_files(resume_from)
    retained = select_retained(receipts, order, floors)  # type: ignore[arg-type]
    if not retained:
        raise ResumeMismatchError("No valid receipts to retain")
    if not missing_slots(order, retained):  # type: ignore[arg-type]
        raise ResumeMismatchError("Every slot is already retained; nothing to run")
    entries = verify_retained_evidence(resume_from, retained, floors)
    kept_files = {r.file for r in retained}
    plan["study"] = (
        "S1 part 2, Amendment 4: resume of the Amendment 3 run "
        "after a memory-guard stop"
    )
    plan["limits"] = {**plan["limits"], "memory_guard": memory_guard_policy()}  # type: ignore[call-overload]
    plan["resume"] = {
        "source_directory": display(resume_from),
        "source_plan": {
            "path": display(prior_plan),
            "plan_hash": prior["plan_hash"],
            "file_sha256": audit.digest(prior_plan),
        },
        "shard_sha256": plan["heldout_phase"]["shard"]["sha256"],  # type: ignore[index]
        "rule": "Only receipts with validity.valid true and a successful summary "
        "are retained, and each must match its frozen slot, program hashes and "
        "step floor. Retained slots are never run again.",
        "retained": entries,
        "missing_slots": [
            {k: item[k] for k in ("index", "kind", "arm", "seed")}
            for item in missing_slots(order, retained)  # type: ignore[arg-type]
        ],
        "not_retained": [
            {
                "receipt": name,
                "receipt_sha256": audit.digest(resume_from / name),
                "reason": why_not_retained(receipt),
            }
            for name, receipt in receipts.items()
            if name not in kept_files
        ],
        "stop_evidence": {
            name: audit.digest(resume_from / name)
            for name in STOP_FILES
            if (resume_from / name).is_file()
        },
    }
    plan["code"] = {
        **plan["code"],  # type: ignore[call-overload]
        "audit_heldout_resume_script_sha256": audit.digest(Path(__file__)),
        "resume_module_sha256": audit.digest(REPO / "src/openloop/studies/resume.py"),
    }
    return {"plan_hash": content_hash(plan), "plan": plan}


def pending_slots(plan: Mapping[str, object]) -> list[Mapping[str, object]]:
    """The slots to run: the frozen order minus every retained slot."""
    retained = {e["slot"] for e in plan["resume"]["retained"]}  # type: ignore[index]
    return [item for item in plan["order"] if item["index"] not in retained]  # type: ignore[attr-defined]


def retained_receipts(plan: Mapping[str, object]) -> list[dict[str, object]]:
    """Re-read the retained receipts, check their hashes, and mark their source."""
    resume = plan["resume"]
    directory = Path(resume["source_directory"])  # type: ignore[index]
    if not directory.is_absolute():
        directory = REPO / directory
    result: list[dict[str, object]] = []
    for entry in resume["retained"]:  # type: ignore[index]
        path = directory / entry["receipt"]
        if audit.digest(path) != entry["receipt_sha256"]:
            raise ResumeMismatchError(f"{entry['receipt']} changed since the plan")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        result.append(
            {
                **receipt,
                SOURCE_KEY: resume["source_directory"],  # type: ignore[index]
                "source_receipt_sha256": entry["receipt_sha256"],
            }
        )
    return result


def write_reports(output: Path, plan: dict[str, object]) -> None:
    """Write new analysis files from retained plus new valid attempts only."""
    retained = retained_receipts(plan)
    new = [
        {**receipt, SOURCE_KEY: display(output)}
        for receipt in load_receipt_files(output).values()
    ]
    receipts = [*retained, *new]
    valid_ones = valid_receipts(receipts)
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
    completion = [r for r in valid_ones if r.get("kind") == base.KIND_COMPLETION]
    audit.save(
        output / base.SEED_COMPLETION_REPORT,
        analyze_audit(
            comparisons,
            combined_observations(
                list(valid.load_receipts(base.PARENT_RUN)), completion
            ),
        ),
    )
    audit.save(
        output / base.HELDOUT_REPORT,
        analyze_heldout(
            heldout_run_observations(valid_ones), child=base.CHILD, parent=base.PARENT
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
                    "source_directory": r.get(SOURCE_KEY),
                    "source_receipt_sha256": r.get("source_receipt_sha256"),
                    "validity": r.get("validity"),
                }
                for r in receipts
            ],
            "retained": len(retained),
            "excluded_invalid": sum(1 for r in receipts if r not in valid_ones),
            "prior_attempts_not_retained": plan["resume"]["not_retained"],  # type: ignore[index]
        },
    )


async def run_watched(
    executor: LocalExecutor,
    job: ProcessJob,
    *,
    key: str,
    read: audit.PressureReader,
    interval: float,
    floor: int,
    wall: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
) -> valid.Watched:
    """Like `audit_heldout_valid.run_watched`, with the Amendment 4 floor."""
    guard = FloorMemoryGuard(floor)
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
                return valid.Watched(snapshot, memory_trip, None, drift.total)
            drift_trip: DriftTrip | None = drift.observe(wall(), monotonic())
            if drift_trip is not None:
                snapshot = await executor.cancel(job_id)
                return valid.Watched(snapshot, None, drift_trip, drift.total)
            done, _ = await asyncio.wait({collector}, timeout=interval)
            if done:
                drift_trip = drift.observe(wall(), monotonic())
                return valid.Watched(collector.result(), None, drift_trip, drift.total)
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
    """Like `audit.stop_for_memory`, but records the Amendment 4 thresholds."""
    ledger.fail(run_id, f"memory guard: {trip.reason}")
    record["failure"] = f"memory guard: {trip.reason}"
    audit.save(receipt_path, record)
    audit.save(
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
    raise audit.MemoryGuardTripped(f"Memory guard tripped: {trip.reason}")


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
    floor = int(plan["limits"]["memory_guard"]["free_percent_floor"])  # type: ignore[index]
    total = len(plan["order"])  # type: ignore[arg-type]
    pending = pending_slots(plan)  # type: ignore[arg-type]
    retained_receipts(plan)  # type: ignore[arg-type]  # Re-verify before any run.
    read()  # Refuse to start unguarded.
    output.mkdir(parents=True, exist_ok=False)
    audit.save(output / "plan.json", prepared)
    audit.save(
        output / "started.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "macos": platform.mac_ver()[0],
            "slots_to_run": [item["index"] for item in pending],
        },
    )
    executor = LocalExecutor(output / "executor", memory_sample_seconds=5)
    results = 0
    failures = 0
    try:
        with Ledger(output / "ledger.sqlite3") as ledger:
            async with asyncio.timeout(STUDY_WALL_SECONDS):
                for item in pending:
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
                        item: Mapping[str, object] = item,
                    ) -> valid.Attempt:
                        nonlocal results, failures
                        print(
                            f"S1p2-A4 {index}/{total}: {kind} {arm} seed {seed} "
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
                        key = f"s1p2a4:{plan_hash}:{kind}:{arm}:{seed}:a{number}"
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
                                    {"study": "s1-part2-amendment-4", "plan": plan_hash}
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
                                f"S1p2 Amendment 4 {kind} of {arm} on this machine"
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
                            floor=floor,
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
                            stop_for_memory(
                                output,
                                output / name,
                                ledger,
                                run.id,
                                record,
                                watched.memory_trip,
                            )
                        verdict = valid.Attempt(True)
                        if watched.drift_trip is not None:
                            reason = watched.drift_trip.reason
                            verdict = valid.Attempt(False, reason)
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
                                    verdict = valid.Attempt(False, reason)
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

                    await valid.run_slot(attempt_once, settle, max_attempts=cap)
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
    parser.add_argument(
        "--resume-from", type=Path, help="the Amendment 3 run directory to resume"
    )
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
    if args.source is None or args.worker_python is None or args.resume_from is None:
        parser.error("--source, --worker-python and --resume-from are required")
    source, python = args.source.resolve(), args.worker_python.absolute()
    prepared = preflight(source, python, args.resume_from)
    slots = len(pending_slots(prepared["plan"]))  # type: ignore[arg-type]
    if not args.execute_approved:
        if args.plan is None:
            parser.error("preparing requires --plan")
        audit.save(
            args.plan,
            {
                **prepared,
                "prepared_at": datetime.now(UTC).isoformat(),
                "training_runs": slots,
                "native_training_executed": False,
            },
        )
        print(
            f"Prepared {slots} runs, plan hash {prepared['plan_hash']}. "
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
    if args.output.resolve() == args.resume_from.resolve():
        raise ConflictError("Output must be a new directory, not the resumed run")
    asyncio.run(execute(source, python, prepared, args.output.resolve()))


if __name__ == "__main__":
    main()
