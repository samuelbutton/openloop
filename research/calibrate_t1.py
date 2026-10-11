"""Prepare or explicitly execute S1 on the local MLX GPU, with no paid services.

Preflight reads frozen source, tokenizer, and train/validation memberships.
Execution uses a NEW output directory and ledger through public core APIs.
No existing evidence is overwritten. The held-out membership is never opened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from openloop.executors import LocalExecutor
from openloop.ledger import (
    ConflictError,
    JSONValue,
    Ledger,
    Purpose,
    Run,
    Status,
    content_hash,
)
from openloop.ledger.validation import plain
from openloop.loops import T1, ExperimentEnv, LoopAction, Phase, TrainingSource
from openloop.loops.corpus import Corpus
from openloop.loops.t1 import EVALUATION_TOKENS
from openloop.studies.s1 import StudyPlan, analyze, build_plan


def save(path: Path, value: object) -> None:
    """Write one new evidence record, refusing all overwrites."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def preflight(
    upstream: Path,
    corpus: Path,
    tokenizer: Path,
    python: Path,
) -> tuple[T1, StudyPlan, dict[str, JSONValue]]:
    """Check bytes, installed locked versions, hardware and all generated programs.

    This call starts no training and imports no MLX backend.
    """
    source = TrainingSource.load(upstream)
    data = Corpus.load(corpus)
    data.verify()
    environment = {
        "PATH": os.defpath,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    runtime = subprocess.run(
        [
            str(python),
            "-c",
            "import json,platform,sys\n"
            "from openloop.loops.t1_worker import verify_runtime\n"
            "print(json.dumps({'python':platform.python_version(),"
            "'architecture':platform.machine(),"
            "'dependencies':verify_runtime(sys.stdin.read())}))",
        ],
        input=source.lock,
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    observed = plain(json.loads(runtime.stdout))
    if not isinstance(observed, Mapping):
        raise ConflictError("The runtime probe must return a JSON object")
    if platform.system() != "Darwin" or observed["architecture"] != "arm64":
        raise ConflictError("S1 requires native Apple silicon macOS")
    hardware = {
        "chip": subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "ram_bytes": int(
            subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True)
        ),
        "macos": platform.mac_ver()[0],
        **observed,
    }
    loop = T1(source, data, tokenizer, python)
    configs = {
        name: loop.normalize_config(config)
        for name, config in {
            "baseline": {},
            "matrix_lr_half": {"MATRIX_LR": 0.02},
            "matrix_lr_double": {"MATRIX_LR": 0.08},
            "embedding_lr_half": {"EMBEDDING_LR": 0.3},
            "embedding_lr_double": {"EMBEDDING_LR": 1.2},
            "weight_decay_zero": {"WEIGHT_DECAY": 0.0},
        }.items()
    }
    plan = build_plan(loop.spec, configs, content_hash(hardware))
    for trial in plan.trials:
        program = loop.build_job(trial.inputs).program
        if program is None:
            raise ConflictError("T1 omitted the training program")
        compile(program, "candidate/train.py", "exec")
    return loop, plan, hardware


async def execute(loop: T1, plan: StudyPlan, output: Path) -> None:
    """Run the frozen jobs serially. Stop on failure, cancellation or wall cap."""
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {"hash": plan.hash, "plan": plan})
    save(output / "started.json", {"started_at": datetime.now(UTC).isoformat()})
    executor = LocalExecutor(output / "executor")
    worker = T1(
        loop.source, loop.corpus, loop.tokenizer_dir, loop.python, executor=executor
    )
    runs: dict[str, Run] = {}
    try:
        with Ledger(output / "ledger.sqlite3") as ledger:
            envs = {
                candidate: ExperimentEnv(
                    worker,
                    ledger,
                    runner=worker.run,
                    config=next(
                        trial.inputs.config
                        for trial in plan.trials
                        if trial.candidate == candidate
                    ),
                    environment_hash=plan.trials[0].inputs.environment_hash,
                    budget=sum(
                        trial.inputs.budget_amount
                        for trial in plan.trials
                        if trial.candidate == candidate
                    ),
                    hypothesis="S1 calibration on validation only",
                )
                for candidate in dict.fromkeys(trial.candidate for trial in plan.trials)
            }
            async with asyncio.timeout(plan.wall_limit_seconds):
                for i, trial in enumerate(plan.trials):
                    print(f"S1 {i + 1}/{len(plan.trials)}: {trial.slot}", flush=True)
                    step = await envs[trial.candidate].step(
                        LoopAction(
                            trial.inputs.seed,
                            trial.inputs.fidelity,
                            Phase(trial.inputs.phase),
                            request_key=f"s1:{plan.hash}:{trial.slot}",
                            purpose=Purpose.REPLICATION,
                        )
                    )
                    run = ledger.get(step.run_id)
                    save(
                        output / f"run-{i + 1:03}.json",
                        {"slot": trial.slot, "run": run},
                    )
                    if run.status is not Status.SUCCEEDED or not step.executed:
                        raise ConflictError(
                            "S1 requires a newly executed successful sample"
                        )
                    runs[trial.slot] = run
                    assert run.result is not None
                    print(
                        f"val_bpb={run.result.metrics['val_bpb'].value:.6f}", flush=True
                    )
            save(output / "report.json", analyze(plan, runs))
    except BaseException as error:
        save(
            output / "interrupted.json",
            {
                "observed_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "reason": str(error),
                "completed_runs": len(runs),
            },
        )
        raise
    finally:
        await executor.close()
    save(output / "finished.json", {"finished_at": datetime.now(UTC).isoformat()})


def main() -> None:
    """Prepare a reviewable plan by default. Execution requires an explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    loop, plan, hardware = preflight(
        args.upstream, args.corpus, args.tokenizer, args.worker_python
    )
    if not args.execute_approved:
        save(
            args.plan,
            {
                "plan_hash": plan.hash,
                "plan": plan,
                "environment": hardware,
                "prepared_at": datetime.now(UTC).isoformat(),
                "training_runs": len(plan.trials),
                "training_tokens": plan.training_tokens,
                "evaluation_tokens": len(plan.trials) * EVALUATION_TOKENS,
                "native_training_executed": False,
            },
        )
        print(
            f"Prepared {len(plan.trials)} runs, "
            f"{plan.training_tokens:,} training tokens; "
            f"cap {plan.wall_limit_seconds / 3600:g} hours. No training executed."
        )
        return
    if args.output is None:
        parser.error("--execute-approved requires a new --output directory")
    prepared = json.loads(args.plan.read_text(encoding="utf-8"))
    if prepared["plan_hash"] != plan.hash or prepared["plan"] != plain(plan):
        raise ConflictError(
            "Prepared plan or execution environment changed; prepare again"
        )
    asyncio.run(execute(loop, plan, args.output))


if __name__ == "__main__":
    main()
