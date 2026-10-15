"""Prepare, then explicitly execute one bounded T2 engineering smoke, not S6."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from openloop.env_probe import environment_identity
from openloop.executors import LocalExecutor
from openloop.ledger import Ledger, canonical_json, content_hash
from openloop.loops import T2, ExperimentEnv, LoopAction
from openloop.loops.t2 import T2_FILES
from openloop.training import TrainingError
from openloop.training.grpo import ArithmeticData
from openloop.training.prepare_model import load_snapshot


async def execute(loop: T2, root: Path, plan: dict) -> None:
    """Use a new ledger and retain its complete successful or failed attempts."""
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).parents[1] / "loops"
    (root / "source.json").write_text(
        canonical_json(
            {
                "files": {name: (source_root / name).read_text() for name in T2_FILES},
                "runtime_lock": loop.runtime_lock,
                "adapter_hash": loop.spec.adapter_hash,
                "coordinator": Path(__file__).read_text(),
            }
        )
        + "\n"
    )
    executor = LocalExecutor(root / "executor")
    try:
        workload = T2(
            loop.model,
            loop.data,
            loop.python,
            runtime_lock=loop.runtime_lock,
            output_root=root / "checkpoints",
            executor=executor,
        )
        with Ledger(root / "ledger.sqlite3") as ledger:
            env = ExperimentEnv(
                workload,
                ledger,
                runner=workload.run,
                config=plan["inputs"]["config"],
                environment_hash=plan["inputs"]["environment_hash"],
                budget=2,
            )
            step = await env.step(LoopAction(42, "smoke"))
            report = {
                "kind": "engineering_smoke",
                "plan_hash": content_hash(plan),
                "finished_at": datetime.now(UTC).isoformat(),
                "run": ledger.get(step.run_id),
            }
            (root / "report.json").write_text(canonical_json(report) + "\n")
            print(canonical_json(report))
    finally:
        await executor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model = load_snapshot(args.model_manifest)
    model.verify()
    lock = (Path(__file__).parents[3] / "uv.lock").read_text()
    loop = T2(
        model,
        ArithmeticData.default(),
        Path(sys.executable),
        runtime_lock=lock,
        output_root=args.output / "checkpoints",
    )
    environment = environment_identity()
    inputs = loop.spec.inputs(
        loop.normalize_config({}), LoopAction(42, "smoke"), content_hash(environment)
    )
    plan = {
        "kind": "engineering_smoke",
        "environment": environment,
        "inputs": inputs,
        "model_snapshot": model.hash,
        "loop_spec": loop.spec,
        "limits": {
            "updates": 2,
            "rollout_completions": 8,
            "max_new_tokens": 64,
            "validation_tasks": 16,
            "max_generated_tokens": 1536,
            "wall_seconds": 1800,
            "paid_services": False,
        },
    }
    if not args.execute:
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open("x") as stream:
            stream.write(canonical_json(plan) + "\n")
        print(
            canonical_json({"plan_hash": content_hash(plan), "limits": plan["limits"]})
        )
        return
    recorded = json.loads(args.plan.read_text())
    if canonical_json(recorded) != canonical_json(plan):
        raise TrainingError("Smoke plan differs from current frozen inputs")
    asyncio.run(execute(loop, args.output, recorded))


if __name__ == "__main__":
    main()
