"""Trusted GRPO worker. No network, arbitrary candidate code, or held-out reads."""

import json
import sys
import time
import tomllib
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from openloop.ledger import ExperimentInputs, canonical_json
from openloop.ledger.validation import finite_number, freeze_object
from openloop.training import AdamParams, LossConfig, SamplingParams, TrainingError
from openloop.training.assets import ModelSnapshot
from openloop.training.grpo import ArithmeticData, ArithmeticTask, train_grpo
from openloop.training.mlx_lm_backend import load_mlx_clients

from .models import Job, JobOutput
from .t2 import CLIP_EPSILON, T2

RUNTIME_PACKAGES = (
    "mlx",
    "mlx-lm",
    "numpy",
    "transformers",
    "tokenizers",
    "safetensors",
)


def verify_runtime(lock: str) -> dict[str, str]:
    """Require the selected installed training packages to occur in the full lock."""
    if sys.version_info[:2] != (3, 12):
        raise TrainingError("T2 requires Python 3.12")
    packages = tomllib.loads(lock)["package"]
    observed = {name: version(name) for name in RUNTIME_PACKAGES}
    for name, actual in observed.items():
        if actual not in {
            entry["version"] for entry in packages if entry["name"] == name
        }:
            raise TrainingError(f"T2 runtime differs from lock: {name}")
    return observed


def run_posttraining(loop: T2, job: Job, output: Path) -> JobOutput:
    """Run the exact declared update count, fixed greedy validation, and checkpoint."""
    if loop.build_job(job.inputs) != job:
        raise TrainingError("T2 worker received a changed job")
    observed = verify_runtime(loop.runtime_lock)
    config = job.inputs.config
    params = AdamParams(
        learning_rate=finite_number(config["learning_rate"], "Learning rate")
    )
    limit, group = config["max_new_tokens"], config["group_size"]
    if not isinstance(limit, int) or not isinstance(group, int):
        raise TrainingError("T2 requires integer rollout limits")
    actor, reference, tokenizer = load_mlx_clients(
        loop.model, params=params, seed=job.inputs.seed
    )
    start = time.monotonic()
    result = train_grpo(
        actor,
        reference,
        tokenizer,
        loop.data,
        updates=job.inputs.budget_amount,
        sampling=SamplingParams(
            limit,
            group,
            1.0,
            job.inputs.seed,
            tokenizer.stop_tokens,
        ),
        loss=LossConfig(
            clip_epsilon=CLIP_EPSILON,
            kl_beta=finite_number(config["kl_beta"], "KL beta"),
        ),
    )
    seconds = time.monotonic() - start
    checkpoint = actor.save_state(output / "checkpoint")
    return freeze_object(
        {
            **asdict(result),
            "work_units": result.updates,
            "run_seconds": seconds,
            "checkpoint_sha256": checkpoint.sha256,
            "runtime": observed,
        }
    )


def main() -> None:
    payload = json.loads(sys.stdin.read())
    record = payload["model"]
    model = ModelSnapshot(
        Path(record["root"]), record["files"], record["model_id"], record["revision"]
    )
    data = ArithmeticData(
        **{
            name: tuple(ArithmeticTask(**task) for task in tasks)
            for name, tasks in payload["data"].items()
        }
    )
    output = Path(payload["output_root"])
    loop = T2(
        model,
        data,
        Path(sys.executable),
        runtime_lock=payload["runtime_lock"],
        output_root=output,
    )
    job = Job(ExperimentInputs(**payload["job"]["inputs"]), payload["job"]["program"])
    print(
        "OPENLOOP_RESULT " + canonical_json(run_posttraining(loop, job, output)),
        flush=True,
    )


if __name__ == "__main__":
    main()
