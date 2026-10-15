"""T2 post-training adapter: frozen arithmetic scoring and bounded local GRPO.

Candidates change declared recipe settings, never model assets or reward code.
Construction starts no inference, training, or network requests.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from openloop.executors import Executor
from openloop.ledger import (
    Direction,
    ExperimentInputs,
    JSONValue,
    Metric,
    Reproducibility,
    Result,
    canonical_json,
    content_hash,
)
from openloop.ledger.validation import finite_number, freeze_object
from openloop.training import AdamParams, LossConfig, SamplingParams
from openloop.training.assets import ModelSnapshot
from openloop.training.grpo import ArithmeticData
from openloop.training.mlx_lm_backend import LORA_CONFIG, LORA_LAYERS

from .files import adapter_digest
from .mlx_process import execution_metrics, run_worker
from .models import ContractError, Fidelity, Job, JobOutput, LoopSpec, RunContext

T2_FILES = (
    "t2.py",
    "t2_worker.py",
    "models.py",
    "mlx_process.py",
    "../training/contracts.py",
    "../training/assets.py",
    "../training/grpo.py",
    "../training/loss.py",
    "../training/mlx_backend.py",
    "../training/mlx_lm_backend.py",
    "../training/checkpoints.py",
)
DEFAULT_CONFIG: Mapping[str, JSONValue] = freeze_object(
    {
        "learning_rate": 1e-4,
        "clip_epsilon": 0.2,
        "kl_beta": 0.04,
        "group_size": 4,
        "max_new_tokens": 64,
    }
)
WALL_SECONDS = 1800


@dataclass(frozen=True)
class T2:
    """Two GRPO update fidelities on a pinned local 0.5B quantized model."""

    model: ModelSnapshot
    data: ArithmeticData
    python: Path
    runtime_lock: str = field(kw_only=True)
    output_root: Path = field(kw_only=True)
    executor: Executor | None = field(default=None, kw_only=True, compare=False)
    _spec: LoopSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        model, data, runtime_lock = self.model, self.data, self.runtime_lock
        adapter = adapter_digest(*T2_FILES)
        spec = LoopSpec(
            name="t2",
            source_hash=content_hash({"model": model.hash, "adapter": adapter}),
            dependencies_hash=content_hash(runtime_lock),
            data_hash=content_hash(data),
            evaluator_hash=adapter,
            adapter_hash=adapter,
            budget_unit="optimizer_updates",
            reproducibility=Reproducibility.NOISY,
            fidelities=(Fidelity("smoke", 2), Fidelity("short", 8)),
            mutable_keys=tuple(DEFAULT_CONFIG),
            metric_name="pass_rate",
            metric_unit="fraction",
            direction=Direction.MAXIMIZE,
            tokenizer_hash=content_hash(model.files),
            execution={
                "backend": "mlx",
                "device": "gpu",
                "model": model.model_id,
                "revision": model.revision,
                "model_snapshot": model.hash,
                "lora": LORA_CONFIG,
                "lora_layers": LORA_LAYERS,
                "max_prompt_tokens": 128,
                "wall_limit_seconds": WALL_SECONDS,
                "evaluation_tasks": len(data.validation),
                "evaluation_temperature": 0.0,
            },
        )
        object.__setattr__(self, "_spec", spec)

    @property
    def spec(self) -> LoopSpec:
        return self._spec

    def normalize_config(
        self, config: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        if set(config) - set(DEFAULT_CONFIG):
            raise ContractError("Candidate changes a frozen T2 setting")
        settings = dict(DEFAULT_CONFIG) | dict(config)
        learning_rate = finite_number(settings["learning_rate"], "Learning rate")
        clip = finite_number(settings["clip_epsilon"], "Clip epsilon")
        beta = finite_number(settings["kl_beta"], "KL beta")
        group, limit = settings["group_size"], settings["max_new_tokens"]
        if type(group) is not int or not 2 <= group <= 8 or type(limit) is not int:
            raise ContractError(
                "T2 group size must be 2-8 and generation limit an integer"
            )
        AdamParams(learning_rate=learning_rate)
        LossConfig(clip_epsilon=clip, kl_beta=beta)
        SamplingParams(max_tokens=limit, num_samples=group)
        return freeze_object(settings)

    def build_job(self, inputs: ExperimentInputs) -> Job:
        self.spec.validate_inputs(inputs)
        if self.normalize_config(inputs.config) != inputs.config:
            raise ContractError("T2 settings must include normalized defaults")
        if inputs.seed >= 2**32 - 16384:
            raise ContractError("T2 seed must fit the bounded uint32 rollout schedule")
        return Job(inputs)

    async def run(self, job: Job, context: RunContext) -> JobOutput:
        """Execute only through the caller's reviewed, shared GPU executor."""
        if self.build_job(job.inputs) != job or self.executor is None:
            raise ContractError("T2 requires a valid frozen job and an executor")
        payload = canonical_json(
            {
                "model": {
                    "root": str(self.model.root),
                    "files": self.model.files,
                    "model_id": self.model.model_id,
                    "revision": self.model.revision,
                },
                "data": self.data,
                "runtime_lock": self.runtime_lock,
                "output_root": str(self.output_root / context.attempt_id),
                "job": job,
            }
        ).encode()
        return await run_worker(
            payload,
            self.python,
            self.executor,
            context,
            module="openloop.loops.t2_worker",
            wall_seconds=WALL_SECONDS,
        )

    def evaluate(self, job: Job, output: JobOutput) -> Result:
        if self.build_job(job.inputs) != job:
            raise ContractError("T2 receipt does not match its frozen job")
        updates, tasks = job.inputs.budget_amount, len(self.data.validation)
        if (
            output.get("updates") != updates
            or output.get("work_units") != updates
            or output.get("validation_count") != tasks
        ):
            raise ContractError("T2 update or evaluation budget differs")
        if any(
            type(output.get(key)) is not int
            for key in (
                "updates",
                "work_units",
                "validation_count",
                "generated_tokens",
                "gradient_tokens",
                "uniform_groups",
            )
        ):
            raise ContractError("T2 count fields must be integers")
        config = job.inputs.config
        group, limit = config["group_size"], config["max_new_tokens"]
        if not isinstance(group, int) or not isinstance(limit, int):
            raise ContractError("T2 group and token limits must be integers")
        generated = finite_number(output.get("generated_tokens"), "Generated tokens")
        gradients = finite_number(output.get("gradient_tokens"), "Gradient tokens")
        if (
            not updates * group <= gradients <= updates * group * limit
            or not tasks <= generated - gradients <= tasks * limit
        ):
            raise ContractError("T2 generated-token count exceeds the frozen envelope")
        rate = finite_number(output.get("validation_pass_rate"), "Pass rate")
        if not 0 <= rate <= 1:
            raise ContractError("Pass rate must be in [0,1]")
        if abs(rate * tasks - round(rate * tasks)) > 1e-9:
            raise ContractError("Pass rate must correspond to an integer task count")
        if (
            not 0
            <= finite_number(output.get("uniform_groups"), "Uniform groups")
            <= updates
        ):
            raise ContractError("Uniform-group count exceeds the update budget")
        if not 0 <= finite_number(output.get("mean_reward"), "Mean reward") <= 1:
            raise ContractError("Training reward must be in [0,1]")
        if (
            min(
                finite_number(output.get(name), name)
                for name in ("training_seconds", "evaluation_seconds")
            )
            <= 0
        ):
            raise ContractError(
                "Observed training and evaluation times must be positive"
            )
        if finite_number(output.get("max_gradient_norm"), "Gradient norm") < 0:
            raise ContractError("Gradient norm must be nonnegative")
        observed, logs = execution_metrics(output)
        checkpoint = output.get("checkpoint_sha256")
        if not isinstance(checkpoint, str):
            raise ContractError("T2 lacks a saved training checkpoint")
        return Result(
            {
                "pass_rate": Metric(
                    rate, "fraction", Direction.MAXIMIZE, sample_count=tasks
                )
            },
            artifacts={**logs, "training_state": checkpoint},
            observations={
                **observed,
                "generated_tokens": Metric(generated, "tokens", split="execution"),
                "gradient_tokens": Metric(gradients, "tokens", split="train"),
                "training_seconds": Metric(
                    finite_number(output.get("training_seconds"), "Training seconds"),
                    "seconds",
                    split="train",
                ),
                "evaluation_seconds": Metric(
                    finite_number(
                        output.get("evaluation_seconds"), "Evaluation seconds"
                    ),
                    "seconds",
                    split="val",
                ),
                "training_reward": Metric(
                    finite_number(output.get("mean_reward"), "Mean reward"),
                    "fraction",
                    split="train",
                ),
                "training_loss": Metric(
                    finite_number(output.get("mean_loss"), "Mean loss"),
                    "loss",
                    split="train",
                ),
                "uniform_groups": Metric(
                    finite_number(output.get("uniform_groups"), "Uniform groups"),
                    "groups",
                    split="train",
                ),
                "max_gradient_norm": Metric(
                    finite_number(output.get("max_gradient_norm"), "Gradient norm"),
                    "l2",
                    split="train",
                ),
            },
        )
