"""T1 MLX adapter. Constants are mutable; training source, data, and scoring are frozen.

Importing or constructing this adapter starts no training and downloads nothing.
The explicit run method requires an approved native MLX training environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openloop.ledger import (
    Direction,
    ExperimentInputs,
    JSONValue,
    Metric,
    Reproducibility,
    Result,
    content_hash,
)
from openloop.ledger.identity import canonical_json, check_text, freeze_object, plain

from .corpus import Corpus
from .files import adapter_digest, file_digest
from .models import (
    ContractError,
    Fidelity,
    FidelityName,
    Job,
    JobOutput,
    LoopSpec,
    finite_number,
)
from .training import TrainingSetting, baseline_config, training_program

UPSTREAM_REVISION = "766a25ff22afa799efd8d0aa450a4348e4749df2"
_UPSTREAM_HASHES = {
    "train.py": "49f0993f10b5f8af223c6be272b067fe1c8b5171ee6f5add8ed177b71e459124",
    "prepare.py": "561d40660b39628fbc9cc45430613534acaea23765c13b5fcd3c5563cf4983b1",
    "uv.lock": "d260945ae38ffc55443f7bdad5cf4c3b3b9b5313cf7cb36ca795ffe7422ca0ab",
}
SEQUENCE_LENGTH = 2048
EVALUATION_TOKENS = 1_572_864
EVALUATION_BATCH = 256
WALL_LIMIT_SECONDS = 3600


@dataclass(frozen=True)
class TrainingSource:
    """Complete trusted source snapshots. The loader checks the pinned upstream."""

    train: str
    prepare: str
    lock: str

    def __post_init__(self) -> None:
        for snapshot in (self.train, self.prepare, self.lock):
            check_text(snapshot)

    @classmethod
    def load(cls, repository: Path) -> TrainingSource:
        snapshots: list[str] = []
        for name, expected in _UPSTREAM_HASHES.items():
            data = (repository / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ContractError(f"Pinned upstream source changed: {name}")
            snapshots.append(data.decode("utf-8"))
        return cls(*snapshots)

    @property
    def hash(self) -> str:
        return content_hash({"train.py": self.train, "prepare.py": self.prepare})


def tokenizer_digest(directory: Path) -> str:
    return content_hash(
        {
            name: file_digest(directory / name)
            for name in ("tokenizer.pkl", "token_bytes.npy")
        }
    )


class T1:
    """Two fixed-token fidelities using frozen FineWeb train/validation memberships."""

    def __init__(
        self,
        source: TrainingSource,
        corpus: Corpus,
        tokenizer_dir: Path,
    ) -> None:
        self._source = source
        self._corpus = corpus
        self._tokenizer_dir = tokenizer_dir.resolve()
        self._defaults = baseline_config(source.train)
        adapter = adapter_digest(
            "t1.py",
            "training.py",
            "corpus.py",
            "t1_worker.py",
            "models.py",
            "files.py",
        )
        self._spec = LoopSpec(
            name="t1",
            source_hash=source.hash,
            dependencies_hash=content_hash(source.lock),
            data_hash=corpus.snapshot_hash,
            evaluator_hash=content_hash({"prepare": source.prepare, "worker": adapter}),
            adapter_hash=adapter,
            budget_unit="tokens",
            reproducibility=Reproducibility.NOISY,
            fidelities=(
                Fidelity(FidelityName.SCREEN, 8_388_608),
                Fidelity(FidelityName.CONFIRM, 20_971_520),
            ),
            mutable_keys=tuple(TrainingSetting),
            metric_name="val_bpb",
            metric_unit="BPB",
            tokenizer_hash=tokenizer_digest(self.tokenizer_dir),
            execution={
                "device": "gpu",
                "backend": "mlx",
                "sequence_length": SEQUENCE_LENGTH,
                "evaluation_tokens": EVALUATION_TOKENS,
                "evaluation_batch": EVALUATION_BATCH,
                "wall_limit_seconds": WALL_LIMIT_SECONDS,
            },
        )

    @property
    def source(self) -> TrainingSource:
        return self._source

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def tokenizer_dir(self) -> Path:
        return self._tokenizer_dir

    @property
    def spec(self) -> LoopSpec:
        return self._spec

    def normalize_config(
        self, config: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        if set(config) - set(TrainingSetting):
            raise ContractError("Candidate changes a frozen or unknown setting")
        settings = dict(self._defaults) | dict(config)
        for setting in (
            TrainingSetting.DEPTH,
            TrainingSetting.ASPECT_RATIO,
            TrainingSetting.HEAD_DIM,
            TrainingSetting.TOTAL_BATCH_SIZE,
            TrainingSetting.DEVICE_BATCH_SIZE,
        ):
            value = settings[setting]
            if type(value) is not int or value < 1:
                raise ContractError(f"{setting} must be a positive integer")
        batch = settings[TrainingSetting.TOTAL_BATCH_SIZE]
        device_batch = settings[TrainingSetting.DEVICE_BATCH_SIZE]
        if not isinstance(batch, int) or not isinstance(device_batch, int):
            raise ContractError("Batch sizes must be integers")
        if batch % (device_batch * SEQUENCE_LENGTH) or any(
            level.amount % batch for level in self.spec.fidelities
        ):
            raise ContractError(
                "Batch sizes must divide accumulation and fidelity budgets"
            )
        window = settings[TrainingSetting.WINDOW_PATTERN]
        if not isinstance(window, str) or not window or set(window) - set("SL"):
            raise ContractError("Window pattern must contain only S and L")
        for setting in (
            TrainingSetting.EMBEDDING_LR,
            TrainingSetting.UNEMBEDDING_LR,
            TrainingSetting.MATRIX_LR,
            TrainingSetting.SCALAR_LR,
            TrainingSetting.WEIGHT_DECAY,
            TrainingSetting.WARMUP_RATIO,
            TrainingSetting.WARMDOWN_RATIO,
            TrainingSetting.FINAL_LR_FRAC,
        ):
            value = finite_number(settings[setting], str(setting))
            if value < 0:
                raise ContractError(f"{setting} must be finite and nonnegative")
        warmup, cooldown = (
            settings[TrainingSetting.WARMUP_RATIO],
            settings[TrainingSetting.WARMDOWN_RATIO],
        )
        if not isinstance(warmup, (int, float)) or not isinstance(
            cooldown, (int, float)
        ):
            raise ContractError("Schedule ratios must be numeric")
        if (
            warmup + cooldown > 1
            or finite_number(
                settings[TrainingSetting.FINAL_LR_FRAC], "Final learning rate fraction"
            )
            > 1
        ):
            raise ContractError("Schedule ratios must fit within one token budget")
        betas = settings[TrainingSetting.ADAM_BETAS]
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(not 0 <= finite_number(beta, "Adam beta") < 1 for beta in betas)
        ):
            raise ContractError("Adam betas must contain two numbers in [0, 1)")
        return freeze_object(settings)

    def build_job(self, inputs: ExperimentInputs) -> Job:
        self.spec.validate_inputs(inputs)
        if self.normalize_config(inputs.config) != inputs.config:
            raise ContractError("Candidate settings must include normalized defaults")
        return Job(
            inputs,
            training_program(
                self.source.train,
                inputs.config,
                seed=inputs.seed,
                token_budget=inputs.budget_amount,
                tokenizer_dir=str(self.tokenizer_dir),
            ),
        )

    async def run(self, job: Job) -> JobOutput:
        """Run trusted training explicitly, after the caller approves GPU training."""
        if self.build_job(job.inputs) != job:
            raise ContractError("Job program differs from the frozen adapter")
        payload = canonical_json(
            {
                "source": self.source,
                "corpus_root": str(self.corpus.root),
                "tokenizer_dir": str(self.tokenizer_dir),
                "job": job,
            }
        ).encode("utf-8")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "openloop.loops.t1_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.defpath, "PYTHONPATH": str(Path(__file__).parents[2])},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload), timeout=WALL_LIMIT_SECONDS
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            raise ContractError(f"Training worker failed: {stderr.decode('utf-8')}")
        records = stdout.decode("utf-8").splitlines()
        if not records or not records[-1].startswith("OPENLOOP_RESULT "):
            raise ContractError("Worker omitted its trusted result record")
        output = plain(json.loads(records[-1].removeprefix("OPENLOOP_RESULT ")))
        if not isinstance(output, Mapping):
            raise ContractError("Worker result must be a JSON object")
        return freeze_object(output)

    def evaluate(self, job: Job, output: JobOutput) -> Result:
        batch = job.inputs.config[TrainingSetting.TOTAL_BATCH_SIZE]
        if not isinstance(batch, int):
            raise ContractError("Batch size must be an integer")
        if (
            output.get("work_units") != job.inputs.budget_amount
            or output.get("steps") != (job.inputs.budget_amount // batch)
            or output.get("evaluation_tokens") != EVALUATION_TOKENS
        ):
            raise ContractError(
                "Training or evaluation did not use the declared token budget"
            )
        if any(
            type(output.get(key)) is not int
            for key in (
                "work_units",
                "steps",
                "evaluation_tokens",
            )
        ):
            raise ContractError("Token and step counts must be integers")
        score = finite_number(output.get("val_bpb"), "Validation BPB")
        if score < 0:
            raise ContractError("Validation BPB must be finite and nonnegative")
        seconds = finite_number(output.get("training_seconds"), "Training seconds")
        if seconds <= 0:
            raise ContractError("Observed training seconds must be positive")
        return Result(
            {
                self.spec.metric_name: Metric(score, self.spec.metric_unit),
                "training_seconds": Metric(seconds, "seconds", split="train"),
                "tokens_per_second": Metric(
                    job.inputs.budget_amount / seconds,
                    "tokens/s",
                    Direction.MAXIMIZE,
                    split="train",
                ),
            }
        )
