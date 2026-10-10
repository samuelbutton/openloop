"""T1 MLX adapter. Constants are mutable; training source, data, and scoring are frozen.

Importing or constructing this adapter starts no training and downloads nothing.
The explicit run method requires an approved native MLX training environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openloop.executors import (
    ArtifactRole,
    Executor,
    Limits,
    ProcessJob,
    Snapshot,
    State,
)
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
from openloop.ledger.validation import (
    check_text,
    finite_number,
    freeze_object,
    plain,
)

from .corpus import Corpus
from .files import adapter_digest, file_digest
from .models import (
    ContractError,
    Fidelity,
    Job,
    JobOutput,
    LoopSpec,
    RunContext,
)
from .training import TrainingSetting, baseline_config, training_program

UPSTREAM_REVISION = "766a25ff22afa799efd8d0aa450a4348e4749df2"
UPSTREAM_HASHES = {
    "train.py": "49f0993f10b5f8af223c6be272b067fe1c8b5171ee6f5add8ed177b71e459124",
    "prepare.py": "561d40660b39628fbc9cc45430613534acaea23765c13b5fcd3c5563cf4983b1",
    "uv.lock": "d260945ae38ffc55443f7bdad5cf4c3b3b9b5313cf7cb36ca795ffe7422ca0ab",
}
SEQUENCE_LENGTH = 2048
EVALUATION_TOKENS = 1_572_864
EVALUATION_BATCH = 256
# The wall limit counts from process start. T1 sets no queue, CPU, or file-size
# limit: jobs wait for the GPU lane, and training is bounded by wall time.
WALL_LIMIT_SECONDS = 3600
MEMORY_LIMIT_BYTES = 16 * 1024**3
WORKER_LOG_LIMIT_BYTES = 16 * 1024**2
LOG_TAIL_CHARACTERS = 4_000
EXECUTION_KEY = "execution"
FIDELITIES = (Fidelity("8m", 8_388_608), Fidelity("21m", 20_971_520))
ADAPTER_FILES = (
    "t1.py",
    "training.py",
    "corpus.py",
    "t1_worker.py",
    "models.py",
    "files.py",
)
_POSITIVE_INTEGERS = (
    TrainingSetting.DEPTH,
    TrainingSetting.ASPECT_RATIO,
    TrainingSetting.HEAD_DIM,
    TrainingSetting.TOTAL_BATCH_SIZE,
    TrainingSetting.DEVICE_BATCH_SIZE,
)
_NONNEGATIVE_NUMBERS = (
    TrainingSetting.EMBEDDING_LR,
    TrainingSetting.UNEMBEDDING_LR,
    TrainingSetting.MATRIX_LR,
    TrainingSetting.SCALAR_LR,
    TrainingSetting.WEIGHT_DECAY,
    TrainingSetting.WARMUP_RATIO,
    TrainingSetting.WARMDOWN_RATIO,
    TrainingSetting.FINAL_LR_FRAC,
)


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
        for name, expected in UPSTREAM_HASHES.items():
            data = (repository / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ContractError(f"Pinned upstream source changed: {name}")
            snapshots.append(data.decode("utf-8"))
        return cls(*snapshots)

    @property
    def hash(self) -> str:
        return content_hash({"train.py": self.train, "prepare.py": self.prepare})


def _tail(data: bytes) -> str:
    """Return a bounded end of a worker stream, so failure reasons stay small."""
    return data.decode("utf-8", errors="replace")[-LOG_TAIL_CHARACTERS:]


def _execution_evidence(snapshot: Snapshot) -> Mapping[str, JSONValue]:
    """Coordinator-observed executor values. Absent values stay absent."""
    evidence: dict[str, JSONValue] = {
        "job_id": snapshot.job_id,
        "state": str(snapshot.state),
        "logs": {
            artifact.path.name: {
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in snapshot.artifacts
            if artifact.role == ArtifactRole.LOG
        },
    }
    for name in ("exit_code", "queued_seconds", "elapsed_seconds", "peak_rss_bytes"):
        value = getattr(snapshot, name)
        if value is not None:
            evidence[name] = value
    return evidence


def _log_digests(snapshot: Snapshot) -> str:
    return ", ".join(
        f"{artifact.path.name} sha256={artifact.sha256}"
        for artifact in snapshot.artifacts
        if artifact.role == ArtifactRole.LOG
    )


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
        python: Path,
        *,
        executor: Executor | None = None,
    ) -> None:
        """`python` runs the worker. It must have the locked upstream dependencies.

        The path is machine-specific, so it never enters the specification.
        The worker checks installed versions against the locked dependencies.
        The caller owns `executor`: it creates and closes it, and keeps its logs.
        Only `run` needs one; the worker builds the loop without it.
        """
        self._executor = executor
        self._python = python
        self._source = source
        self._corpus = corpus
        self._tokenizer_dir = tokenizer_dir.resolve()
        self._defaults = baseline_config(source.train)
        adapter = adapter_digest(*ADAPTER_FILES)
        self._spec = LoopSpec(
            name="t1",
            source_hash=source.hash,
            dependencies_hash=content_hash(source.lock),
            data_hash=corpus.snapshot_hash,
            evaluator_hash=content_hash({"prepare": source.prepare, "worker": adapter}),
            adapter_hash=adapter,
            budget_unit="tokens",
            reproducibility=Reproducibility.NOISY,
            fidelities=FIDELITIES,
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
                "queue_limit_seconds": None,
                "cpu_limit_seconds": None,
                "file_limit_bytes": None,
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                "worker_log_limit_bytes": WORKER_LOG_LIMIT_BYTES,
            },
        )

    @property
    def source(self) -> TrainingSource:
        return self._source

    @property
    def python(self) -> Path:
        return self._python

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
        for setting in _POSITIVE_INTEGERS:
            value = settings[setting]
            if type(value) is not int or value < 1:
                raise ContractError(f"{setting} must be a positive integer")
        for setting in _NONNEGATIVE_NUMBERS:
            if finite_number(settings[setting], str(setting)) < 0:
                raise ContractError(f"{setting} must be finite and nonnegative")
        window = settings[TrainingSetting.WINDOW_PATTERN]
        if not isinstance(window, str) or not window or set(window) - set("SL"):
            raise ContractError("Window pattern must contain only S and L")
        betas = settings[TrainingSetting.ADAM_BETAS]
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(not 0 <= finite_number(beta, "Adam beta") < 1 for beta in betas)
        ):
            raise ContractError("Adam betas must contain two numbers in [0, 1)")
        self._check_batches(settings)
        schedule = finite_number(
            settings[TrainingSetting.WARMUP_RATIO], "Warmup ratio"
        ) + finite_number(settings[TrainingSetting.WARMDOWN_RATIO], "Warmdown ratio")
        final_fraction = finite_number(
            settings[TrainingSetting.FINAL_LR_FRAC], "Final learning rate fraction"
        )
        if schedule > 1 or final_fraction > 1:
            raise ContractError("Schedule ratios must fit within one token budget")
        return freeze_object(settings)

    def _check_batches(self, settings: Mapping[str, JSONValue]) -> None:
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

    async def run(self, job: Job, context: RunContext) -> JobOutput:
        """Run trusted training explicitly, after the caller approves GPU training.

        The executor job key is the attempt, so a replication runs again.
        The result holds the worker receipt and, under `execution`, the values
        that the coordinator observed. The worker cannot supply that key.
        """
        executor = self._executor
        if executor is None:
            raise ContractError("T1.run requires an executor")
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
        request = ProcessJob(
            (str(self.python), "-m", "openloop.loops.t1_worker"),
            stdin=payload,
            environment={
                "PATH": os.defpath,
                "PYTHONPATH": str(Path(__file__).parents[2]),
            },
            reviewed=True,
            gpu=True,
            limits=Limits(
                wall_seconds=WALL_LIMIT_SECONDS,
                memory_bytes=MEMORY_LIMIT_BYTES,
                output_bytes=WORKER_LOG_LIMIT_BYTES,
            ),
        )
        identifier = await executor.submit(request, key=context.attempt_id)
        try:
            snapshot = await executor.collect(identifier)
        except asyncio.CancelledError:
            # collect() is shielded, so a cancelled step must stop its own job;
            # otherwise training keeps running and holds the GPU lane.
            await executor.cancel(identifier)
            raise
        logs = {
            artifact.path.name: artifact.path.read_bytes()
            for artifact in snapshot.artifacts
            if artifact.role == ArtifactRole.LOG
        }
        stdout, stderr = logs.get("stdout", b""), logs.get("stderr", b"")
        if snapshot.state != State.SUCCEEDED:
            raise ContractError(
                f"Training worker failed with exit code {snapshot.exit_code} "
                f"({snapshot.state}): {snapshot.error or ''}"
                f"\nexecutor job {snapshot.job_id}; log digests: "
                f"{_log_digests(snapshot)}"
                f"\n--- stdout tail ---\n{_tail(stdout)}"
                f"\n--- stderr tail ---\n{_tail(stderr)}"
            )
        records = stdout.decode("utf-8").splitlines()
        if not records or not records[-1].startswith("OPENLOOP_RESULT "):
            raise ContractError("Worker omitted its trusted result record")
        output = plain(json.loads(records[-1].removeprefix("OPENLOOP_RESULT ")))
        if not isinstance(output, Mapping):
            raise ContractError("Worker result must be a JSON object")
        if EXECUTION_KEY in output:
            raise ContractError(f"Worker receipt must not contain {EXECUTION_KEY!r}")
        return freeze_object({**output, EXECUTION_KEY: _execution_evidence(snapshot)})

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
        evidence = output.get(EXECUTION_KEY)
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("state") != State.SUCCEEDED
        ):
            raise ContractError("Result lacks executor evidence of a successful job")
        observed = {
            "queued_seconds": Metric(
                finite_number(evidence.get("queued_seconds"), "Queued seconds"),
                "seconds",
                split="execution",
            ),
            "process_seconds": Metric(
                finite_number(evidence.get("elapsed_seconds"), "Process seconds"),
                "seconds",
                split="execution",
            ),
        }
        if evidence.get("peak_rss_bytes") is not None:
            observed["peak_rss_bytes"] = Metric(
                finite_number(evidence["peak_rss_bytes"], "Peak RSS"),
                "bytes",
                split="execution",
            )
        logs = evidence.get("logs")
        if not isinstance(logs, Mapping):
            raise ContractError("Executor evidence lacks log digests")
        digests: dict[str, str] = {}
        for role in ("stdout", "stderr"):
            log = logs.get(role)
            digest = log.get("sha256") if isinstance(log, Mapping) else None
            if not isinstance(digest, str):
                raise ContractError(f"Executor evidence lacks the {role} digest")
            digests[role] = digest
        return Result(
            {
                self.spec.metric_name: Metric(score, self.spec.metric_unit),
            },
            artifacts=digests,
            observations={
                **observed,
                "training_seconds": Metric(seconds, "seconds", split="train"),
                "tokens_per_second": Metric(
                    job.inputs.budget_amount / seconds,
                    "tokens/s",
                    Direction.MAXIMIZE,
                    split="train",
                ),
            },
        )
