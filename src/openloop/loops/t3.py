"""T3 varies training-mixture ratios; T1 owns the fixed recipe and validation.

The source catalog is trusted host input, never a candidate-supplied path.
Construction and job generation do not download data or start training.
"""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from openloop.ledger import (
    ExperimentInputs,
    JSONValue,
    Metric,
    Result,
    canonical_json,
    content_hash,
)
from openloop.ledger.validation import check_text, freeze_object

from .corpus import Corpus
from .files import adapter_digest
from .mixture import batch_counts, normalize_weights
from .mlx_process import run_worker
from .models import ContractError, Job, JobOutput, LoopAction, LoopSpec, RunContext
from .t1 import ADAPTER_FILES, SEQUENCE_LENGTH, T1
from .training import TrainingSetting

MIXTURE_ADAPTER_FILES = (*ADAPTER_FILES, "mixture.py", "t3.py", "t3_worker.py")


class T3:
    """A fixed T1 recipe with mutable ratios over named frozen training corpora."""

    def __init__(
        self,
        training: T1,
        sources: Mapping[str, Corpus],
        *,
        training_config: Mapping[str, JSONValue] | None = None,
    ) -> None:
        if not sources:
            raise ContractError("T3 needs at least one frozen training source")
        for name, corpus in sources.items():
            check_text(name)
            if not name or not isinstance(corpus, Corpus):
                raise ContractError("T3 sources need names and frozen Corpus records")
        if len({corpus.snapshot_hash for corpus in sources.values()}) != len(sources):
            raise ContractError(
                "Do not alias the same corpus under multiple source names"
            )
        self._training = training
        self._sources = MappingProxyType(dict(sorted(sources.items())))
        self._training_config = training.normalize_config(
            {} if training_config is None else training_config
        )
        catalog = {name: corpus.snapshot_hash for name, corpus in self.sources.items()}
        adapter = adapter_digest(*MIXTURE_ADAPTER_FILES)
        self._spec = replace(
            training.spec,
            name="t3",
            source_hash=content_hash(
                {
                    "source": training.source.hash,
                    "training_config": self.training_config,
                    "training_sources": catalog,
                }
            ),
            data_hash=content_hash(
                {
                    "training_sources": catalog,
                    "validation": training.corpus.snapshot_hash,
                }
            ),
            evaluator_hash=content_hash(
                {"t1_evaluator": training.spec.evaluator_hash, "adapter": adapter}
            ),
            adapter_hash=adapter,
            mutable_keys=("weights",),
            execution={
                **training.spec.execution,
                "mixture_sampling": "packed-token-batches-v1",
                "training_sources": catalog,
                "validation_snapshot": training.corpus.snapshot_hash,
                "training_config": self.training_config,
            },
        )

    @property
    def training(self) -> T1:
        return self._training

    @property
    def sources(self) -> Mapping[str, Corpus]:
        return self._sources

    @property
    def training_config(self) -> Mapping[str, JSONValue]:
        return self._training_config

    @property
    def spec(self) -> LoopSpec:
        return self._spec

    def normalize_config(
        self, config: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        """Allow only a complete integer weight vector over the frozen catalog."""
        if set(config) != {"weights"}:
            raise ContractError("T3 candidates may change only mixture weights")
        return freeze_object(
            {"weights": normalize_weights(config["weights"], tuple(self.sources))}
        )

    def training_job(self, inputs: ExperimentInputs) -> Job:
        """Translate a validated mixture job into the fixed T1 training program."""
        self.spec.validate_inputs(inputs)
        if self.normalize_config(inputs.config) != inputs.config:
            raise ContractError("Mixture settings must be normalized")
        return self.training.build_job(
            self.training.spec.inputs(
                self.training_config,
                LoopAction(inputs.seed, inputs.fidelity, inputs.phase),
                inputs.environment_hash,
            )
        )

    def build_job(self, inputs: ExperimentInputs) -> Job:
        return Job(inputs, self.training_job(inputs).program)

    def token_allocation(self, inputs: ExperimentInputs) -> Mapping[str, int]:
        """Return the exact trained-token quota after whole-microbatch rounding."""
        self.training_job(inputs)
        device_batch = self.training_config[TrainingSetting.DEVICE_BATCH_SIZE]
        if not isinstance(device_batch, int):
            raise ContractError("Device batch must be an integer")
        quantum = device_batch * SEQUENCE_LENGTH
        weights = normalize_weights(inputs.config["weights"], tuple(self.sources))
        counts = batch_counts(
            weights, batches=inputs.budget_amount // quantum, seed=inputs.seed
        )
        return MappingProxyType(
            {name: count * quantum for name, count in counts.items()}
        )

    async def run(self, job: Job, context: RunContext) -> JobOutput:
        """Run an approved mixture through the same caller-owned executor as T1."""
        if self.build_job(job.inputs) != job:
            raise ContractError("Job program differs from the frozen mixture adapter")
        executor = self.training.executor
        if executor is None:
            raise ContractError("T3.run requires an executor")
        payload = canonical_json(
            {
                "source": self.training.source,
                "corpus_root": str(self.training.corpus.root),
                "tokenizer_dir": str(self.training.tokenizer_dir),
                "sources": {
                    name: str(corpus.root) for name, corpus in self.sources.items()
                },
                "training_config": self.training_config,
                "job": job,
            }
        ).encode("utf-8")
        return await run_worker(
            payload,
            self.training.python,
            executor,
            context,
            module="openloop.loops.t3_worker",
        )

    def evaluate(self, job: Job, output: JobOutput) -> Result:
        """Require exact source-token quotas and T1's trusted scoring receipt."""
        if self.build_job(job.inputs) != job:
            raise ContractError("Job program differs from the frozen mixture adapter")
        allocation = output.get("mixture_tokens")
        if (
            not isinstance(allocation, Mapping)
            or any(type(value) is not int for value in allocation.values())
            or allocation != self.token_allocation(job.inputs)
        ):
            raise ContractError("Worker mixture tokens differ from declared allocation")
        result = self.training.evaluate(self.training_job(job.inputs), output)
        return replace(
            result,
            observations={
                **result.observations,
                **{
                    f"mixture_tokens/{name}": Metric(tokens, "tokens", split="train")
                    for name, tokens in self.token_allocation(job.inputs).items()
                },
            },
        )
