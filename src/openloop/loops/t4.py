"""T4 lookup workloads behind the shared contract; no training or network calls.

The trusted evaluator selects scores from frozen rows. Runners cannot submit
their own score. These objects are not file-access or oracle security boundaries.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
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
from openloop.ledger.validation import finite_number, freeze_object

from .files import adapter_digest
from .models import ContractError, Job, JobOutput, LoopSpec, RunContext
from .tabular import Benchmark, MetaSplit, NASDataset, RecordedTable


class TabularLoop:
    """One fixed benchmark task. Only the candidate's table index is mutable."""

    _benchmark: Benchmark | None = None

    def __init__(self, table: RecordedTable) -> None:
        if self._benchmark is not None and table.benchmark is not self._benchmark:
            raise ContractError("Table belongs to a different benchmark")
        self._table = table
        adapter = adapter_digest(
            "tabular.py", "tabular_loaders.py", "t4.py", "models.py", "files.py"
        )
        self._spec = LoopSpec(
            name=f"t4-{table.benchmark}",
            source_hash=content_hash(
                {
                    "adapter": adapter,
                    "benchmark": table.benchmark,
                    "task": table.provenance,
                    "candidates": table.candidates,
                }
            ),
            adapter_hash=adapter,
            evaluator_hash=adapter,
            dependencies_hash=table.dependencies_hash,
            data_hash=table.hash,
            budget_unit="epochs" if table.benchmark is Benchmark.NAS201 else "lookups",
            reproducibility=Reproducibility.DETERMINISTIC,
            fidelities=table.fidelities,
            mutable_keys=(str(table.setting),),
            metric_name="validation_score",
            metric_unit="percent"
            if table.benchmark is Benchmark.NAS201
            else "normalized response",
            direction=Direction.MAXIMIZE,
            execution={
                "device": "cpu",
                "recorded_benchmark": True,
                "task": table.provenance,
            },
        )

    @property
    def spec(self) -> LoopSpec:
        return self._spec

    @property
    def candidates(self) -> tuple[JSONValue, ...]:
        """Candidate features in index order; no scores are exposed here."""
        return self._table.candidates

    def seeds(self, index: int, *, fidelity: str | None = None) -> tuple[int, ...]:
        self.normalize_config({str(self._table.setting): index})
        return self._table.seeds(index, self.spec.fidelity(fidelity).name)

    def normalize_config(
        self, config: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        if set(config) != set(self.spec.mutable_keys):
            raise ContractError("T4 permits only its declared candidate index")
        index = config[self._table.setting]
        if type(index) is not int or not 0 <= index < len(self.candidates):
            raise ContractError("Candidate index is outside the frozen table")
        return freeze_object({str(self._table.setting): index})

    def build_job(self, inputs: ExperimentInputs) -> Job:
        self.spec.validate_inputs(inputs)
        config = self.normalize_config(inputs.config)
        index = config[self._table.setting]
        assert isinstance(index, int)
        self._table.lookup(index, inputs.fidelity, inputs.seed)
        return Job(inputs)

    async def run(self, job: Job, _context: RunContext) -> JobOutput:
        start = time.perf_counter()
        if self.build_job(job.inputs) != job:
            raise ContractError("T4 jobs cannot contain candidate programs")
        return freeze_object(
            {
                "work_units": job.inputs.budget_amount,
                "lookup_seconds": time.perf_counter() - start,
            }
        )

    def evaluate(self, job: Job, output: JobOutput) -> Result:
        if self.build_job(job.inputs) != job:
            raise ContractError("T4 evaluator requires the exact declared job")
        if (
            set(output) != {"work_units", "lookup_seconds"}
            or type(output["work_units"]) is not int
            or output["work_units"] != job.inputs.budget_amount
        ):
            raise ContractError(
                "T4 output must report exactly the declared lookup work"
            )
        elapsed = finite_number(output["lookup_seconds"], "Lookup seconds")
        if elapsed < 0:
            raise ContractError("Lookup seconds must be nonnegative")
        index = job.inputs.config[self._table.setting]
        assert isinstance(index, int)
        row = self._table.lookup(index, job.inputs.fidelity, job.inputs.seed)
        observations = {"lookup_seconds": Metric(elapsed, "seconds", split="execution")}
        if row.training_seconds is not None:
            observations["recorded_training_seconds"] = Metric(
                row.training_seconds, "seconds", split="benchmark"
            )
        return Result(
            {
                self.spec.metric_name: Metric(
                    row.score, self.spec.metric_unit, direction=self.spec.direction
                )
            },
            observations=observations,
        )


class NASBench201(TabularLoop):
    """NAS-Bench-201 validation-only architecture lookups at 12 and 200 epochs."""

    _benchmark = Benchmark.NAS201

    @classmethod
    def load(
        cls, data_root: Path, *, dataset: NASDataset | str = NASDataset.CIFAR10
    ) -> NASBench201:
        from .tabular_loaders import load_nas201

        return cls(load_nas201(data_root, dataset=dataset))


class HPOB(TabularLoop):
    """One HPO-B task from meta-train or meta-validation; no meta-test tasks."""

    _benchmark = Benchmark.HPOB

    @classmethod
    def load(
        cls,
        data_root: Path,
        search_space: str,
        dataset: str,
        *,
        split: MetaSplit | str = MetaSplit.VALIDATION,
    ) -> HPOB:
        from .tabular_loaders import load_hpob

        return cls(load_hpob(data_root, search_space, dataset, split=split))
