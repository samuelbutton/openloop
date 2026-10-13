"""Trusted final evaluation of T4 candidates on held-out scores.

Boundary: this is the only consumer of held-out `RecordedTable`s. Selection-phase
objects (`NASBench201`, `HPOB`, `TabularLoop`) reject those tables and cannot be
built from them. `HeldOutEvaluator` is a `FinalEvaluator` for the decider: it runs
once, after selection closes, and writes ordinary ledger runs in phase `held_out`.
It is not a file-access or oracle security boundary: the host must still keep the
held-out files away from candidate code.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace

from openloop.decider.models import Pair, Sample, StagePlan
from openloop.ledger import (
    ExperimentInputs,
    InvalidInputError,
    InvalidTransitionError,
    JSONValue,
    Ledger,
    Metric,
    Reproducibility,
    Result,
    Stage,
    Status,
)
from openloop.ledger.validation import freeze_object

from .files import adapter_digest
from .models import ContractError, LoopAction, LoopSpec, Phase
from .t4 import TabularLoop
from .tabular import Benchmark, RecordedTable

HELD_OUT_PHASE = "held_out"


class HeldOutEvaluator:
    """Score a selected candidate and the baseline on a held-out table, once.

    The held-out loop adopts the selection loop's candidate identity, which is
    sound only when both tables list identical candidates in the same order. The
    caller maps each candidate hash of the protocol to its configuration here, so
    the table index is never inferred from a hash.
    """

    def __init__(
        self,
        table: RecordedTable,
        selection: TabularLoop,
        ledger: Ledger,
        *,
        baseline: Mapping[str, JSONValue],
        candidates: Mapping[str, Mapping[str, JSONValue]],
    ) -> None:
        if not isinstance(table, RecordedTable) or not table.held_out:
            raise ContractError("The final evaluator requires a held-out table")
        if table.benchmark is not selection.benchmark:
            raise ContractError("Held-out table belongs to a different benchmark")
        if table.candidates != selection.candidates:
            raise ContractError(
                "Held-out candidates differ from the selection candidates; "
                "an index would not name the same candidate"
            )
        if table.benchmark is Benchmark.NAS201:
            task = selection.spec.execution["task"]
            if not isinstance(task, Mapping) or any(
                task[key] != table.provenance[key]
                for key in ("dataset", "artifact_sha256")
            ):
                raise ContractError("Held-out NAS task differs from the selection task")
        self._table = table
        self._ledger = ledger
        self._used = False
        adapter = adapter_digest(
            "tabular.py", "tabular_loaders.py", "t4.py", "t4_final.py", "models.py"
        )
        base = selection.spec
        self._spec = LoopSpec(
            name=f"{base.name}-held-out",
            source_hash=base.source_hash,
            dependencies_hash=base.dependencies_hash,
            data_hash=table.hash,
            evaluator_hash=adapter,
            adapter_hash=adapter,
            budget_unit=base.budget_unit,
            reproducibility=Reproducibility.DETERMINISTIC,
            fidelities=table.fidelities,
            mutable_keys=base.mutable_keys,
            metric_name=base.metric_name,
            metric_unit=base.metric_unit,
            direction=base.direction,
            execution={
                "device": "cpu",
                "recorded_benchmark": True,
                "held_out": True,
                "task": table.provenance,
            },
        )
        self._setting = str(table.setting)
        self._baseline = freeze_object(selection.normalize_config(baseline))
        self._baseline_hash = base.candidate_hash(self._baseline)
        self._candidates: dict[str, Mapping[str, JSONValue]] = {}
        for expected, config in candidates.items():
            frozen = freeze_object(selection.normalize_config(config))
            if base.candidate_hash(frozen) != expected:
                raise ContractError("Candidate hash does not match its configuration")
            self._candidates[expected] = frozen

    @property
    def spec(self) -> LoopSpec:
        return self._spec

    def baseline_template(
        self, *, environment_hash: str, fidelity: str | None = None, seed: int = 0
    ) -> ExperimentInputs:
        """Build the held-out `StagePlan.template` for the frozen baseline."""
        return self._inputs(
            self._baseline,
            LoopAction(seed, fidelity, Phase.CONFIRM),
            environment_hash,
        )

    def _inputs(
        self,
        config: Mapping[str, JSONValue],
        action: LoopAction,
        environment_hash: str,
    ) -> ExperimentInputs:
        return replace(
            self._spec.inputs(config, action, environment_hash), phase=HELD_OUT_PHASE
        )

    async def __call__(self, candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        """Evaluate the whole held-out batch. Any call consumes the permission."""
        if self._used:
            raise InvalidTransitionError("Held-out scores are read once")
        self._used = True
        config = self._candidates.get(candidate)
        if config is None:
            raise InvalidInputError(
                "Candidate has no registered held-out configuration"
            )
        if plan.template.candidate_hash != self._baseline_hash:
            raise ContractError(
                "Held-out baseline differs from the registered baseline"
            )
        if (
            plan.metric.split != HELD_OUT_PHASE
            or plan.metric_name != self._spec.metric_name
            or plan.metric.unit != self._spec.metric_unit
            or plan.metric.direction != self._spec.direction
        ):
            raise ContractError("Held-out plan must declare the held-out table metric")
        return tuple(
            Pair(
                self._sample(plan, self._baseline, seed),
                self._sample(plan, config, seed),
            )
            for seed in plan.seeds
        )

    def _sample(
        self, plan: StagePlan, config: Mapping[str, JSONValue], seed: int
    ) -> Sample:
        template = plan.template
        inputs = replace(template, config=config, seed=seed)
        expected = self._inputs(
            config,
            LoopAction(seed, template.fidelity, Phase.CONFIRM),
            template.environment_hash,
        )
        if inputs != expected:
            raise ContractError(
                "Held-out template differs from the frozen held-out task"
            )
        index = config[self._setting]
        assert isinstance(index, int)
        start = time.perf_counter()
        row = self._table.lookup(index, inputs.fidelity, seed)
        run = self._ledger.submit(
            inputs, hypothesis="Final held-out evaluation, read once"
        )
        if run.reused_from is not None or run.status is not Status.QUEUED:
            raise InvalidTransitionError("Held-out scores are read once")
        try:
            self._ledger.start_stage(run.id, Stage.EXECUTION)
            elapsed = time.perf_counter() - start
            self._ledger.start_stage(run.id, Stage.EVALUATION)
            observations = {
                "lookup_seconds": Metric(elapsed, "seconds", split="execution")
            }
            if row.training_seconds is not None:
                observations["recorded_training_seconds"] = Metric(
                    row.training_seconds, "seconds", split="benchmark"
                )
            run = self._ledger.complete(
                run.id,
                Result(
                    {
                        self._spec.metric_name: Metric(
                            row.score,
                            self._spec.metric_unit,
                            direction=self._spec.direction,
                            split=HELD_OUT_PHASE,
                        )
                    },
                    observations=observations,
                ),
            )
        except BaseException as error:
            with suppress(InvalidTransitionError):
                self._ledger.fail(run.id, f"{type(error).__name__}: {error}")
            raise
        return Sample.from_run(run, self._spec.metric_name)
