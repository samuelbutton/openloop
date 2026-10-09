"""T0 adapter: a planted quadratic and seeded Gaussian samples, without file writes.

The oracle is for verification tests. Only sampled loss enters agent observations.
"""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean

from openloop.ledger import (
    ExperimentInputs,
    JSONValue,
    Metric,
    Reproducibility,
    Result,
    content_hash,
)
from openloop.ledger.validation import finite_number, freeze_object

from .files import adapter_digest
from .models import (
    ContractError,
    Fidelity,
    Job,
    JobOutput,
    LoopSpec,
)


@dataclass(frozen=True)
class PlantedTruth:
    optimum: tuple[float, ...] = (1.0, -1.0)
    noise_std: float = 0.1

    def __post_init__(self) -> None:
        if not self.optimum:
            raise ContractError("The planted optimum must have finite coordinates")
        optimum = tuple(finite_number(value, "Optimum") for value in self.optimum)
        noise_std = finite_number(self.noise_std, "Noise standard deviation")
        if noise_std < 0:
            raise ContractError(
                "Noise standard deviation must be finite and nonnegative"
            )
        object.__setattr__(self, "optimum", optimum)
        object.__setattr__(self, "noise_std", noise_std)

    def loss(self, coordinates: tuple[float, ...]) -> float:
        """Return the finite planted objective; its optimum has zero loss."""
        if len(coordinates) != len(self.optimum):
            raise ContractError(
                "Candidate dimension differs from the planted objective"
            )
        loss = sum(
            (value - target) * (value - target)
            for value, target in zip(
                coordinates,
                self.optimum,
                strict=True,
            )
        )
        if not math.isfinite(loss):
            raise ContractError("Candidate objective is not finite")
        return loss


class T0:
    """Known optimum and known Gaussian noise.

    Noise is drawn from the full experiment identity, so identical inputs
    reproduce their samples while other candidates, fidelities, or phases at
    the same seed receive independent draws. A seed alone must not select the
    noise: that would reuse draws across a screen and its confirmation and
    cancel noise in paired comparisons, understating false acceptances.
    """

    def __init__(self, truth: PlantedTruth | None = None) -> None:
        self._truth = truth if truth is not None else PlantedTruth()
        source = adapter_digest("t0.py", "models.py", "files.py")
        self._spec = LoopSpec(
            name="t0",
            source_hash=source,
            adapter_hash=source,
            evaluator_hash=source,
            dependencies_hash=content_hash({"dependencies": "Python standard library"}),
            data_hash=content_hash(self.truth),
            budget_unit="observations",
            reproducibility=Reproducibility.DETERMINISTIC,
            fidelities=(
                Fidelity("1x", 1),
                Fidelity("4x", 4),
            ),
            mutable_keys=("coordinates",),
            metric_name="loss",
            metric_unit="squared distance",
            execution={"device": "cpu", "simulated": True},
        )

    @property
    def truth(self) -> PlantedTruth:
        return self._truth

    @property
    def spec(self) -> LoopSpec:
        return self._spec

    def normalize_config(
        self, config: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        if set(config) != set(self.spec.mutable_keys):
            raise ContractError("T0 permits only the coordinates setting")
        values = config["coordinates"]
        if not isinstance(values, (list, tuple)):
            raise ContractError("Coordinates must be a sequence of numbers")
        coordinates = tuple(finite_number(value, "Coordinate") for value in values)
        self.truth.loss(coordinates)
        return freeze_object({"coordinates": coordinates})

    def build_job(self, inputs: ExperimentInputs) -> Job:
        self.spec.validate_inputs(inputs)
        if self.normalize_config(inputs.config) != inputs.config:
            raise ContractError("Candidate settings must be normalized")
        return Job(inputs)

    async def run(self, job: Job) -> JobOutput:
        self.build_job(job.inputs)
        values = job.inputs.config["coordinates"]
        if not isinstance(values, (list, tuple)):
            raise ContractError("Coordinates must be a sequence")
        coordinates = tuple(finite_number(value, "Coordinate") for value in values)
        truth = self.truth.loss(coordinates)
        rng = random.Random(int(job.inputs.hash, 16))
        return freeze_object(
            {
                "samples": tuple(
                    truth + rng.gauss(0, self.truth.noise_std)
                    for _ in range(job.inputs.budget_amount)
                ),
                "work_units": job.inputs.budget_amount,
            }
        )

    def evaluate(self, job: Job, output: JobOutput) -> Result:
        values = output.get("samples")
        if (
            not isinstance(values, (list, tuple))
            or len(values) != job.inputs.budget_amount
        ):
            raise ContractError("T0 output must contain the declared number of samples")
        if any(type(value) not in (int, float) for value in values):
            raise ContractError("T0 samples must be finite numbers")
        metric = Metric(
            fmean(finite_number(value, "Sample") for value in values),
            self.spec.metric_unit,
            sample_count=len(values),
        )
        return Result({self.spec.metric_name: metric})
