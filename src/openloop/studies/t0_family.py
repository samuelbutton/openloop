"""CPU-only T0 comparison families with planted truth. All evidence is simulated.

Boundary: selection scores come from the real T0 `run` and `evaluate` methods.
Held-out scores come from a separate trusted synthetic oracle with its own data
hash; no candidate runner sees it. This is a synthetic split. It does not
validate FineWeb-Edu or any T1 data boundary.
"""

import itertools
import math
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from statistics import fmean

from openloop.decider import (
    Calibration,
    ConfirmationScope,
    DecisionStage,
    Pair,
    Policy,
    Protocol,
    Sample,
    StagePlan,
)
from openloop.ledger import InvalidInputError, JSONValue, Metric, content_hash
from openloop.loops import T0, LoopAction, Phase, PlantedTruth, RunContext

BASELINE_LOSS = 1.0
SCREEN_NOISE_SIGMA = 0.1
MAX_FAMILY_SIZE = 23  # Distinct candidates whose loss is exactly the baseline loss.


@dataclass(frozen=True)
class Arm:
    """One decision rule under test; the name labels reports."""

    name: str
    policy: Policy = Policy.NOISE_AWARE
    screen_margin_sigmas: float = 0.5
    confirmation_scope: ConfirmationScope = ConfirmationScope.FAMILY


def _exact_null_coordinates() -> Iterator[tuple[float, ...]]:
    """Yield coordinates whose squared norm is exactly 1 in floating point.

    Unit vectors and the sign patterns of (0.5,) * 4 only use exact binary
    arithmetic, so a null has no rounding error. The all-positive pattern is
    the baseline itself and is skipped.
    """
    for axis in range(4):
        for sign in (-1.0, 1.0):
            yield tuple(sign if index == axis else 0.0 for index in range(4))
    for signs in itertools.product((-0.5, 0.5), repeat=4):
        if signs != (0.5,) * 4:
            yield signs


class T0Study:
    """An independent trial with a frozen family and shared, cached samples.

    A planted `gain` and `worsening` are loss changes relative to the baseline
    loss of 1, placed on the first axis. Nulls have exactly the baseline loss.
    The planted loss is the square of `sqrt(1 -/+ change)`, so it carries one
    rounding step near 1e-16: far below any gain studied, and absent from nulls.
    Samples depend only on input identity, so arms that share a trial see
    identical scores (a paired comparison).
    """

    def __init__(
        self,
        trial: int,
        *,
        gain: float | None = None,
        worsening: float | None = None,
        family_size: int = 8,
        arm: Arm | None = None,
    ) -> None:
        planted = (gain, worsening) != (None, None)
        if not 1 <= family_size <= MAX_FAMILY_SIZE or (planted and family_size < 3):
            raise InvalidInputError("Family size must give distinct exact nulls")
        if planted and (gain is None or worsening is None):
            raise InvalidInputError("A planted family needs a gain and a worsening")
        if gain is not None and not 0 < gain <= BASELINE_LOSS:
            raise InvalidInputError("A planted gain must be in (0, baseline loss]")
        if worsening is not None and worsening <= 0:
            raise InvalidInputError("A planted worsening must be positive")
        self.arm = arm if arm is not None else Arm("default")
        self.loop = T0(
            PlantedTruth(
                optimum=(0, 0, 0, 0),
                noise_std=SCREEN_NOISE_SIGMA,
            )
        )
        self.environment = content_hash({"decider_t0_trial": trial, "version": 2})
        self.baseline = self.loop.normalize_config({"coordinates": (0.5,) * 4})
        nulls = _exact_null_coordinates()
        coordinates: list[tuple[float, ...]] = []
        if gain is not None and worsening is not None:
            coordinates.append((math.sqrt(BASELINE_LOSS - gain), 0.0, 0.0, 0.0))
            coordinates.append((math.sqrt(BASELINE_LOSS + worsening), 0.0, 0.0, 0.0))
            # Skip the first axis so no null shares a planted axis position.
            for _ in range(2):
                next(nulls)
        coordinates.extend(itertools.islice(nulls, family_size - len(coordinates)))
        configs = [
            self.loop.normalize_config({"coordinates": item}) for item in coordinates
        ]
        self.configs = {
            self.loop.spec.candidate_hash(config): config for config in configs
        }
        hashes = list(self.configs)
        self.improvement = hashes[0] if gain is not None else None
        self.worsening = hashes[1] if worsening is not None else None
        self._samples: dict[str, Sample] = {}
        self._calibrations: dict[DecisionStage, Calibration] = {}
        low = self.loop.spec.inputs(self.baseline, LoopAction(0), self.environment)
        high = self.loop.spec.inputs(
            self.baseline, LoopAction(2, "4x", Phase.CONFIRM), self.environment
        )
        held_out = replace(
            high,
            phase=DecisionStage.HELD_OUT.value,
            data_hash=content_hash({"held_out_truth": self.loop.truth, "version": 1}),
        )
        unit = self.loop.spec.metric_unit
        name = self.loop.spec.metric_name
        self.screen_plan = StagePlan(low, Metric(0, unit), (0, 1), metric_name=name)
        self.confirm_plan = StagePlan(
            high, Metric(0, unit, sample_count=4), tuple(range(2, 18)), metric_name=name
        )
        self.held_out_plan = StagePlan(
            held_out,
            Metric(0, unit, split="held_out", sample_count=4),
            tuple(range(18, 34)),
            metric_name=name,
        )
        self.protocol = self.protocol_for(self.arm)

    def protocol_for(self, arm: Arm) -> Protocol:
        """Build the frozen protocol that applies one arm's rule to this family."""
        return Protocol(
            tuple(self.configs),
            self.screen_plan,
            self.confirm_plan,
            self.held_out_plan,
            tuple(range(34, 44)),
            tuple(range(44, 54)),
            policy=arm.policy,
            screen_margin_sigmas=arm.screen_margin_sigmas,
            confirmation_scope=arm.confirmation_scope,
        )

    async def sample(
        self, plan: StagePlan, seed: int, config: Mapping[str, JSONValue]
    ) -> Sample:
        """Measure through T0, or through the separate final trusted oracle."""
        inputs = replace(plan.template, seed=seed, config=config)
        cached = self._samples.get(inputs.hash)
        if cached is not None:
            return cached
        reference = "simulated:" + inputs.hash
        if plan.metric.split == "held_out":
            values = config["coordinates"]
            if not isinstance(values, tuple):
                raise AssertionError("T0 normalized coordinates must be a tuple")
            coordinates = tuple(
                float(value) for value in values if isinstance(value, (int, float))
            )
            loss = self.loop.truth.loss(coordinates)
            rng = random.Random(int(inputs.hash, 16))
            score = fmean(
                loss + rng.gauss(0, self.loop.truth.noise_std)
                for _ in range(inputs.budget_amount)
            )
            metric = replace(plan.metric, value=score)
        else:
            job = self.loop.build_job(inputs)
            output = await self.loop.run(job, RunContext(reference, reference))
            metric = self.loop.evaluate(job, output).metrics[self.loop.spec.metric_name]
        sample = Sample(
            reference, reference, inputs, metric, metric_name=plan.metric_name
        )
        self._samples[inputs.hash] = sample
        return sample

    async def calibration(self, stage: DecisionStage) -> Calibration:
        """Freeze five baseline seed pairs before candidate comparisons."""
        cached = self._calibrations.get(stage)
        if cached is not None:
            return cached
        screen = stage is DecisionStage.SCREEN
        plan = self.screen_plan if screen else self.confirm_plan
        seeds = self.protocol.screen_noise_seeds
        if not screen:
            seeds = self.protocol.confirm_noise_seeds
        samples = tuple(
            [await self.sample(plan, seed, self.baseline) for seed in seeds]
        )
        calibration = Calibration(tuple(zip(samples[::2], samples[1::2], strict=True)))
        self._calibrations[stage] = calibration
        return calibration

    async def pairs(self, candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        """Return a whole fixed-count batch; final callers must be trusted."""
        return tuple(
            [
                Pair(
                    await self.sample(plan, seed, self.baseline),
                    await self.sample(plan, seed, self.configs[candidate]),
                )
                for seed in plan.seeds
            ]
        )
