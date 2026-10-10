"""Independent CPU-only T0 trials and synthetic unit-test scores."""

import asyncio
from dataclasses import replace

import pytest

from openloop.decider import Calibration, DecisionStage, Pair, StagePlan
from openloop.studies.t0_family import T0Study


@pytest.fixture
def study() -> T0Study:
    return T0Study(99, gain=1.0, worsening=3.0)


@pytest.fixture
def low_noise(study: T0Study) -> Calibration:
    return asyncio.run(study.calibration(DecisionStage.SCREEN))


@pytest.fixture
def high_noise(study: T0Study) -> Calibration:
    return asyncio.run(study.calibration(DecisionStage.CONFIRM))


def scored_pairs(
    study: T0Study,
    plan: StagePlan,
    gains: tuple[float, ...],
    candidate: str | None = None,
) -> tuple[Pair, ...]:
    """Supply controlled scores, never persisted as observed ledger evidence."""
    candidate = candidate or study.improvement
    assert candidate is not None
    pairs = asyncio.run(study.pairs(candidate, plan))
    return tuple(
        Pair(
            replace(pair.baseline, metric=replace(pair.baseline.metric, value=0)),
            replace(pair.candidate, metric=replace(pair.candidate.metric, value=-gain)),
        )
        for pair, gain in zip(pairs, gains, strict=False)
    )
