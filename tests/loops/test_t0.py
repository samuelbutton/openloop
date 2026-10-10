"""Known optimum, seeded noise, and independent frozen input checks."""

import asyncio
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.ledger import InvalidInputError
from openloop.loops import (
    T0,
    ContractError,
    LoopAction,
    Phase,
    PlantedTruth,
    RunContext,
)

CONTEXT = RunContext("run", "attempt")


@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False),
        min_size=2,
        max_size=2,
    )
)
def test_planted_optimum_is_global(coordinates: list[float]) -> None:
    truth = PlantedTruth()
    assert truth.loss(tuple(coordinates)) >= truth.loss(truth.optimum) == 0


def test_fixed_seed_and_fidelity_reproduce_samples(t0: T0) -> None:
    config = t0.normalize_config({"coordinates": [0, 0]})
    inputs = t0.spec.inputs(config, LoopAction(42), "a" * 64)
    first = asyncio.run(t0.run(t0.build_job(inputs), CONTEXT))
    assert asyncio.run(t0.run(t0.build_job(inputs), CONTEXT)) == first
    other = replace(inputs, seed=43)
    assert asyncio.run(t0.run(t0.build_job(other), CONTEXT)) != first
    full = t0.spec.inputs(config, LoopAction(42, "4x", Phase.CONFIRM), "a" * 64)
    assert (
        t0.evaluate(
            t0.build_job(full), asyncio.run(t0.run(t0.build_job(full), CONTEXT))
        )
        .metrics["loss"]
        .sample_count
        == 4
    )


@pytest.mark.parametrize(
    "field", ["data_hash", "evaluator_hash", "source_hash", "loop_hash"]
)
def test_frozen_inputs_cannot_change(t0: T0, field: str) -> None:
    config = t0.normalize_config({"coordinates": [0, 0]})
    inputs = t0.spec.inputs(config, LoopAction(42), "a" * 64)
    with pytest.raises(ContractError, match="frozen"):
        t0.build_job(replace(inputs, **{field: "b" * 64}))


@pytest.mark.parametrize(
    "config",
    [
        {"coordinates": [0]},
        {"coordinates": [0, float("nan")]},
        {"coordinates": [True, 1]},
        {"coordinates": [0, 0], "score": 0},
    ],
)
def test_invalid_candidate_is_rejected(t0: T0, config) -> None:
    with pytest.raises(InvalidInputError):
        t0.normalize_config(config)


def test_held_out_phase_is_not_a_selection_action() -> None:
    with pytest.raises(InvalidInputError, match="phase"):
        LoopAction(42, phase="held_out")


def sample_residuals(t0: T0, inputs) -> list[float]:
    job = t0.build_job(inputs)
    samples = asyncio.run(t0.run(job, CONTEXT))["samples"]
    assert isinstance(samples, tuple)
    coordinates = inputs.config["coordinates"]
    assert isinstance(coordinates, tuple)
    truth = t0.truth.loss(tuple(float(value) for value in coordinates))  # pyright: ignore[reportArgumentType]
    return [float(sample) - truth for sample in samples]  # pyright: ignore[reportArgumentType]


def test_candidates_at_one_seed_receive_independent_noise(t0: T0) -> None:
    first = t0.spec.inputs(
        t0.normalize_config({"coordinates": [0, 0]}), LoopAction(42), "a" * 64
    )
    second = t0.spec.inputs(
        t0.normalize_config({"coordinates": [0.5, 0]}), LoopAction(42), "a" * 64
    )
    assert sample_residuals(t0, first) == sample_residuals(t0, first)
    assert sample_residuals(t0, first) != sample_residuals(t0, second)


def test_fidelities_and_phases_at_one_seed_share_no_draws(t0: T0) -> None:
    config = t0.normalize_config({"coordinates": [0, 0]})
    low = sample_residuals(t0, t0.spec.inputs(config, LoopAction(42), "a" * 64))
    high = sample_residuals(
        t0, t0.spec.inputs(config, LoopAction(42, "4x", Phase.CONFIRM), "a" * 64)
    )
    same_fidelity_other_phase = sample_residuals(
        t0, t0.spec.inputs(config, LoopAction(42, phase=Phase.CONFIRM), "a" * 64)
    )
    assert len(low) == 1
    assert len(high) == 4
    assert low[0] not in high
    assert low != same_fidelity_other_phase
