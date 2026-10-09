"""Known optimum, seeded noise, and independent frozen input checks."""

import asyncio
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.ledger import InvalidInputError
from openloop.loops import T0, ContractError, FidelityName, LoopAction, PlantedTruth


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
    first = asyncio.run(t0.run(t0.build_job(inputs)))
    assert asyncio.run(t0.run(t0.build_job(inputs))) == first
    other = replace(inputs, seed=43)
    assert asyncio.run(t0.run(t0.build_job(other))) != first
    full = t0.spec.inputs(config, LoopAction(42, FidelityName.CONFIRM), "a" * 64)
    assert (
        t0.evaluate(t0.build_job(full), asyncio.run(t0.run(t0.build_job(full))))
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
    with pytest.raises(InvalidInputError, match="fidelity"):
        LoopAction(42, fidelity="held_out")
