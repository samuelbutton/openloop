"""Check canonical identity, candidate grouping, and input validation."""

from dataclasses import replace

import pytest

from openloop.ledger import (
    IDENTITY_VERSION,
    InvalidInputError,
    Metric,
    Reproducibility,
    canonical_json,
    content_hash,
)


def test_canonical_identity_and_frozen_snapshots(inputs):
    assert canonical_json({"z": 1, "a": [2]}) == '{"a":[2],"z":1}'
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    config = {"layers": [1, {"width": 32}]}
    snapshot = replace(inputs, config=config)
    identity = snapshot.hash
    config["layers"][1]["width"] = 64
    assert snapshot.hash == identity
    with pytest.raises(TypeError):
        snapshot.config["new"] = 1
    with pytest.raises(TypeError):
        snapshot.config["layers"][1]["width"] = 64


def test_golden_hashes(inputs):
    # Changing these literals invalidates every stored identity.
    # Bump IDENTITY_VERSION deliberately when the encoding or a field's meaning changes.
    assert IDENTITY_VERSION == 2
    assert inputs.hash == (
        "276728b9e6f54c7dbdb3c679cab8e159a5d0c202a775b8b7219bfbd00e03e87c"
    )
    assert inputs.candidate_hash == (
        "45685d1ee6a3575e13638b8869adae267d3ce0c591bff1f718f5c940e6568aea"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_hash", "a" * 64),
        ("dependencies_hash", "a" * 64),
        ("data_hash", "a" * 64),
        ("environment_hash", "a" * 64),
        ("evaluator_hash", "a" * 64),
        ("loop_hash", "a" * 64),
        ("tokenizer_hash", "a" * 64),
        ("seed", 43),
        ("fidelity", "confirm"),
        ("budget_unit", "epochs"),
        ("budget_amount", 2048),
        ("reproducibility", Reproducibility.NOISY),
        ("config", {"depth": 5}),
        ("execution", {"device": "gpu"}),
        ("phase", "held_out"),
    ],
)
def test_every_declared_input_changes_identity(inputs, field, value):
    assert replace(inputs, **{field: value}).hash != inputs.hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 43),
        ("fidelity", "confirm"),
        ("phase", "held_out"),
        ("data_hash", "a" * 64),
        ("evaluator_hash", "a" * 64),
        ("environment_hash", "a" * 64),
        ("loop_hash", "a" * 64),
        ("execution", {"device": "gpu"}),
    ],
)
def test_candidate_hash_ignores_experiment_only_inputs(inputs, field, value):
    assert replace(inputs, **{field: value}).candidate_hash == inputs.candidate_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_hash", "a" * 64),
        ("dependencies_hash", "a" * 64),
        ("config", {"depth": 5}),
    ],
)
def test_candidate_hash_follows_code_and_configuration(inputs, field, value):
    assert replace(inputs, **{field: value}).candidate_hash != inputs.candidate_hash


def test_reproducibility_accepts_strings_and_members(inputs):
    noisy = replace(inputs, reproducibility="noisy")
    assert noisy.reproducibility is Reproducibility.NOISY
    assert noisy == replace(inputs, reproducibility=Reproducibility.NOISY)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_inputs_and_results_are_rejected(inputs, value):
    with pytest.raises(InvalidInputError):
        replace(inputs, config={"value": value})
    with pytest.raises(InvalidInputError):
        Metric(value, "BPB")


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": True},
        {"seed": -1},
        {"budget_amount": 0},
        {"source_hash": "unversioned"},
        {"config": {1: "bad key"}},
        {"execution": ["not an object"]},
        {"reproducibility": "sometimes"},
        {"phase": " "},
    ],
)
def test_invalid_inputs_fail_before_admission(inputs, changes):
    with pytest.raises(InvalidInputError) as error:
        replace(inputs, **changes)
    assert isinstance(error.value, ValueError)
