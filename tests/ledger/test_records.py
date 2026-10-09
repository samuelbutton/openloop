"""Check typed results and pure comparison."""

from dataclasses import replace

import pytest

from openloop.ledger import (
    Direction,
    InvalidInputError,
    Metric,
    Result,
    Stage,
    compare_results,
)


def test_comparison_tolerances_and_metadata(result):
    slightly_changed = replace(result, metrics={"val_bpb": Metric(1.751, "BPB")})
    assert compare_results(result, slightly_changed, atol=0.002) == ()
    assert compare_results(result, slightly_changed, rtol=0.001) == ()
    assert compare_results(result, slightly_changed)
    metadata_changed = replace(result, metrics={"val_bpb": Metric(1.75, "nats")})
    assert compare_results(result, metadata_changed) == (
        "metric val_bpb: metadata differs",
    )
    artifact_changed = replace(result, artifacts={"checkpoint": "8" * 64})
    assert compare_results(result, artifact_changed) == (
        "artifact checkpoint: hash differs",
    )
    missing_metric = Result({"accuracy": Metric(0.5, "fraction", Direction.MAXIMIZE)})
    assert "metric val_bpb: missing or extra" in compare_results(result, missing_metric)
    for tolerance in (-1, float("nan"), float("inf"), True):
        with pytest.raises(InvalidInputError):
            compare_results(result, result, atol=tolerance)


def test_enum_values_accept_strings_and_reject_unknowns():
    # Strings are accepted at the boundary, so these calls pass plain text.
    assert (
        Metric(1, "x", "maximize").direction  # pyright: ignore[reportArgumentType]
        is Direction.MAXIMIZE
    )
    # The ignores below pass deliberately invalid values to test validation.
    with pytest.raises(InvalidInputError, match="direction"):
        Metric(1, "x", "sideways")  # pyright: ignore[reportArgumentType]
    assert [stage.value for stage in Stage] == ["queue", "execution", "evaluation"]


def test_result_validation_and_immutability():
    # The ignores below pass deliberately invalid values to test validation.
    with pytest.raises(InvalidInputError):
        Result({})
    with pytest.raises(InvalidInputError):
        Result({"score": 1.0})  # pyright: ignore[reportArgumentType]
    with pytest.raises(InvalidInputError):
        Result({"score": Metric(1, "x")}, {"model": "not a digest"})
    result = Result({"score": Metric(1, "x")})
    with pytest.raises(TypeError):
        result.metrics["other"] = Metric(2, "x")  # pyright: ignore[reportIndexIssue]
