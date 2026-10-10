"""Exact gate boundaries, conservative sign inference, and separate noise kinds."""

import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.decider import (
    Calibration,
    NoiseKind,
    Verdict,
    autoscientists_gate,
    median_interval,
    sign_p_value,
)
from openloop.ledger import InvalidInputError


@pytest.mark.parametrize(
    ("gain", "sigma", "second", "expected"),
    [
        (-0.1, 0.1, 1.0, Verdict.DISCARD),
        (0.0, 0.1, 1.0, Verdict.DISCARD),
        (0.200001, 0.1, None, Verdict.PROMOTE),
        (0.2, 0.1, None, Verdict.INCONCLUSIVE),
        (0.2, 0.1, 0.01, Verdict.PROMOTE),
        (0.01, 0.1, 0.0, Verdict.DISCARD),
        (0.01, 0.1, -0.1, Verdict.DISCARD),
        (0.01, 0.0, None, Verdict.PROMOTE),
    ],
)
def test_published_gate(gain, sigma, second, expected) -> None:
    assert autoscientists_gate(gain, sigma, second_gain=second) is expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_gate_rejects_invalid_measurement(value) -> None:
    with pytest.raises(InvalidInputError):
        autoscientists_gate(value, 0.1)
    with pytest.raises(InvalidInputError):
        autoscientists_gate(1, value)


def test_noise_formula_and_lock(low_noise: Calibration) -> None:
    pairs = tuple(
        (
            replace(left, metric=replace(left.metric, value=math.sqrt(2) * 0.1)),
            replace(right, metric=replace(right.metric, value=0)),
        )
        for left, right in low_noise.pairs
    )
    partial = Calibration(pairs[:3])
    assert not partial.locked
    assert partial.sigma == pytest.approx(0.1)
    assert Calibration(pairs).locked
    with pytest.raises(InvalidInputError, match="three to five"):
        Calibration(pairs[:2])
    with pytest.raises(InvalidInputError, match="independent"):
        Calibration((pairs[0], pairs[0], pairs[0]))


def test_fixed_seed_noise_cannot_be_distinct_seed_noise(low_noise: Calibration) -> None:
    original = low_noise.samples[0]
    samples = tuple(
        replace(original, reference=f"repeat-{i}", attempt_id=f"repeat-{i}")
        for i in range(10)
    )
    pairs = tuple(zip(samples[::2], samples[1::2], strict=True))
    fixed = Calibration(pairs, kind=NoiseKind.FIXED_SEED)
    assert fixed.sigma == 0
    with pytest.raises(InvalidInputError, match="independent"):
        Calibration(pairs)


def test_exact_test_and_interval() -> None:
    assert sign_p_value((1.0,) * 16) == 2**-16
    assert sign_p_value((0.0,) * 16) == 1
    assert sign_p_value((0.1,) * 16, minimum_effect=0.1) == 1
    assert sign_p_value(()) == 1
    assert median_interval((1.0,) * 16, alpha=0.00625) == (1, 1)
    assert median_interval((1.0,)) == (None, None)


@given(
    st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=1, max_size=30
    )
)
def test_greater_gains_cannot_increase_p_value(values: list[float]) -> None:
    gains = tuple(values)
    assert 0 <= sign_p_value(gains) <= 1
    assert sign_p_value(tuple(value + 1 for value in gains)) <= sign_p_value(gains)
    low, high = median_interval(gains)
    assert (low is None and high is None) or (
        low is not None and high is not None and low <= high
    )
