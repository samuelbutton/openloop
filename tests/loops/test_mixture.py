"""Token allocations preserve finite budgets and equivalent candidate ratios."""

from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.loops import ContractError
from openloop.loops.mixture import batch_counts, batch_schedule, normalize_weights


@given(
    weights=st.lists(st.integers(0, 100), min_size=1, max_size=8).filter(any),
    batches=st.integers(1, 300),
    seed=st.integers(0, 1000),
)
def test_finite_quota_and_schedule_conserve_tokens(weights, batches, seed) -> None:
    ratios = {str(index): weight for index, weight in enumerate(weights)}
    counts = batch_counts(ratios, batches=batches, seed=seed)
    assert sum(counts.values()) == batches
    observed = Counter(batch_schedule(counts, seed=seed))
    assert dict(observed) == {name: n for name, n in counts.items() if n}
    for name, weight in ratios.items():
        assert abs(counts[name] - batches * weight / sum(weights)) < 1
        if not weight:
            assert counts[name] == 0
    assert list(batch_schedule(counts, seed=seed)) == list(
        batch_schedule(dict(reversed(list(counts.items()))), seed=seed)
    )


def test_scaled_and_reordered_weights_have_one_identity() -> None:
    assert normalize_weights({"a": 2, "b": 6}, ("b", "a")) == {"a": 1, "b": 3}
    assert batch_counts({"a": 1, "b": 3}, batches=100, seed=42) == {"a": 25, "b": 75}


@pytest.mark.parametrize(
    "weights",
    [
        None,
        [],
        {},
        {"a": 0},
        {"a": -1},
        {"a": True},
        {"a": 0.5},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": 1, "extra": 1},
    ],
)
def test_invalid_ratios_are_rejected(weights) -> None:
    with pytest.raises(ContractError):
        normalize_weights(weights, ("a",))


def test_one_hot_and_seeded_remainder_ties() -> None:
    assert batch_counts({"a": 0, "b": 7}, batches=17, seed=2) == {"a": 0, "b": 17}
    results = {
        tuple(batch_counts({"a": 1, "b": 1}, batches=1, seed=seed).values())
        for seed in range(20)
    }
    assert results == {(0, 1), (1, 0)}
