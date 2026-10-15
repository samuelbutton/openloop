"""Portable training contracts and scalar GRPO math need no model runtime."""

from math import e

import pytest

from openloop.training import (
    Datum,
    LossConfig,
    Objective,
    SamplingParams,
    TrainingError,
    group_advantages,
)
from openloop.training.loss import reference_loss


def test_group_standardization_and_zero_variance() -> None:
    assert group_advantages((0, 0, 0, 0)) == (0, 0, 0, 0)
    assert group_advantages((0, 1)) == pytest.approx((-1, 1), abs=3e-6)


@pytest.mark.parametrize(
    ("tokens", "start"), [([], 0), ([0], 1), ([0, 1], 0), ([True, 1], 1)]
)
def test_invalid_datum(tokens, start) -> None:
    with pytest.raises(TrainingError):
        Datum(tokens, start)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": 257},
        {"num_samples": True},
        {"temperature": 0.8},
    ],
)
def test_sampling_limits(kwargs) -> None:
    with pytest.raises(TrainingError):
        SamplingParams(**kwargs)


def test_grpo_rejects_missing_or_misaligned_rollout_statistics() -> None:
    with pytest.raises(TrainingError):
        Datum((0, 1, 2), 1, old_logprobs=(-1,))
    with pytest.raises(TrainingError):
        LossConfig().validate((Datum((0, 1), 1),))


def test_loss_uses_equal_sequence_weight_and_signed_clipping() -> None:
    rows = (Datum((0, 1, 2), 1), Datum((0, 1), 1))
    loss = reference_loss(
        ((-2.0, -4.0), (-1.0,)), rows, LossConfig(Objective.CROSS_ENTROPY)
    )
    assert loss == 2.0
    policy_rows = (
        Datum((0, 1), 1, 2.0, (-2.0,)),
        Datum((0, 1), 1, -2.0, (-2.0,)),
    )
    clipped = reference_loss(((-1.0,), (-1.0,)), policy_rows, LossConfig(kl_beta=0))
    assert clipped == pytest.approx((-2.4 + 2 * e) / 2)
