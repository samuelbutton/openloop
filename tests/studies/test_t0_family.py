"""Exact families: null losses must equal the baseline loss with no rounding."""

import math
from collections.abc import Mapping

import pytest

from openloop.decider import ConfirmationScope, Policy
from openloop.ledger import InvalidInputError, JSONValue
from openloop.ledger.validation import finite_number
from openloop.studies.t0_family import MAX_FAMILY_SIZE, Arm, T0Study


def loss(study: T0Study, config: Mapping[str, JSONValue]) -> float:
    coordinates = config["coordinates"]
    assert isinstance(coordinates, tuple)
    return study.loop.truth.loss(
        tuple(finite_number(value, "Coordinate") for value in coordinates)
    )


@pytest.mark.parametrize("size", [1, 8, MAX_FAMILY_SIZE])
def test_null_family_has_exactly_equal_planted_losses(size: int) -> None:
    study = T0Study(0, family_size=size)
    assert len(study.configs) == size
    assert study.improvement is None
    for config in (study.baseline, *study.configs.values()):
        assert loss(study, config) == 1


@pytest.mark.parametrize("gain", [0.01, 0.025, 0.05, 0.3, 1.0])
def test_planted_family_changes_only_the_planted_candidates(gain: float) -> None:
    study = T0Study(3, gain=gain, worsening=0.1)
    assert study.improvement is not None
    assert study.worsening is not None
    assert math.isclose(loss(study, study.configs[study.improvement]), 1 - gain)
    assert math.isclose(loss(study, study.configs[study.worsening]), 1.1)
    others = set(study.configs) - {study.improvement, study.worsening}
    assert len(others) == 6
    assert all(loss(study, study.configs[item]) == 1 for item in others)


def test_arms_share_scores_and_differ_only_in_rule() -> None:
    study = T0Study(5, gain=0.05, worsening=0.1)
    first = study.protocol_for(Arm("a"))
    second = study.protocol_for(
        Arm(
            "b", screen_margin_sigmas=2.0, confirmation_scope=ConfirmationScope.SCREENED
        )
    )
    third = study.protocol_for(Arm("c", policy=Policy.AUTOSCIENTISTS))
    assert len({first.hash, second.hash, third.hash}) == 3
    assert first.screen == second.screen == third.screen


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family_size": 0},
        {"family_size": MAX_FAMILY_SIZE + 1},
        {"gain": 0.1},
        {"gain": 0.1, "worsening": 0.1, "family_size": 2},
        {"gain": 0.0, "worsening": 0.1},
        {"gain": 1.5, "worsening": 0.1},
        {"gain": 0.1, "worsening": 0.0},
    ],
)
def test_invalid_families_are_rejected(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(InvalidInputError):
        T0Study(0, **kwargs)  # type: ignore[arg-type]
