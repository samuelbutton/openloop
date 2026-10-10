"""The power study: statistics helpers, determinism, and report shape."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from openloop.ledger import InvalidInputError
from openloop.studies.t0_decider import (
    ARMS,
    RAW_ARM,
    main,
    validate,
    wilson_interval,
)


def study(
    *, trials: int, effects: Sequence[float], family_size: int = 8
) -> dict[str, Any]:  # Any: the nested JSON report is read back untyped.
    report = asyncio.run(
        validate(trials=trials, effects=effects, family_size=family_size)
    )
    return json.loads(json.dumps(report))


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(0, 100)
    assert low == 0
    assert high == pytest.approx(0.0370, abs=1e-3)
    low, high = wilson_interval(50, 100)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-3)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)


@pytest.mark.parametrize("args", [(0, 0), (-1, 5), (6, 5)])
def test_wilson_interval_rejects_impossible_counts(args: tuple[int, int]) -> None:
    with pytest.raises(InvalidInputError):
        wilson_interval(*args)


def test_study_report_is_simulated_paired_and_deterministic() -> None:
    first = study(trials=3, effects=(0, 10))
    second = study(trials=3, effects=(0, 10))
    assert first == second
    assert first["simulated"] is True
    results = first["results"]
    assert set(results) == {arm.name for arm in ARMS} | {RAW_ARM}
    for arm in ARMS:
        null = results[arm.name]["0"]
        assert "detection" not in null
        assert null["false_verification"]["trials"] == 3
        planted = results[arm.name]["10"]
        # A gain of 1.0 against noise 0.1 is detected by every rule.
        assert planted["detection"]["count"] == 3
        assert planted["worsening_promotions"]["count"] == 0
        assert planted["family_wise_false_confirmation"]["count"] == 0
    assert results[RAW_ARM]["10"]["planted_promoted"]["count"] == 3


def test_all_noise_aware_arms_use_the_same_noise_estimates() -> None:
    results = study(trials=2, effects=(1,))["results"]
    sigmas = {
        arm.name: results[arm.name]["1"]["mean_screen_noise_sigma"] for arm in ARMS
    }
    assert len(set(sigmas.values())) == 1


@pytest.mark.parametrize(
    ("effects", "family_size"),
    [((), 8), ((1, 1.0), 8), ((11,), 8), ((-1,), 8), ((1,), 2), ((1,), 24)],
)
def test_invalid_study_flags(effects: tuple[float, ...], family_size: int) -> None:
    with pytest.raises(InvalidInputError):
        asyncio.run(validate(trials=1, effects=effects, family_size=family_size))


def test_cli_writes_the_same_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    flags = ["--trials", "2", "--effects", "0,1", "--family-size", "5"]
    main([*flags, "--output", str(path)])
    report = json.loads(path.read_text())
    assert report["flags"] == {"trials": 2, "effects": [0.0, 1.0], "family_size": 5}
    assert path.read_text().endswith("}\n")
