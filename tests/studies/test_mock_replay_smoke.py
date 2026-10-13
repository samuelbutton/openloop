"""The mock-replay smoke report regenerates deterministically from receipts."""

import asyncio
import json
from pathlib import Path

import pytest

from openloop.executors import InvalidJobError
from openloop.studies import mock_replay_smoke as smoke

ROOT = Path(__file__).resolve().parents[2]


def write_receipt(
    directory: Path, number: int, *, config: int, fidelity: str, seconds: float
) -> None:
    run = {
        "status": "succeeded",
        "inputs": {"config": {"DEPTH": config}, "fidelity": fidelity},
        "result": {
            "metrics": {"val_bpb": {"value": 2.0 + seconds / 100, "unit": "BPB"}},
            "artifacts": {},
            "observations": {
                "process_seconds": {
                    "value": seconds,
                    "unit": "seconds",
                    "split": "execution",
                }
            },
        },
    }
    (directory / f"run-{number:03d}.json").write_text(json.dumps({"run": run}))


@pytest.fixture
def receipts(tmp_path: Path) -> Path:
    for number, (config, seconds) in enumerate(
        [(1, 10), (1, 12), (1, 15), (2, 20), (2, 21)], start=1
    ):
        write_receipt(tmp_path, number, config=config, fidelity="8m", seconds=seconds)
    return tmp_path


def report(source: Path, **options: object) -> dict[str, object]:
    return asyncio.run(smoke.build_report(source, workers=(1, 3), **options))  # type: ignore[arg-type]


def test_report_is_deterministic_simulated_and_records_provenance(
    receipts: Path,
) -> None:
    first = report(receipts, seed=5)
    assert smoke.render(first) == smoke.render(report(receipts, seed=5))
    assert first["simulated"] is True
    assert first["pool_count"] == 2
    assert first["recording_count"] == 5
    assert len(str(first["source_receipts_sha256"])) == 64
    assert first["replay"] == {
        "mode": "resample",
        "seed": 5,
        "pool": "candidate config hash and fidelity; seed is not part of the pool",
        "duration_metric": "process_seconds",
    }
    write_receipt(receipts, 6, config=3, fidelity="8m", seconds=1)
    assert (
        report(receipts, seed=5)["source_receipts_sha256"]
        != (first["source_receipts_sha256"])
    )


def test_replay_seed_changes_draws_and_strict_mode_replays_exactly(
    receipts: Path,
) -> None:
    seeds = {smoke.render(report(receipts, seed=seed)) for seed in range(8)}
    assert len(seeds) > 1
    exact = report(receipts, strict=True)
    scenarios = exact["scenarios"]
    assert isinstance(scenarios, list)
    assert scenarios[0]["resampled_submissions"] == 0
    assert scenarios[0]["distinct_recordings_served"] == 5
    assert scenarios[0]["simulated_elapsed_seconds"] == 10 + 12 + 15 + 20 + 21


def test_non_succeeded_or_empty_sources_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidJobError, match="No run"):
        report(tmp_path)
    (tmp_path / "run-001.json").write_text(json.dumps({"run": {"status": "failed"}}))
    with pytest.raises(InvalidJobError, match="not a succeeded"):
        report(tmp_path)


def test_check_flag_detects_drift(receipts: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    arguments = ["--source", str(receipts), "--output", str(output)]
    assert smoke.main([*arguments, "--workers", "1", "2"]) == 0
    assert smoke.main([*arguments, "--workers", "1", "2", "--check"]) == 0
    output.write_text(output.read_text().replace("true", "false", 1))
    assert smoke.main([*arguments, "--workers", "1", "2", "--check"]) == 1


def test_committed_report_matches_a_fresh_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (ROOT / smoke.DEFAULT_SOURCE).exists():
        pytest.skip("S1 receipts are not present")
    monkeypatch.chdir(ROOT)  # The report records its source root as given.
    assert smoke.main(["--check"]) == 0
