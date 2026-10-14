"""Amendment 3 script boundaries: watched runs and valid-only analysis; no GPU."""

import asyncio
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from openloop.executors import Limits, LocalExecutor, ProcessJob, State
from openloop.studies.claims_audit import MemoryReading

SCRIPT = Path(__file__).parents[2] / "research/audit_heldout_valid.py"
spec = importlib.util.spec_from_file_location("audit_heldout_valid", SCRIPT)
assert spec is not None
assert spec.loader is not None
valid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(valid)


def _job(tmp_path: Path, seconds: int) -> ProcessJob:
    return ProcessJob(
        ("/bin/sleep", str(seconds)),
        cwd=tmp_path,
        environment={"PATH": "/usr/bin:/bin"},
        reviewed=True,
        limits=Limits(wall_seconds=60, memory_bytes=1024**3, output_bytes=1024**2),
    )


def _calm() -> MemoryReading:
    return MemoryReading("t", 1, 80)


def _watch(
    tmp_path: Path,
    seconds: int,
    *,
    read: Callable[[], MemoryReading] = _calm,
    jumps: dict[int, float] | None = None,
):
    calls = {"n": 0}
    offset = {"wall": 0.0}
    mono = {"t": 0.0}

    def monotonic() -> float:
        mono["t"] += 1.0
        return mono["t"]

    def wall() -> float:
        calls["n"] += 1
        offset["wall"] += (jumps or {}).get(calls["n"], 0.0)
        return mono["t"] + offset["wall"]

    async def go():
        executor = LocalExecutor(tmp_path / "executor", memory_sample_seconds=1)
        try:
            return await valid.run_watched(
                executor,
                _job(tmp_path, seconds),
                key="k",
                read=read,
                interval=0.05,
                wall=wall,
                monotonic=monotonic,
            )
        finally:
            await executor.close()

    return asyncio.run(go())


def test_sleep_above_threshold_cancels_the_job(tmp_path: Path) -> None:
    result = _watch(tmp_path, 30, jumps={3: 40.0})
    assert result.drift_trip is not None
    assert result.memory_trip is None
    assert result.snapshot.state == State.CANCELLED
    assert result.drift_seconds == pytest.approx(40.0)
    assert "invalid: machine sleep" in result.drift_trip.reason


def test_sleep_below_threshold_lets_the_job_finish(tmp_path: Path) -> None:
    result = _watch(tmp_path, 1, jumps={3: 4.0})
    assert result.drift_trip is None
    assert result.snapshot.state == State.SUCCEEDED
    assert result.drift_seconds == pytest.approx(4.0)


def test_memory_guard_still_cancels_the_job(tmp_path: Path) -> None:
    readings = iter([MemoryReading("t", 2, 50), MemoryReading("t", 4, 50)])
    result = _watch(
        tmp_path, 30, read=lambda: next(readings, MemoryReading("t", 4, 50))
    )
    assert result.memory_trip is not None
    assert "critical" in result.memory_trip.reason
    assert result.snapshot.state == State.CANCELLED


def test_unreadable_memory_stops_the_job(tmp_path: Path) -> None:
    def broken() -> MemoryReading:
        raise OSError("no sysctl")

    result = _watch(tmp_path, 30, read=broken)
    assert result.memory_trip is not None


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_analysis_excludes_invalid_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original"
    output = tmp_path / "output"
    original.mkdir()
    output.mkdir()
    comparison = {
        "child": "k",
        "parent": "p",
        "claimed": "not_win",
        "published_child_bpb": 1.43,
        "published_parent_bpb": 1.42,
    }
    _write(original / "plan.json", {"plan": {"comparisons": [comparison]}})
    summary = {"num_steps": 800}
    for seed in range(2000, 2005):
        _write(
            original / f"run-p-{seed}.json",
            {"arm": "p", "seed": seed, "summary": {**summary, "val_bpb": 1.4}},
        )
        if seed != 2004:
            _write(
                original / f"run-k-{seed}.json",
                {"arm": "k", "seed": seed, "summary": {**summary, "val_bpb": 1.39}},
            )
    _write(
        output / "run-01-seed_completion-k-s2004-a1.json",
        {
            "kind": "seed_completion",
            "arm": "k",
            "seed": 2004,
            "summary": {**summary, "val_bpb": 1.39},
            "validity": {"valid": True},
        },
    )
    for seed in valid.base.HELDOUT_SEEDS:
        for arm, offset in (("1d83776", 0.0), ("bc5cdc1", 0.01)):
            # An invalid first attempt with a wildly different score is ignored,
            # even though it carries a summary.
            _write(
                output / f"run-{arm}-{seed}-a1.json",
                {
                    "kind": "heldout",
                    "arm": arm,
                    "seed": seed,
                    "summary": {
                        **summary,
                        "val_bpb": 9.0,
                        "heldout_bpb": 9.0 - offset * 100,
                    },
                    "validity": {"valid": False, "reason": "x"},
                },
            )
            _write(
                output / f"run-{arm}-{seed}-a2.json",
                {
                    "kind": "heldout",
                    "arm": arm,
                    "seed": seed,
                    "summary": {
                        **summary,
                        "val_bpb": 1.40 + offset,
                        "heldout_bpb": 1.41 + offset + seed / 1e7,
                    },
                    "validity": {"valid": True},
                },
            )
    monkeypatch.setattr(valid.base, "PARENT_RUN", original)
    valid.write_reports(output, {})
    held = json.loads((output / valid.base.HELDOUT_REPORT).read_text())
    assert held["finding"] == "replicated"
    assert held["heldout"]["seeds"] == list(valid.base.HELDOUT_SEEDS)
    assert held["heldout"]["gains"] == pytest.approx([0.01] * 5)
    attempts = json.loads((output / valid.ATTEMPTS_REPORT).read_text())
    assert attempts["excluded_invalid"] == 10
