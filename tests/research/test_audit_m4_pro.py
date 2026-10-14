"""S1 part 2 audit script boundaries; no training and no native MLX imports."""

import asyncio
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from openloop.executors import Limits, LocalExecutor, ProcessJob, Snapshot, State
from openloop.ledger import ExperimentInputs, Ledger, Purpose, Reproducibility, Stage
from openloop.studies.claims_audit import GuardTrip, MemoryReading

SCRIPT = Path(__file__).parents[2] / "research/audit_m4_pro.py"
spec = importlib.util.spec_from_file_location("audit_m4_pro", SCRIPT)
assert spec is not None
assert spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

PROGRAM = b"import x\nmx.random.seed(42)\nsoftcap = 15\n"


def test_evidence_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    audit.save(path, {"a": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        audit.save(path, {"a": 2})
    assert path.read_bytes() == before


def test_registered_comparisons_use_known_arms() -> None:
    for child, parent, _ in audit.COMPARISONS:
        assert child in audit.ARMS
        assert parent in audit.ARMS
    assert len(audit.COMPARISONS) == 5
    assert len(audit.ARMS) * len(audit.SEEDS) == 30
    assert 42 not in audit.SEEDS


def test_materialize_changes_only_the_seed_call(tmp_path: Path) -> None:
    files = {"train_mlx.py": PROGRAM, "prepare.py": b"TIME_BUDGET = 300\n"}
    digest = audit.materialize(files, 2003, tmp_path)
    assert digest == audit.sha((tmp_path / "train_mlx.py").read_bytes())
    assert (tmp_path / "train_mlx.py").read_text() == PROGRAM.decode().replace(
        "seed(42)", "seed(2003)"
    )
    assert (tmp_path / "prepare.py").read_bytes() == files["prepare.py"]
    with pytest.raises(FileExistsError):
        audit.materialize(files, 2003, tmp_path)


def _fake_job(tmp_path: Path, seconds: int) -> ProcessJob:
    return ProcessJob(
        ("/bin/sleep", str(seconds)),
        cwd=tmp_path,
        environment={"PATH": "/usr/bin:/bin"},
        reviewed=True,
        limits=Limits(wall_seconds=60, memory_bytes=1024**3, output_bytes=1024**2),
    )


def _reader(levels_and_free: list[tuple[int, int]]) -> Callable[[], MemoryReading]:
    samples = iter(levels_and_free)

    def read() -> MemoryReading:
        level, free = next(samples, (1, 80))
        return MemoryReading("t", level, free)

    return read


def _guarded(
    tmp_path: Path, seconds: int, read: Callable[[], MemoryReading], interval: float
) -> tuple[Snapshot, GuardTrip | None]:
    async def go() -> tuple[Snapshot, GuardTrip | None]:
        executor = LocalExecutor(tmp_path / "executor", memory_sample_seconds=1)
        try:
            return await audit.run_guarded(
                executor,
                _fake_job(tmp_path, seconds),
                key="k",
                read=read,
                interval=interval,
            )
        finally:
            await executor.close()

    return asyncio.run(go())


def test_guard_cancels_the_job_on_critical_pressure(tmp_path: Path) -> None:
    snapshot, trip = _guarded(tmp_path, 30, _reader([(2, 50), (4, 50)]), 0.05)
    assert trip is not None
    assert "critical" in trip.reason
    assert snapshot.state == State.CANCELLED
    assert snapshot.elapsed_seconds is not None
    assert snapshot.elapsed_seconds < 10


def test_guard_cancels_on_two_consecutive_low_free_samples(tmp_path: Path) -> None:
    snapshot, trip = _guarded(tmp_path, 30, _reader([(2, 10), (2, 9)]), 0.05)
    assert trip is not None
    assert snapshot.state == State.CANCELLED


def test_guard_lets_a_job_finish_at_warning_and_single_low_sample(
    tmp_path: Path,
) -> None:
    read = _reader([(2, 60), (2, 10), (2, 60), (2, 30)] * 3)
    snapshot, trip = _guarded(tmp_path, 1, read, 0.1)
    assert trip is None
    assert snapshot.state == State.SUCCEEDED


def test_unreadable_pressure_stops_the_job(tmp_path: Path) -> None:
    def broken() -> MemoryReading:
        raise OSError("no sysctl")

    snapshot, trip = _guarded(tmp_path, 30, broken, 0.05)
    assert trip is not None
    assert snapshot.state == State.CANCELLED


def test_trip_records_failure_stop_file_and_raises(tmp_path: Path) -> None:
    hashes = dict.fromkeys(
        (
            "source_hash",
            "dependencies_hash",
            "data_hash",
            "environment_hash",
            "evaluator_hash",
            "loop_hash",
        ),
        "0" * 64,
    )
    trip = GuardTrip("critical memory pressure", (MemoryReading("t", 4, 9),))
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        run = ledger.submit(
            ExperimentInputs(
                **hashes,
                seed=2000,
                fidelity="300s",
                budget_unit="training_seconds",
                budget_amount=300,
                reproducibility=Reproducibility.NOISY,
                config={"arm": "bc5cdc1"},
                execution={"slot": 2},
            ),
            hypothesis="fake",
            request_key="fake",
            purpose=Purpose.REPLICATION,
        )
        ledger.start_stage(run.id, Stage.EXECUTION)
        record: dict[str, object] = {"index": 2, "arm": "bc5cdc1", "seed": 2000}
        receipt = tmp_path / "run-02-bc5cdc1-s2000.json"
        with pytest.raises(audit.MemoryGuardTripped):
            audit.stop_for_memory(tmp_path, receipt, ledger, run.id, record, trip)
        assert ledger.get(run.id).status == "failed"
    stop = json.loads((tmp_path / "memory-guard-stop.json").read_text())
    assert stop["run"]["arm"] == "bc5cdc1"
    assert stop["run"]["index"] == 2
    assert stop["run"]["seed"] == 2000
    assert stop["readings"][0]["free_percent"] == 9
    assert stop["threshold"]["free_percent_floor"] == 15
    assert stop["automatic_retry"] is False
    assert "memory guard" in json.loads(receipt.read_text())["failure"]
