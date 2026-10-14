"""Amendment 4 script boundaries: verification, pending slots, combined reports."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from openloop.executors import Limits, LocalExecutor, ProcessJob, State
from openloop.studies.claims_audit import MemoryReading
from openloop.studies.resume import ResumeMismatchError, select_retained

SCRIPT = Path(__file__).parents[2] / "research/audit_heldout_resume.py"
spec = importlib.util.spec_from_file_location("audit_heldout_resume", SCRIPT)
assert spec is not None
assert spec.loader is not None
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)

ARMS = ("1d83776", "bc5cdc1")
SEEDS = (3000, 3001, 3002)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _order() -> list[dict[str, object]]:
    slots = [(arm, seed) for seed in SEEDS for arm in ARMS]
    return [
        {
            "index": i,
            "kind": "heldout",
            "arm": arm,
            "seed": seed,
            "base_program_sha256": f"b{i}",
            "program_sha256": f"p{i}",
        }
        for i, (arm, seed) in enumerate(slots, start=1)
    ]


def _receipt(
    root: Path, slot: dict[str, object], val: float, held: float
) -> dict[str, object]:
    text = f"x\nheldout_bpb: {held}\n---\nval_bpb: {val}\nnum_steps: 800\n"
    out = root / "executor" / f"job{slot['index']}" / "stdout"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return {
        "index": slot["index"],
        "kind": "heldout",
        "arm": slot["arm"],
        "seed": slot["seed"],
        "attempt": 1,
        "base_program_sha256": slot["base_program_sha256"],
        "program_sha256": slot["program_sha256"],
        "execution": {
            "artifacts": [
                {
                    "path": str(out.relative_to(root)),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            ]
        },
        "summary": {"val_bpb": val, "num_steps": 800, "heldout_bpb": held},
        "validity": {"valid": True},
    }


def test_retained_evidence_checks_stdout(tmp_path: Path) -> None:
    order = _order()
    receipt = _receipt(tmp_path, order[0], 1.4, 1.41)
    _write(tmp_path / "run-01.json", receipt)
    kept = select_retained({"run-01.json": receipt}, order, {"1d83776": 700.0})
    entries = resume.verify_retained_evidence(tmp_path, kept, {"1d83776": 700.0})
    assert entries[0]["receipt_sha256"] == resume.audit.digest(tmp_path / "run-01.json")
    receipt["summary"]["heldout_bpb"] = 1.0  # type: ignore[index]
    with pytest.raises(ResumeMismatchError):
        resume.verify_retained_evidence(tmp_path, kept, {"1d83776": 700.0})


def test_retained_evidence_refuses_changed_artifact(tmp_path: Path) -> None:
    order = _order()
    receipt = _receipt(tmp_path, order[0], 1.4, 1.41)
    kept = select_retained({"r.json": receipt}, order, {"1d83776": 700.0})
    stdout = tmp_path / "executor" / "job1" / "stdout"
    stdout.write_text(stdout.read_text() + "tamper")
    _write(tmp_path / "r.json", receipt)
    with pytest.raises(ResumeMismatchError):
        resume.verify_retained_evidence(tmp_path, kept, {"1d83776": 700.0})


def _plan(source: Path, retained: list[dict[str, object]], output: Path):
    entries = []
    for receipt in retained:
        name = f"run-{receipt['index']:02d}.json"
        _write(source / name, receipt)
        entries.append(
            {
                "slot": receipt["index"],
                "receipt": name,
                "receipt_sha256": resume.audit.digest(source / name),
            }
        )
    return {
        "order": _order(),
        "resume": {
            "source_directory": str(source),
            "retained": entries,
            "not_retained": [{"receipt": "run-05.json", "reason": "memory guard"}],
        },
    }


def test_pending_slots_exclude_retained(tmp_path: Path) -> None:
    order = _order()
    plan = _plan(
        tmp_path / "old",
        [_receipt(tmp_path, order[i], 1.4, 1.41) for i in range(4)],
        tmp_path / "new",
    )
    assert [i["index"] for i in resume.pending_slots(plan)] == [5, 6]


def test_retained_receipt_changed_after_planning_refuses(tmp_path: Path) -> None:
    order = _order()
    plan = _plan(tmp_path / "old", [_receipt(tmp_path, order[0], 1.4, 1.41)], tmp_path)
    path = tmp_path / "old" / "run-01.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ResumeMismatchError):
        resume.retained_receipts(plan)


def test_combined_report_unions_both_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order = _order()
    old, new, parent = tmp_path / "old", tmp_path / "new", tmp_path / "parent"
    gains = {3000: 0.01, 3001: 0.012, 3002: 0.008}
    scores = {}
    for slot in order:
        offset = gains[slot["seed"]] if slot["arm"] == "bc5cdc1" else 0.0  # type: ignore[index]
        scores[slot["index"]] = (1.40 + offset, 1.41 + offset)
    retained = [_receipt(tmp_path, order[i], *scores[i + 1]) for i in range(4)]
    plan = _plan(old, retained, new)
    new.mkdir()
    for i in (4, 5):
        receipt = _receipt(tmp_path, order[i], *scores[i + 1])
        _write(new / f"run-{i + 1:02d}-heldout-a1.json", receipt)
    # A failed attempt in the new directory is excluded.
    _write(
        new / "run-05-failed.json",
        {**retained[0], "index": 5, "summary": None, "validity": {"valid": False}},
    )
    _write(parent / "plan.json", {"plan": {"comparisons": []}})
    monkeypatch.setattr(resume.base, "PARENT_RUN", parent)
    monkeypatch.setattr(resume, "analyze_audit", lambda *_a, **_k: {"stub": True})
    resume.write_reports(new, plan)
    held = json.loads((new / resume.base.HELDOUT_REPORT).read_text())
    assert held["heldout"]["seeds"] == list(SEEDS)
    assert held["heldout"]["gains"] == pytest.approx([0.01, 0.012, 0.008])
    attempts = json.loads((new / resume.ATTEMPTS_REPORT).read_text())
    sources = {a["slot"]: a["source_directory"] for a in attempts["attempts"]}
    assert sources[1] == str(old)
    assert sources[6] == resume.display(new)
    assert attempts["retained"] == 4
    assert attempts["excluded_invalid"] == 1
    assert attempts["prior_attempts_not_retained"][0]["receipt"] == "run-05.json"


def test_watched_run_uses_the_new_floor(tmp_path: Path) -> None:
    def watch(free: int, level: int):
        async def go():
            executor = LocalExecutor(
                tmp_path / f"e{free}{level}", memory_sample_seconds=1
            )
            try:
                return await resume.run_watched(
                    executor,
                    ProcessJob(
                        ("/bin/sleep", "30"),
                        cwd=tmp_path,
                        environment={"PATH": "/usr/bin:/bin"},
                        reviewed=True,
                        limits=Limits(
                            wall_seconds=60, memory_bytes=1024**3, output_bytes=1024**2
                        ),
                    ),
                    key=f"k{free}{level}",
                    read=lambda: MemoryReading("t", level, free),
                    interval=0.05,
                    floor=5,
                )
            finally:
                await executor.close()

        return asyncio.run(go())

    assert watch(4, 2).memory_trip is not None
    assert watch(9, 2).memory_trip is None
    assert watch(4, 2).snapshot.state == State.CANCELLED
    assert watch(80, 4).memory_trip is not None
