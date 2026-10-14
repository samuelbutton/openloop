"""Amendment 5 script boundaries: multi-source retention, pending slot, reports."""

import importlib.util
import json
from pathlib import Path

import pytest

from openloop.studies.resume import ResumeMismatchError

SCRIPT = Path(__file__).parents[2] / "research/audit_heldout_complete.py"
spec = importlib.util.spec_from_file_location("audit_heldout_complete", SCRIPT)
assert spec is not None
assert spec.loader is not None
complete = importlib.util.module_from_spec(spec)
spec.loader.exec_module(complete)

ARMS = ("1d83776", "bc5cdc1")
SEEDS = (3000, 3001, 3002)
GAINS = {3000: 0.010, 3001: 0.012, 3002: 0.020}


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


def _receipt(slot: dict[str, object], *, failed: bool = False) -> dict[str, object]:
    child = slot["arm"] == "1d83776"
    val = 1.40 + (0.0 if child else GAINS[slot["seed"]])  # type: ignore[index]
    summary = {"val_bpb": val, "num_steps": 800, "heldout_bpb": val + 0.01}
    receipt: dict[str, object] = {
        "index": slot["index"],
        "kind": "heldout",
        "arm": slot["arm"],
        "seed": slot["seed"],
        "attempt": 1,
        "summary": None if failed else summary,
        "validity": {"valid": True},
    }
    if failed:
        receipt["failure"] = "ExecutorError: failed: None"
    return receipt


def _plan(
    tmp_path: Path, placement: dict[str, list[int]], failed: dict[str, list[int]]
) -> dict[str, object]:
    """Receipts per directory; failed slots get receipts that are not kept."""
    order = _order()
    entries = []
    not_retained = []
    for directory, slots in placement.items():
        for i in slots:
            name = f"run-{i:02d}.json"
            _write(tmp_path / directory / name, _receipt(order[i - 1]))
            entries.append(
                {
                    "slot": i,
                    "receipt": name,
                    "source_directory": str(tmp_path / directory),
                    "receipt_sha256": complete.audit.digest(
                        tmp_path / directory / name
                    ),
                }
            )
    for directory, slots in failed.items():
        for i in slots:
            name = f"run-{i:02d}-failed.json"
            _write(tmp_path / directory / name, _receipt(order[i - 1], failed=True))
            not_retained.append({"receipt": name, "source_directory": directory})
    return {
        "order": order,
        "resume": {
            "source_directories": [str(tmp_path / d) for d in placement],
            "retained": entries,
            "not_retained": not_retained,
        },
    }


@pytest.fixture
def plan(tmp_path: Path) -> dict[str, object]:
    return _plan(tmp_path, {"b": [1, 2, 3, 4], "c": [5]}, {"c": [6]})


def test_only_the_missing_slot_is_pending(plan: dict[str, object]) -> None:
    assert [s["index"] for s in complete.pending_slots(plan)] == [6]  # type: ignore[index]


def test_retains_from_both_sources_and_checks_hashes(
    plan: dict[str, object], tmp_path: Path
) -> None:
    kept = complete.retained_receipts(plan)
    assert {r["index"]: r[complete.SOURCE_KEY] for r in kept} == {
        1: str(tmp_path / "b"),
        2: str(tmp_path / "b"),
        3: str(tmp_path / "b"),
        4: str(tmp_path / "b"),
        5: str(tmp_path / "c"),
    }
    path = tmp_path / "c" / "run-05.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ResumeMismatchError):
        complete.retained_receipts(plan)


def test_hooks_are_restored(plan: dict[str, object]) -> None:
    before = complete.resume.pending_slots
    with complete._completion_hooks():
        assert complete.resume.pending_slots is complete.pending_slots
        assert complete.resume.retained_receipts is complete.retained_receipts
        assert complete.resume.write_reports is complete.write_reports
    assert complete.resume.pending_slots is before


def _run_reports(
    tmp_path: Path, plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> Path:
    parent = tmp_path / "parent"
    _write(parent / "plan.json", {"plan": {"comparisons": []}})
    monkeypatch.setattr(complete.base, "PARENT_RUN", parent)
    monkeypatch.setattr(complete, "analyze_audit", lambda *_a, **_k: {"stub": True})
    out = tmp_path / "d"
    _write(out / "run-06-heldout-a1.json", _receipt(_order()[5]))
    # A new failed attempt is listed but never used.
    _write(out / "run-06-failed.json", _receipt(_order()[5], failed=True))
    complete.write_reports(out, plan)
    return out


def test_reports_give_five_and_four_pair_results(
    plan: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Slots 5 and 6 are seed 3002; completing the pair adds one gain.
    out = _run_reports(tmp_path, plan, monkeypatch)
    # Only the child of seed 3002 is retained, so the earlier-only analysis
    # has two matched pairs and the primary analysis has three.
    primary = json.loads((out / complete.base.HELDOUT_REPORT).read_text())
    four = json.loads((out / complete.FOUR_PAIR_REPORT).read_text())
    assert primary["analysis"].startswith("primary")
    assert four["analysis"].startswith("secondary")
    assert primary["heldout"]["seeds"] == list(SEEDS)
    assert primary["heldout"]["gains"] == pytest.approx([GAINS[s] for s in SEEDS])
    assert four["heldout"]["seeds"] == [3000, 3001]
    assert four["unmatched"] == [3002]


def test_attempts_report_covers_every_directory(
    plan: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run_reports(tmp_path, plan, monkeypatch)
    report = json.loads((out / complete.ATTEMPTS_REPORT).read_text())
    rows = {(r["source_directory"], r["receipt"]): r for r in report["attempts"]}
    assert len(rows) == 4 + 1 + 1 + 2
    failed = rows[(str(tmp_path / "c"), "run-06-failed.json")]
    assert failed["failure"]
    assert not failed["used_in_analysis"]
    assert not rows[(complete.display(out), "run-06-failed.json")]["used_in_analysis"]
    assert rows[(complete.display(out), "run-06-heldout-a1.json")]["used_in_analysis"]
    assert rows[(str(tmp_path / "b"), "run-01.json")]["used_in_analysis"]
    assert report["retained"] == 5
    assert report["used_in_analysis"] == 6
