"""Amendment 2 script boundaries; no training and no native MLX imports."""

import importlib.util
import json
from pathlib import Path

import pytest

from openloop.ledger import InvalidInputError
from openloop.studies.heldout import EVAL_ANCHOR, HELDOUT_EVAL

SCRIPT = Path(__file__).parents[2] / "research/audit_heldout.py"
spec = importlib.util.spec_from_file_location("audit_heldout", SCRIPT)
assert spec is not None
assert spec.loader is not None
heldout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heldout)

PROGRAM = f"import os\nmx.random.seed(42)\n{EVAL_ANCHOR}print(val_bpb)\n"
FILES = {"train_mlx.py": PROGRAM.encode(), "prepare.py": b"TIME_BUDGET = 300\n"}


def test_seeds_are_disjoint_from_everything_used_before() -> None:
    used = {42, 2000, 2001, 2002, 2003, 2004}
    assert not used & set(heldout.HELDOUT_SEEDS)
    assert heldout.JOB_COUNT == 11
    assert (heldout.COMPLETION_ARM, heldout.COMPLETION_SEED) == ("8f00506", 2004)


def test_heldout_job_adds_only_the_seed_and_the_evaluation(tmp_path: Path) -> None:
    base, final = heldout.write_job_files(FILES, "heldout", 3001, tmp_path)
    text = (tmp_path / "train_mlx.py").read_text()
    seeded = PROGRAM.replace("seed(42)", "seed(3001)")
    assert text == seeded.replace(EVAL_ANCHOR, EVAL_ANCHOR + HELDOUT_EVAL)
    assert base == heldout.audit.sha(seeded.encode())
    assert final == heldout.audit.sha(text.encode())
    assert (tmp_path / "prepare.py").read_bytes() == FILES["prepare.py"]


def test_completion_job_is_the_unchanged_audit_program(tmp_path: Path) -> None:
    base, final = heldout.write_job_files(FILES, "seed_completion", 2004, tmp_path)
    assert base == final
    assert (tmp_path / "train_mlx.py").read_text() == PROGRAM.replace(
        "seed(42)", "seed(2004)"
    )


def test_job_fails_loudly_when_the_anchor_is_missing(tmp_path: Path) -> None:
    files = {"train_mlx.py": b"mx.random.seed(42)\n", "prepare.py": b"x\n"}
    with pytest.raises(InvalidInputError):
        heldout.write_job_files(files, "heldout", 3000, tmp_path)
    assert not (tmp_path / "train_mlx.py").exists()


def test_shard_check_rejects_a_changed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / heldout.HELDOUT_SHARD
    shard.write_bytes(b"abc")
    monkeypatch.setattr(heldout, "HELDOUT_DIR", tmp_path)
    good = heldout.audit.digest(shard)
    heldout.verify_heldout_shard(good, 3)
    with pytest.raises(heldout.ConflictError):
        heldout.verify_heldout_shard(good, 4)
    with pytest.raises(heldout.ConflictError):
        heldout.verify_heldout_shard("0" * 64, 3)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _summary(val: float, held: float | None = None) -> dict[str, object]:
    summary: dict[str, object] = {"val_bpb": val, "num_steps": 800}
    if held is not None:
        summary["heldout_bpb"] = held
    return summary


def test_reports_combine_receipts_without_touching_the_original(
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
    for seed in range(2000, 2005):
        _write(
            original / f"run-p-{seed}.json",
            {"arm": "p", "seed": seed, "summary": _summary(1.40 + seed / 1e6)},
        )
        if seed != 2004:
            _write(
                original / f"run-k-{seed}.json",
                {"arm": "k", "seed": seed, "summary": _summary(1.39 + seed / 1e6)},
            )
    _write(
        original / "run-k-2004.json",
        {"arm": "k", "seed": 2004, "summary": None, "failure": "env"},
    )
    before = {p.name: p.read_bytes() for p in original.iterdir()}
    _write(
        output / "run-01-seed_completion-k-s2004.json",
        {
            "kind": "seed_completion",
            "arm": "k",
            "seed": 2004,
            "summary": _summary(1.39),
        },
    )
    for seed in heldout.HELDOUT_SEEDS:
        for arm, offset in (("1d83776", 0.0), ("bc5cdc1", 0.01)):
            _write(
                output / f"run-{arm}-{seed}.json",
                {
                    "kind": "heldout",
                    "arm": arm,
                    "seed": seed,
                    "summary": _summary(1.40 + offset, 1.41 + offset + seed / 1e7),
                },
            )
    monkeypatch.setattr(heldout, "PARENT_RUN", original)
    heldout.write_reports(output, {})
    combined = json.loads((output / heldout.SEED_COMPLETION_REPORT).read_text())
    assert combined["comparisons"][0]["seeds"] == [2000, 2001, 2002, 2003, 2004]
    assert combined["unmatched"] == {}
    held = json.loads((output / heldout.HELDOUT_REPORT).read_text())
    assert held["finding"] == "replicated"
    assert held["heldout"]["seeds"] == list(heldout.HELDOUT_SEEDS)
    assert {p.name: p.read_bytes() for p in original.iterdir()} == before
    with pytest.raises(FileExistsError):
        heldout.write_reports(output, {})
