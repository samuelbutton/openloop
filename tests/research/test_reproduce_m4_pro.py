"""Reference report checks use synthetic logs, never GPU training."""

import importlib.util
import sys
from pathlib import Path

import pytest

from openloop.executors import ExecutorError

SCRIPT = Path(__file__).parents[2] / "research/reproduce_m4_pro.py"
spec = importlib.util.spec_from_file_location("reference_reproduction", SCRIPT)
assert spec is not None
assert spec.loader is not None
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_final_summary_ignores_training_output() -> None:
    summary = reference.parse_summary(
        "loss: 99\rstep 00001\n---\nval_bpb: 1.429396\nnum_steps: 751\n"
        "depth: 6\nbackend: mlx\nchip: Apple M4 Pro\n"
    )
    assert summary["val_bpb"] == 1.429396
    assert summary["num_steps"] == 751
    assert summary["backend"] == "mlx"
    assert "loss" not in summary


@pytest.mark.parametrize(
    "log",
    [
        "val_bpb: 1.429396\nnum_steps: 751",
        "\n---\nnum_steps: 751\n",
        "\n---\nval_bpb: nan\nnum_steps: 751\n",
        "\n---\nval_bpb: 1.429396\nnum_steps: -1\n",
        "\n---\nval_bpb: 1.429396\nnum_steps: 1.5\n",
    ],
)
def test_invalid_summary_is_rejected(log: str) -> None:
    with pytest.raises(ExecutorError):
        reference.parse_summary(log)


def test_evidence_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    reference.save(path, {"measured": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        reference.save(path, {"measured": 2})
    assert path.read_bytes() == original


def test_cli_preserves_venv_interpreter_symlink(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base-python"
    base.touch()
    interpreter = tmp_path / "venv-python"
    interpreter.symlink_to(base)
    selected = []

    async def fake_reproduction(source: Path, python: Path, output: Path) -> None:
        selected.append(python)

    monkeypatch.setattr(reference, "reproduce", fake_reproduction)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--source",
            str(tmp_path),
            "--python",
            str(interpreter),
            "--output",
            str(tmp_path / "run"),
            "--approve-local-training",
        ],
    )
    reference.main()
    assert selected == [interpreter]
    assert selected[0] != base
