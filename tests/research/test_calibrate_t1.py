"""S1 preparation and consent boundaries; no training or native MLX imports."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "research/calibrate_t1.py"
spec = importlib.util.spec_from_file_location("s1_calibration", SCRIPT)
assert spec is not None
assert spec.loader is not None
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def test_evidence_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    calibration.save(path, {"observed": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        calibration.save(path, {"observed": 2})
    assert path.read_bytes() == before


def test_cli_does_not_execute_without_flag(tmp_path: Path, monkeypatch) -> None:
    class Prepared:
        hash = "f" * 64
        trials = ("first", "second")
        training_tokens = 10
        wall_limit_seconds = 100

    prepared = Prepared()
    saved = []

    def fake_preflight(*args):
        return object(), prepared, {"fixture": True}

    def fake_save(path, value):
        saved.append(value)

    async def forbidden_execute(*args):
        pytest.fail("Preparation must not train")

    monkeypatch.setattr(calibration, "preflight", fake_preflight)
    monkeypatch.setattr(calibration, "save", fake_save)
    monkeypatch.setattr(calibration, "execute", forbidden_execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--upstream",
            str(tmp_path),
            "--corpus",
            str(tmp_path),
            "--tokenizer",
            str(tmp_path),
            "--worker-python",
            str(tmp_path / "python"),
            "--plan",
            str(tmp_path / "plan.json"),
        ],
    )
    calibration.main()
    assert len(saved) == 1
    assert saved[0]["native_training_executed"] is False
