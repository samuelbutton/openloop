"""The static T1 smoke checker agrees with the trusted adapter. No Docker needed."""

import importlib.resources
import subprocess
import sys
from pathlib import Path

import pytest

from openloop.loops import ContractError
from openloop.loops.recipe import shared_training
from openloop.loops.training import TrainingSetting, baseline_config, training_program
from openloop.operators import smoke_checks
from openloop.operators.smoke_checks import SmokeCheckError, check_source
from tests.loops.conftest import TOY_TRAIN


def recipe_source() -> str:
    root = importlib.resources.files("openloop.loops") / "recipes" / "m4_pro_v1"
    return shared_training(
        (root / "train_mlx.py.txt").read_text(), (root / "muon_mlx.py.txt").read_text()
    )


def adapter_accepts(source: str) -> bool:
    try:
        training_program(
            source,
            baseline_config(source),
            seed=0,
            token_budget=2**16,
            tokenizer_dir="t",
        )
    except ContractError:
        return False
    return True


def test_settings_match_the_trusted_adapter() -> None:
    assert set(smoke_checks.SETTINGS) == {name.value for name in TrainingSetting}


@pytest.mark.parametrize("source", [TOY_TRAIN, recipe_source()])
def test_accepts_sources_the_adapter_accepts(source: str) -> None:
    assert adapter_accepts(source)
    check_source(source)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("DEPTH = 4\n", ""),
        lambda s: s.replace("DEPTH = 4", "DEPTH = int('4')"),
        lambda s: s.replace("total_tokens = ", "total_tokenz = "),
        lambda s: s.replace("mx.random.seed(42)", "pass"),
        lambda s: s.replace(
            "mx.random.seed(42)", "mx.random.seed(1)\nmx.random.seed(2)"
        ),
        lambda s: s.replace(
            "STARTUP_EXCLUDE_STEPS = 1",
            "STARTUP_EXCLUDE_STEPS = 1\nSTARTUP_EXCLUDE_STEPS = 2",
        ),
        lambda s: s.replace("        break\n", "        print(1)\n        break\n"),
        lambda s: s.replace("    progress = ", "    progresz = "),
        lambda s: s + "\ndef broken(:\n",
    ],
)
def test_rejects_what_the_adapter_rejects(mutate) -> None:
    source = mutate(TOY_TRAIN)
    assert source != TOY_TRAIN
    with pytest.raises(SmokeCheckError):
        check_source(source)
    try:
        compile(source, "x", "exec")
    except SyntaxError:
        return
    assert not adapter_accepts(source)


def test_command_runs_as_a_script(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.py", tmp_path / "bad.py"
    good.write_text(TOY_TRAIN)
    bad.write_text(TOY_TRAIN.replace("DEPTH = 4\n", ""))
    command = smoke_checks.t1_smoke_command("good.py")
    assert command[:2] == ("python", "-c")
    ran = subprocess.run(
        (sys.executable, *command[1:]), cwd=tmp_path, capture_output=True, text=True
    )
    assert ran.returncode == 0, ran.stderr
    ran = subprocess.run(
        (sys.executable, *command[1:3], "bad.py"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert ran.returncode == 1
    assert "DEPTH" in ran.stderr
    assert not list(tmp_path.glob("__pycache__"))


def test_worker_preserves_bytes_exactly(tmp_path: Path) -> None:
    from openloop.operators.smoke_worker import smoke

    source, outputs = tmp_path / "source", tmp_path / "outputs"
    source.mkdir()
    outputs.mkdir()
    (source / "a.txt").write_bytes(b"x\r\n")
    text = "après ✓\r\nend"
    smoke(
        source,
        tmp_path / "worktree",
        outputs,
        {"base_files": {"a.txt": "x\r\n"}, "files": {"a.txt": text}, "smoke": ["true"]},
    )
    assert (outputs / "a.txt").read_bytes() == text.encode()
