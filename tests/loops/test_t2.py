"""T2 ledger/contract checks use trusted CPU receipts, never pretrained models."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from openloop.ledger import InvalidInputError, Ledger, content_hash
from openloop.loops import T2, ContractError, ExperimentEnv, LoopAction
from openloop.training.assets import MODEL_FILES, ModelSnapshot
from openloop.training.grpo import ArithmeticData


@pytest.fixture
def t2(tmp_path: Path) -> T2:
    root = tmp_path / "model"
    root.mkdir()
    for name in MODEL_FILES:
        (root / name).write_text("CPU fixture; never loaded as model assets")
    return T2(
        ModelSnapshot.capture(root),
        ArithmeticData.default(),
        Path("fixture-python"),
        runtime_lock="version = 1\npackage = []",
        output_root=tmp_path / "checkpoints",
    )


def receipt(t2: T2, updates: int = 2) -> dict:
    return {
        "updates": updates,
        "work_units": updates,
        "validation_count": len(t2.data.validation),
        "generated_tokens": updates * 4 + len(t2.data.validation),
        "gradient_tokens": updates * 4,
        "uniform_groups": 0,
        "mean_reward": 0.5,
        "mean_loss": 0.1,
        "max_gradient_norm": 0.5,
        "validation_pass_rate": 0.5,
        "training_seconds": 0.1,
        "evaluation_seconds": 0.02,
        "checkpoint_sha256": "b" * 64,
        "execution": {
            "state": "succeeded",
            "queued_seconds": 0.0,
            "elapsed_seconds": 0.2,
            "logs": {name: {"sha256": "c" * 64} for name in ("stdout", "stderr")},
        },
    }


def test_ledger_cache_and_update_fidelities(t2: T2, ledger: Ledger) -> None:
    async def runner(job, context):
        return receipt(t2, job.inputs.budget_amount)

    env = ExperimentEnv(
        t2, ledger, runner=runner, config={}, environment_hash="a" * 64, budget=12
    )
    first = asyncio.run(env.step(LoopAction(42)))
    cached = asyncio.run(env.step(LoopAction(42)))
    full = asyncio.run(env.step(LoopAction(43, "short")))
    assert first.work_units == 2
    assert cached.work_units == 0
    assert full.work_units == 8
    assert first.result is not None
    assert first.result.metrics["pass_rate"].direction == "maximize"
    assert first.result.metrics["pass_rate"].sample_count == 16
    assert first.result.artifacts["training_state"] == "b" * 64


@pytest.mark.parametrize(
    "config",
    [
        {"model": "other"},
        {"reward": "unsafe"},
        {"group_size": 1},
        {"max_new_tokens": 257},
        {"learning_rate": 0},
        {"clip_epsilon": 0.1},
    ],
)
def test_frozen_and_invalid_settings_are_rejected(t2: T2, config) -> None:
    with pytest.raises(InvalidInputError):
        t2.normalize_config(config)


def test_evaluation_checks_tokens_tasks_and_coordinator_evidence(t2: T2) -> None:
    inputs = t2.spec.inputs(t2.normalize_config({}), LoopAction(42), "a" * 64)
    job = t2.build_job(inputs)
    for patch in (
        {"updates": 3},
        {"generated_tokens": 99999},
        {"validation_count": 1},
        {"execution": {}},
        {"validation_pass_rate": 2},
        {"validation_pass_rate": 0.2},
        {"gradient_tokens": 1},
        {"uniform_groups": 3},
        {"work_units": 1},
        {"training_seconds": -1},
        {"max_gradient_norm": -1},
    ):
        with pytest.raises(ContractError):
            t2.evaluate(job, receipt(t2) | patch)
    with pytest.raises(ContractError, match="frozen job"):
        t2.evaluate(replace(job, program="unreviewed"), receipt(t2))
    for checkpoint in (None, "", "none"):
        with pytest.raises(InvalidInputError):
            t2.evaluate(job, receipt(t2) | {"checkpoint_sha256": checkpoint})


def test_model_drift_and_candidate_identity(t2: T2) -> None:
    baseline = t2.spec.inputs(t2.normalize_config({}), LoopAction(42), "a" * 64)
    changed = t2.spec.inputs(
        t2.normalize_config({"learning_rate": 0.01}), LoopAction(42), "a" * 64
    )
    assert baseline.candidate_hash != changed.candidate_hash
    assert content_hash(t2.data) == t2.spec.data_hash
    (t2.model.root / "config.json").write_text("corrupt")
    with pytest.raises(ValueError, match="asset changed"):
        t2.model.verify()


def test_tokenizer_identity_covers_only_tokenizer_assets(t2: T2) -> None:
    def tokenizer_hash(changed: str) -> str | None:
        files = dict(t2.model.files) | {changed: "d" * 64}
        model = ModelSnapshot(t2.model.root, files)
        return replace(t2, model=model).spec.tokenizer_hash

    assert tokenizer_hash("model.safetensors") == t2.spec.tokenizer_hash
    assert tokenizer_hash("tokenizer.json") != t2.spec.tokenizer_hash
