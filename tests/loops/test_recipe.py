"""Check the published port on CPU; never import MLX or start model training."""

import ast
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, runtime_checkable

import numpy as np
import pytest

from openloop.loops import T1, ContractError, LoopAction, TrainingSource
from openloop.loops.recipe import RECIPE_HASHES, shared_training
from openloop.loops.t1_report import generate_programs
from openloop.loops.training import baseline_config

ROOT = Path(__file__).parents[2]
RECIPE = ROOT / "src/openloop/loops/recipes/m4_pro_v1"


@runtime_checkable
class CPUOptimizer(Protocol):
    def _step_muon(
        self, grads: np.ndarray, params: np.ndarray, group: Mapping[str, object]
    ) -> np.ndarray: ...

    def _step_adamw(
        self,
        path: str,
        grad: np.ndarray,
        param: np.ndarray,
        group: Mapping[str, object],
    ) -> np.ndarray: ...


@pytest.fixture
def source() -> TrainingSource:
    snapshots = []
    for name, expected in RECIPE_HASHES.items():
        data = (RECIPE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        snapshots.append(data.decode())
    return TrainingSource(shared_training(*snapshots), "CPU evaluator", "CPU lock")


def test_shared_recipe_is_independent_of_hardware_and_preserves_model(
    source: TrainingSource,
) -> None:
    settings = baseline_config(source.train)
    assert settings["DEPTH"] == 6
    assert settings["DEVICE_BATCH_SIZE"] == 4
    assert settings["TOTAL_BATCH_SIZE"] == 8192
    assert settings["WARMDOWN_RATIO"] == 0.7
    assert "MLP_RATIO = 1.5" in source.train
    assert "MuonAdamWMLX(param_groups)" in source.train
    assert "get_hardware_info" not in source.train
    assert "from backends" not in source.train
    original = ast.parse((RECIPE / "train_mlx.py.txt").read_text())
    port = ast.parse(source.train)
    for name in ("GPT", "MLP", "CausalSelfAttention", "RotaryEmbedding"):
        before = next(
            node for node in original.body if getattr(node, "name", "") == name
        )
        after = next(node for node in port.body if getattr(node, "name", "") == name)
        assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize(("fidelity", "tokens"), [("8m", 8388608), ("21m", 20971520)])
def test_generated_recipe_counts_all_updates_and_retains_token_schedule(
    source: TrainingSource, fidelity: str, tokens: int
) -> None:
    program = generate_programs(source)[fidelity]
    compile(program, "shared/train.py", "exec")
    assert "evaluate_bpb" not in program
    assert "backend='mlx'" not in program
    assert "TIME_BUDGET" not in program
    tree = ast.parse(program)
    loop = next(node for node in tree.body if isinstance(node, ast.While))
    # Execute the actual generated schedule, timer, and stopping branches.
    # Replace expensive model/data operations with a constant duration on CPU.
    selected = {"progress", "lrm", "muon_momentum", "muon_weight_decay", "dt", "step"}
    loop.body = [
        node
        for node in loop.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in selected
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "step"
        )
        or (
            isinstance(node, ast.If)
            and any(isinstance(child, ast.Break) for child in ast.walk(node))
        )
        or (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "step >= STARTUP_EXCLUDE_STEPS"
        )
    ]
    for node in loop.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "dt":
            node.value = ast.Constant(0.01)
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and all(isinstance(target, ast.Name) for target in node.targets)
        and isinstance(node.value, (ast.Constant, ast.Tuple))
    ]
    schedules = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("get_")
    ]
    statements: list[ast.stmt] = [*constants, *schedules, loop]
    control = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(control)
    namespace: dict[str, object] = {}
    exec(compile(control, "CPU/control.py", "exec"), namespace)
    assert namespace["step"] == tokens // 8192
    assert namespace["total_training_time"] == pytest.approx(tokens / 8192 * 0.01)
    assert namespace["progress"] == pytest.approx(1 - 8192 / tokens)
    assert namespace["lrm"] == pytest.approx(8192 / tokens / 0.7)


def test_port_changes_identity_and_keeps_contract(
    t1: T1, source: TrainingSource
) -> None:
    port = T1(
        TrainingSource(source.train, t1.source.prepare, t1.source.lock),
        t1.corpus,
        t1.tokenizer_dir,
        t1.python,
    )
    old = t1.spec.inputs(t1.normalize_config({}), LoopAction(42), "a" * 64)
    new = port.spec.inputs(port.normalize_config({}), LoopAction(42), "a" * 64)
    assert old.hash != new.hash
    assert old.candidate_hash != new.candidate_hash
    assert old.data_hash == new.data_hash
    assert old.tokenizer_hash == new.tokenizer_hash
    assert old.evaluator_hash == new.evaluator_hash
    assert old.dependencies_hash == new.dependencies_hash
    assert port.build_job(new).program is not None
    with pytest.raises(ContractError):
        port.normalize_config({"MLP_RATIO": 4})
    with pytest.raises(ContractError):
        port.build_job(old)


def test_frozen_manifest_matches_recipe(source: TrainingSource) -> None:
    manifest = json.loads((RECIPE / "freeze.json").read_text())
    assert manifest["train_sha256"] == hashlib.sha256(source.train.encode()).hexdigest()
    for name, program in generate_programs(source).items():
        expected = next(item for item in manifest["fidelities"] if item["name"] == name)
        assert (
            hashlib.sha256(program.encode()).hexdigest() == expected["program_sha256"]
        )


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2, 2)])
def test_published_optimizer_updates_zero_and_nonzero_gradients_on_cpu(shape) -> None:
    tree = ast.parse((RECIPE / "muon_mlx.py.txt").read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    mx = SimpleNamespace(
        float32=np.float32,
        bfloat16=np.float32,
        array=np.array,
        sqrt=np.sqrt,
        swapaxes=np.swapaxes,
        zeros_like=np.zeros_like,
        zeros=np.zeros,
        maximum=np.maximum,
        rsqrt=lambda value: 1 / np.sqrt(value),
    )
    namespace: dict[str, object] = {"mx": mx}
    exec(compile(tree, "CPU/muon.py", "exec"), namespace)
    optimizer_type = namespace["MuonAdamWMLX"]
    assert callable(optimizer_type)
    group = {"lr": 0.04, "momentum": 0.95, "weight_decay": 0.2, "_group_id": "matrix"}
    optimizer = optimizer_type([group])
    assert isinstance(optimizer, CPUOptimizer)
    muon_step = optimizer._step_muon
    adam_step = optimizer._step_adamw
    params = np.ones((1, *shape), dtype=np.float32)
    zero = muon_step(np.zeros_like(params), params, group)
    assert np.isfinite(zero).all()
    grads = np.arange(params.size, dtype=np.float32).reshape(params.shape) + 1
    updated = muon_step(grads, params, group)
    assert np.isfinite(updated).all()
    assert not np.array_equal(updated, params)
    # Independent AdamW first-step formula, including decoupled decay.
    adam = {"lr": 0.1, "betas": (0.8, 0.95), "eps": 1e-10, "weight_decay": 0.2}
    result = adam_step("embedding", grads, params, adam)
    np.testing.assert_allclose(result, params * 0.98 - 0.1, atol=1e-6)
