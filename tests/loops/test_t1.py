"""T1 contract and trusted worker use CPU stand-ins, never MLX or GPU training."""

import asyncio
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openloop.executors import LocalExecutor
from openloop.ledger import Ledger, Purpose
from openloop.loops import (
    T1,
    ContractError,
    ExperimentEnv,
    LoopAction,
    Phase,
    RunContext,
    TrainingSource,
)
from openloop.loops.t1 import ADAPTER_FILES, LOG_TAIL_CHARACTERS
from openloop.loops.t1_worker import run_training, verify_runtime

CONTEXT = RunContext("run", "attempt")


def test_two_token_fidelities_complete_through_shared_contract(
    t1: T1, ledger: Ledger
) -> None:
    env = ExperimentEnv(
        t1,
        ledger,
        runner=t1.run,
        config={},
        environment_hash="a" * 64,
        budget=8_388_608 + 20_971_520,
    )
    screen = asyncio.run(env.step(LoopAction(42)))
    confirm = asyncio.run(env.step(LoopAction(43, "21m", Phase.CONFIRM)))
    assert screen.work_units == 8_388_608
    assert confirm.work_units == 20_971_520
    assert confirm.episode_done
    assert screen.result is not None
    assert screen.result.metrics["val_bpb"].value == 1.75
    assert screen.result.observations["tokens_per_second"].value == 65536
    assert ledger.get(screen.run_id).inputs.reproducibility == "noisy"


def test_noisy_repeats_require_explicit_replication(t1: T1, ledger: Ledger) -> None:
    env = ExperimentEnv(
        t1,
        ledger,
        runner=t1.run,
        config={},
        environment_hash="a" * 64,
        budget=3 * 8_388_608,
    )
    first = asyncio.run(env.step(LoopAction(42)))
    second = asyncio.run(env.step(LoopAction(42, purpose=Purpose.REPLICATION)))
    assert first.attempt_id != second.attempt_id
    assert first.executed
    assert second.executed
    cached = asyncio.run(env.step(LoopAction(42)))
    assert cached.attempt_id == first.attempt_id
    assert not cached.executed
    assert cached.work_units == 0


@pytest.mark.parametrize(
    "config",
    [
        {"TIME_BUDGET": 1},
        {"FINAL_EVAL_BATCH_SIZE": 1},
        {"DEPTH": 0},
        {"TOTAL_BATCH_SIZE": 12345},
        {"WARMUP_RATIO": 0.8},
        {"ADAM_BETAS": [0.5, 1.0]},
    ],
)
def test_protected_or_invalid_settings_are_rejected(t1: T1, config) -> None:
    with pytest.raises(ContractError):
        t1.normalize_config(config)


def test_declared_architecture_change_has_new_candidate_identity(t1: T1) -> None:
    original = t1.spec.inputs(t1.normalize_config({}), LoopAction(42), "a" * 64)
    changed = t1.spec.inputs(
        t1.normalize_config({"DEPTH": 5}), LoopAction(42), "a" * 64
    )
    assert original.candidate_hash != changed.candidate_hash
    assert "DEPTH = 5" in (t1.build_job(changed).program or "")


def test_pinned_source_loader_rejects_modified_clone(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("not the pinned source")
    with pytest.raises(ContractError, match="Pinned upstream source changed"):
        TrainingSource.load(tmp_path)


def test_tokenizer_drift_is_rejected_before_execution(t1: T1) -> None:
    inputs = t1.spec.inputs(t1.normalize_config({}), LoopAction(42), "a" * 64)
    (t1.tokenizer_dir / "token_bytes.npy").write_bytes(b"changed fixture")
    with pytest.raises(ContractError, match="tokenizer changed"):
        run_training(t1, t1.build_job(inputs))


def test_worker_rejects_modified_program(t1: T1) -> None:
    inputs = t1.spec.inputs(t1.normalize_config({}), LoopAction(42), "a" * 64)
    job = replace(t1.build_job(inputs), program="raise RuntimeError('changed')")
    with pytest.raises(ContractError, match="exact generated"):
        run_training(t1, job)


def test_evaluator_rejects_unmatched_budget(t1: T1) -> None:
    inputs = t1.spec.inputs(t1.normalize_config({}), LoopAction(42), "a" * 64)
    with pytest.raises(ContractError, match="declared token budget"):
        t1.evaluate(t1.build_job(inputs), {"work_units": 1, "val_bpb": 0.0})


def test_worker_rejects_wrong_dependency_versions(monkeypatch) -> None:
    monkeypatch.setattr("openloop.loops.t1_worker.version", lambda name: "2.0")
    lock = """
[[package]]
name = "fixture-library"
version = "1.0"
source = {registry = "test-only"}
"""
    with pytest.raises(ContractError, match="Locked dependency mismatch"):
        verify_runtime(lock)


def test_dependency_check_selects_python_312_branch(monkeypatch) -> None:
    monkeypatch.setattr("openloop.loops.t1_worker.version", lambda name: "2.0")
    lock = """
[[package]]
name = "fixture-library"
version = "1.0"
source = {registry = "test-only"}
resolution-markers = ["python_full_version < '3.11'"]
[[package]]
name = "fixture-library"
version = "2.0"
source = {registry = "test-only"}
resolution-markers = ["python_full_version >= '3.11'"]
"""
    assert verify_runtime(lock) == {"fixture-library": "2.0"}


@pytest.mark.skipif(os.name != "posix", reason="process liveness uses POSIX signal 0")
def test_cancellation_reaps_cpu_worker(
    t1: T1, ledger: Ledger, tmp_path: Path, executor: LocalExecutor
) -> None:
    marker = tmp_path / "worker-started"
    prepare = (
        t1.source.prepare
        + f"""
import os, time
from pathlib import Path
Path({str(marker)!r}).write_text(str(os.getpid()))
time.sleep(3600)
"""
    )
    loop = T1(
        replace(t1.source, prepare=prepare),
        t1.corpus,
        t1.tokenizer_dir,
        t1.python,
        executor=executor,
    )
    env = ExperimentEnv(
        loop,
        ledger,
        runner=loop.run,
        config={},
        environment_hash="a" * 64,
        budget=8_388_608,
    )

    async def scenario() -> None:
        pending = asyncio.create_task(env.step(LoopAction(42)))
        try:
            async with asyncio.timeout(5):
                while not marker.exists():
                    await asyncio.sleep(0.01)
        finally:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        # Check inside the running loop: loop teardown must not be what stops it.
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)

    asyncio.run(scenario())
    assert len(ledger.query(status="failed")) == 1
    assert asyncio.run(env.initial_observation()).remaining_budget == 0


def run_failing_worker(t1: T1, executor: LocalExecutor, failure: str) -> str:
    prepare = t1.source.prepare + "\nimport sys\n" + failure
    loop = T1(
        replace(t1.source, prepare=prepare),
        t1.corpus,
        t1.tokenizer_dir,
        t1.python,
        executor=executor,
    )
    inputs = loop.spec.inputs(loop.normalize_config({}), LoopAction(42), "a" * 64)
    with pytest.raises(ContractError) as error:
        asyncio.run(loop.run(loop.build_job(inputs), CONTEXT))
    return str(error.value)


def test_failure_reason_includes_stdout_when_stderr_is_empty(
    t1: T1, executor: LocalExecutor
) -> None:
    reason = run_failing_worker(t1, executor, "print('FAIL')\nsys.exit(1)\n")
    assert "exit code 1" in reason
    assert "--- stdout tail ---\nFAIL" in reason


def test_failure_reason_keeps_only_bounded_tails(
    t1: T1, executor: LocalExecutor
) -> None:
    reason = run_failing_worker(
        t1,
        executor,
        "print('S' * 20000 + 'STDOUT-END')\n"
        "sys.stderr.write('E' * 20000 + 'STDERR-END')\n"
        "sys.exit(1)\n",
    )
    assert reason.endswith("STDERR-END")
    assert "STDOUT-END" in reason
    assert "S" * (LOG_TAIL_CHARACTERS + 1) not in reason
    assert "E" * (LOG_TAIL_CHARACTERS + 1) not in reason
    assert len(reason) < 3 * LOG_TAIL_CHARACTERS


def test_worker_python_stays_out_of_the_specification(
    t1: T1, tmp_path: Path, executor: LocalExecutor
) -> None:
    other = T1(
        t1.source,
        t1.corpus,
        t1.tokenizer_dir,
        tmp_path / "elsewhere/python",
        executor=executor,
    )
    assert other.spec == t1.spec


def test_failure_reason_names_executor_state_and_log_digests(
    t1: T1, executor: LocalExecutor
) -> None:
    reason = run_failing_worker(t1, executor, "print('FAIL')\nsys.exit(3)\n")
    (directory,) = (path.parent for path in executor.root.glob("*/stdout"))
    assert "exit code 3 (failed)" in reason
    assert f"executor job {directory.name}" in reason
    for name in ("stdout", "stderr"):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert f"{name} sha256={digest}" in reason


def test_replication_through_one_executor_runs_a_second_worker(
    t1: T1, ledger: Ledger, executor: LocalExecutor
) -> None:
    env = ExperimentEnv(
        t1,
        ledger,
        runner=t1.run,
        config={},
        environment_hash="a" * 64,
        budget=2 * 8_388_608,
    )

    async def scenario() -> tuple[Any, Any]:
        first = await env.step(LoopAction(42))
        second = await env.step(LoopAction(42, purpose=Purpose.REPLICATION))
        return first, second

    first, second = asyncio.run(scenario())
    assert first.attempt_id != second.attempt_id
    assert first.executed
    assert second.executed
    assert len(list(executor.root.glob("*/stdout"))) == 2


def test_result_records_coordinator_observed_executor_evidence(
    t1: T1, ledger: Ledger, executor: LocalExecutor
) -> None:
    env = ExperimentEnv(
        t1,
        ledger,
        runner=t1.run,
        config={},
        environment_hash="a" * 64,
        budget=8_388_608,
    )
    step = asyncio.run(env.step(LoopAction(42)))
    assert step.result is not None
    (directory,) = (path.parent for path in executor.root.glob("*/stdout"))
    for role in ("stdout", "stderr"):
        digest = hashlib.sha256((directory / role).read_bytes()).hexdigest()
        assert step.result.artifacts[role] == digest
    observations = step.result.observations
    assert observations["queued_seconds"].value >= 0
    assert observations["process_seconds"].value > 0
    assert observations["process_seconds"].unit == "seconds"
    assert observations["training_seconds"].value > 0
    if "peak_rss_bytes" in observations:
        assert observations["peak_rss_bytes"].unit == "bytes"


def test_worker_cannot_supply_executor_evidence(
    t1: T1, executor: LocalExecutor
) -> None:
    forged = (
        "import json\n"
        'forged = {"execution": {"state": "succeeded"}}\n'
        'print("OPENLOOP_RESULT " + json.dumps(forged))\n'
        "import sys\nsys.exit(0)\n"
    )
    loop = T1(
        replace(t1.source, prepare=t1.source.prepare + "\n" + forged),
        t1.corpus,
        t1.tokenizer_dir,
        t1.python,
        executor=executor,
    )
    inputs = loop.spec.inputs(loop.normalize_config({}), LoopAction(42), "a" * 64)
    with pytest.raises(ContractError, match="must not contain 'execution'"):
        asyncio.run(loop.run(loop.build_job(inputs), CONTEXT))


def test_evaluator_rejects_missing_executor_evidence(t1: T1) -> None:
    inputs = t1.spec.inputs(t1.normalize_config({}), LoopAction(42), "a" * 64)
    output = {
        "work_units": 8_388_608,
        "steps": 128,
        "evaluation_tokens": 1_572_864,
        "val_bpb": 1.0,
        "training_seconds": 1.0,
    }
    with pytest.raises(ContractError, match="executor evidence"):
        t1.evaluate(t1.build_job(inputs), output)


def test_run_requires_an_executor(t1: T1) -> None:
    loop = T1(t1.source, t1.corpus, t1.tokenizer_dir, t1.python)
    inputs = loop.spec.inputs(loop.normalize_config({}), LoopAction(42), "a" * 64)
    with pytest.raises(ContractError, match="requires an executor"):
        asyncio.run(loop.run(loop.build_job(inputs), CONTEXT))


def test_adapter_identity_excludes_executor_code() -> None:
    root = Path(__file__).parents[2] / "src/openloop/loops"
    for name in ADAPTER_FILES:
        assert (root / name).resolve().parent == root.resolve()
