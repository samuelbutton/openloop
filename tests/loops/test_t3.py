"""Exercise mixture worker orchestration with CPU-only training and packers."""

import ast
import asyncio
import json
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from openloop.decider import Pair, Sample
from openloop.executors import LocalExecutor
from openloop.fineweb import file_hash, json_hash
from openloop.ledger import JSONValue, Ledger, Purpose, Status
from openloop.loops import (
    T1,
    T3,
    ContractError,
    ExperimentEnv,
    Job,
    LoopAction,
    Phase,
    TrainingSource,
)
from openloop.loops.corpus import Corpus
from openloop.loops.t3_worker import MixtureLoader, run_mixture_training

from .conftest import TOY_TRAIN, frozen_corpus

MIXTURE_TRAIN = TOY_TRAIN.replace(
    'model = "CPU stand-in"', 'model = {"other_tokens": 0}'
).replace(
    "while True:",
    'loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")\n'
    "x, y, epoch = next(loader)\n"
    "while True:\n"
    "    for _ in range(TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)):\n"
    '        if x.startswith("additional"):\n'
    '            model["other_tokens"] += DEVICE_BATCH_SIZE * MAX_SEQ_LEN\n'
    "        x, y, epoch = next(loader)",
)
MIXTURE_PREPARE = """
MAX_SEQ_LEN = 2048
EVAL_TOKENS = 1572864
TIME_BUDGET = 300
class Tokenizer:
    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        return cls()
def make_dataloader(tokenizer, batch_size, seq_len, split, buffer_size=1000):
    documents = _document_batches(split, 1)
    while True:
        texts, epoch = next(documents)
        yield texts[0], None, epoch
def evaluate_bpb(model, tokenizer, batch_size):
    text, _, epoch = next(make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val"))
    assert text.startswith("document") and epoch == 1
    return 1 + model["other_tokens"] / 8388608
"""


@pytest.fixture
def t3(t1: T1, tmp_path: Path, executor: LocalExecutor) -> T3:
    training = T1(
        replace(t1.source, train=MIXTURE_TRAIN, prepare=MIXTURE_PREPARE),
        t1.corpus,
        t1.tokenizer_dir,
        t1.python,
        executor=executor,
    )
    other = frozen_corpus(
        tmp_path / "other", [f"additional document {i}" for i in range(500)]
    )
    return T3(training, {"edu": t1.corpus, "other": other})


def environment(
    loop: T1 | T3,
    ledger: Ledger,
    config: Mapping[str, JSONValue],
    *,
    budget: int = 4 * 20_971_520,
    parents: tuple[str, ...] = (),
) -> ExperimentEnv:
    return ExperimentEnv(
        loop,
        ledger,
        runner=loop.run,
        config=config,
        environment_hash="a" * 64,
        budget=budget,
        parents=parents,
    )


def job_for(loop: T3, *, seed: int = 42, fidelity: str = "8m") -> Job:
    return loop.build_job(
        loop.spec.inputs(
            loop.normalize_config({"weights": {"edu": 1, "other": 3}}),
            LoopAction(seed, fidelity),
            "a" * 64,
        )
    )


def test_t3_and_t1_share_ledger_lineage_and_both_fidelities(
    t1: T1,
    t3: T3,
    ledger: Ledger,
) -> None:
    baseline = asyncio.run(environment(t1, ledger, {}).step(LoopAction(42)))
    env = environment(
        t3, ledger, {"weights": {"edu": 1, "other": 3}}, parents=(baseline.run_id,)
    )
    screen = asyncio.run(env.step(LoopAction(42)))
    confirm = asyncio.run(env.step(LoopAction(43, "21m", Phase.CONFIRM)))
    assert screen.result is not None
    assert screen.result.metrics["val_bpb"].value == 1.75
    assert screen.result.observations["mixture_tokens/edu"].value == 2_097_152
    assert screen.result.observations["mixture_tokens/other"].value == 6_291_456
    assert screen.work_units == 8_388_608
    assert confirm.work_units == 20_971_520
    assert ledger.get(screen.run_id).parents == (baseline.run_id,)
    assert ledger.get(screen.run_id).inputs.loop_hash == t3.spec.hash
    assert ledger.get(baseline.run_id).inputs.loop_hash == t1.spec.hash
    assert {run.id for run in ledger.lineage(confirm.run_id)} == {
        baseline.run_id,
        confirm.run_id,
    }


def test_ratios_change_consumed_data_not_validation(t3: T3, ledger: Ledger) -> None:
    scores = []
    samples = []
    for weights in ({"edu": 1, "other": 0}, {"edu": 0, "other": 1}):
        env = environment(t3, ledger, {"weights": weights})
        step = asyncio.run(env.step(LoopAction(42)))
        assert step.result is not None
        scores.append(step.result.metrics["val_bpb"].value)
        samples.append(Sample.from_run(ledger.get(step.run_id), "val_bpb"))
    assert scores == [1.0, 2.0]
    assert Pair(*samples).gain == -1.0


def test_equivalent_ratios_cache_but_replication_is_a_new_attempt(
    t3: T3,
    ledger: Ledger,
) -> None:
    first_env = environment(t3, ledger, {"weights": {"edu": 1, "other": 3}})
    first = asyncio.run(first_env.step(LoopAction(42)))
    other_env = environment(t3, ledger, {"weights": {"other": 6, "edu": 2}})
    cached = asyncio.run(other_env.step(LoopAction(42)))
    replicated = asyncio.run(
        other_env.step(LoopAction(42, purpose=Purpose.REPLICATION))
    )
    assert first.attempt_id == cached.attempt_id != replicated.attempt_id
    assert not cached.executed
    assert cached.work_units == 0
    assert replicated.executed


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"weights": {"edu": 1}},
        {"weights": {"edu": 0, "other": 0}},
        {"weights": {"edu": 1, "held_out": 1}},
        {"weights": {"edu": 1, "other": 2}, "DEPTH": 7},
    ],
)
def test_invalid_candidates_leave_ledger_empty(t3: T3, ledger: Ledger, config) -> None:
    with pytest.raises(ContractError):
        environment(t3, ledger, config)
    assert not ledger.query_all()


def test_recipe_catalog_and_weights_all_enter_identity(t3: T3) -> None:
    original = job_for(t3).inputs
    changed = T3(t3.training, t3.sources, training_config={"DEPTH": 5})
    assert original.candidate_hash != job_for(changed).inputs.candidate_hash
    renamed = T3(t3.training, {"edu": t3.sources["other"], "other": t3.sources["edu"]})
    assert original.data_hash != renamed.spec.data_hash
    assert original.candidate_hash != job_for(renamed).inputs.candidate_hash
    changed_weights = t3.spec.inputs(
        t3.normalize_config({"weights": {"edu": 3, "other": 1}}),
        LoopAction(42),
        "a" * 64,
    )
    assert original.candidate_hash != changed_weights.candidate_hash


def test_machine_paths_do_not_enter_identity(t3: T3, tmp_path: Path) -> None:
    moved = {}
    for name, corpus in t3.sources.items():
        destination = tmp_path / "moved" / name
        shutil.copytree(corpus.root, destination)
        moved[name] = Corpus.load(destination)
    assert T3(t3.training, moved).spec == t3.spec


def test_token_quota_receipt_cannot_be_forged(t3: T3) -> None:
    with pytest.raises(ContractError, match="mixture tokens"):
        t3.evaluate(job_for(t3), {"mixture_tokens": {"edu": 0, "other": 0}})


def test_corrupt_source_fails_and_is_recorded(t3: T3, ledger: Ledger) -> None:
    corpus = t3.sources["other"]
    with (corpus.root / "splits/train.parquet").open("ab") as stream:
        stream.write(b"corruption")
    env = environment(t3, ledger, {"weights": {"edu": 1, "other": 1}}, budget=8_388_608)
    with pytest.raises(ContractError, match="Size or SHA-256 mismatch"):
        asyncio.run(env.step(LoopAction(42)))
    assert len(ledger.query_all(status=Status.FAILED)) == 1
    assert asyncio.run(env.initial_observation()).remaining_budget == 0


def test_train_membership_cannot_smuggle_validation_text(t3: T3) -> None:
    corpus = t3.sources["other"]
    manifest = json.loads((corpus.root / "manifest.json").read_text())
    pq.write_table(
        pq.read_table(corpus.root / "splits/val.parquet"),
        corpus.root / "splits/train.parquet",
    )
    entry = manifest["splits"]["train"]
    path = corpus.root / entry["path"]
    entry["sha256"] = file_hash(path)
    entry["size_bytes"] = path.stat().st_size
    manifest.pop("snapshot_sha256")
    manifest["snapshot_sha256"] = json_hash(manifest)
    (corpus.root / "manifest.json").write_text(json.dumps(manifest))
    loop = T3(t3.training, {**t3.sources, "other": Corpus.load(corpus.root)})
    with pytest.raises(ContractError, match="split policy"):
        run_mixture_training(loop, job_for(loop))


def test_same_corpus_cannot_have_two_names(t3: T3) -> None:
    with pytest.raises(ContractError, match="alias"):
        T3(t3.training, {"a": t3.sources["edu"], "b": t3.sources["edu"]})


def test_held_out_memberships_are_never_opened(t3: T3) -> None:
    for corpus in t3.sources.values():
        (corpus.root / "splits/held_out.parquet").unlink()
    output = run_mixture_training(t3, job_for(t3))
    assert output["mixture_tokens"] == {"edu": 2_097_152, "other": 6_291_456}


def test_modified_program_and_over_budget_actions_do_not_run(
    t3: T3, ledger: Ledger
) -> None:
    job = replace(job_for(t3), program="raise RuntimeError('changed')")
    with pytest.raises(ContractError, match="exact generated"):
        run_mixture_training(t3, job)
    env = environment(t3, ledger, {"weights": {"edu": 1, "other": 3}}, budget=1)
    with pytest.raises(ContractError, match="remaining episode budget"):
        asyncio.run(env.step(LoopAction(42)))
    assert not ledger.query_all()


def test_source_drift_and_changed_policy_are_rejected(t3: T3) -> None:
    corpus = t3.sources["other"]
    path = corpus.root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["policy"]["version"] = 2
    manifest.pop("snapshot_sha256")
    manifest["snapshot_sha256"] = json_hash(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="frozen content-split policy"):
        run_mixture_training(t3, job_for(t3))


def test_pinned_upstream_packer_uses_private_sources_and_fixed_validation(
    t3: T3,
) -> None:
    repository = Path(__file__).parents[2] / "research/autoresearch-mlx"
    if not (repository / "prepare.py").exists():
        pytest.skip("Pinned upstream checkout is not installed")
    source = TrainingSource.load(repository).prepare
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "make_dataloader"
    )
    prepare: dict[str, object] = {"mx": SimpleNamespace(array=np.array, int32=np.int32)}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "packer.py", "exec"),
        prepare,
    )

    class Tokenizer:
        def get_bos_token_id(self) -> int:
            return 0

        def encode(self, texts: list[str], *, prepend: int) -> list[list[int]]:
            return [
                [prepend] + ([202] * 97 if text.startswith("additional") else [101] * 7)
                for text in texts
            ]

    mixture = MixtureLoader(t3, job_for(t3)).bind(prepare)
    assert callable(mixture)
    stream = mixture(Tokenizer(), 16, 2048, "train", buffer_size=4)
    markers = []
    for _ in range(16):
        inputs, targets, _ = next(stream)
        assert np.asarray(inputs).shape == np.asarray(targets).shape == (16, 2048)
        values = set(np.unique(np.asarray(inputs))) - {0}
        assert len(values) == 1
        markers.append(values.pop())
    assert markers.count(101) == 4
    assert markers.count(202) == 12
    validation = prepare["make_dataloader"]
    assert callable(validation)
    validation_stream = validation(Tokenizer(), 16, 2048, "val", buffer_size=4)
    assert isinstance(validation_stream, Iterator)
    inputs, _, _ = next(validation_stream)
    assert set(np.unique(np.asarray(inputs))) == {0, 101}
