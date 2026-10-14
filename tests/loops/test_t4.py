"""Offline T4 contract tests with small recorded tables and temporary ledgers."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.decider import Pair, StagePlan
from openloop.ledger import (
    Direction,
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    Metric,
    Reproducibility,
    Status,
)
from openloop.loops import (
    HPOB,
    ContractError,
    ExperimentEnv,
    HeldOutEvaluator,
    Job,
    LoopAction,
    NASBench201,
    RunContext,
    TabularLoop,
)
from openloop.loops import tabular_loaders as loaders
from openloop.loops.files import file_digest
from openloop.loops.tabular import RecordedOutcome, RecordedTable
from openloop.loops.tabular_loaders import (
    NASArchitecture,
    load_hpob_held_out,
    load_nas201,
    nas201_held_out_table,
    nas201_table,
)


class TinyArchitecture:
    def __init__(self, index: int, epochs: int) -> None:
        self.index = index
        self.epochs = epochs

    def get_dataset_seeds(self, dataset: str) -> list[int]:
        assert dataset == "cifar10-valid"
        return [777, 888] if self.epochs == 200 else [777]

    def get_metrics(
        self, dataset: str, setname: str, *, is_random: int
    ) -> Mapping[str, object]:
        assert dataset == "cifar10-valid"
        assert setname in {"x-valid", "train"}, "Held-out metrics must not be queried"
        assert type(is_random) is int
        return {
            "accuracy": 70 + self.index + self.epochs / 20 + (is_random == 888),
            "all_time": self.epochs * 10.0,
        }


class TinyNAS:
    def __len__(self) -> int:
        return 2

    def arch(self, index: int) -> str:
        return f"architecture {index}"

    def query_meta_info_by_index(self, arch_index: int, *, hp: str) -> NASArchitecture:
        return TinyArchitecture(arch_index, int(hp))


class HeldOutArchitecture(TinyArchitecture):
    """Answers only the held-out set for its dataset, and timing from `train`."""

    def __init__(self, index: int, epochs: int, dataset: str, held_out: str) -> None:
        super().__init__(index, epochs)
        self.dataset = dataset
        self.held_out = held_out
        self.queried: list[str] = []

    def get_dataset_seeds(self, dataset: str) -> list[int]:
        return [777, 888] if self.epochs == 200 else [777]

    def get_metrics(
        self, dataset: str, setname: str, *, is_random: int
    ) -> Mapping[str, object]:
        assert dataset == self.dataset
        assert setname in {self.held_out, "train"}, setname
        return {"accuracy": 60 + self.index - self.epochs / 100, "all_time": 1.0}


class TinyHeldOutNAS(TinyNAS):
    def __init__(self, dataset: str, held_out: str) -> None:
        self.dataset = dataset
        self.held_out = held_out

    def query_meta_info_by_index(self, arch_index: int, *, hp: str) -> NASArchitecture:
        return HeldOutArchitecture(arch_index, int(hp), self.dataset, self.held_out)


@pytest.fixture(autouse=True)
def fresh_caches() -> None:
    loaders.clear_caches()


@pytest.fixture
def nas_table() -> RecordedTable:
    return nas201_table(
        TinyNAS(),
        dataset="cifar10-valid",
        artifact_hash="a" * 64,
        dependencies_hash="b" * 64,
    )


@pytest.fixture
def hpob_data(tmp_path: Path) -> Path:
    directory = tmp_path / "hpob/hpob-data"
    directory.mkdir(parents=True)
    task = {"X": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]], "y": [[0.2], [0.5], [0.8]]}
    files = []
    for name in ("meta-train-dataset.json", "meta-validation-dataset.json"):
        path = directory / name
        path.write_text(json.dumps({"space": {"task": task, "other": task}}))
        files.append(
            {
                "path": f"data/tabular/hpob/hpob-data/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
        )
    (directory / "meta-test-dataset.json").write_text("must not be opened")
    (tmp_path / "manifest.json").write_text(json.dumps({"hpob_extracted_files": files}))
    return tmp_path


def test_nas_fidelities_seeds_metrics_and_cache(
    nas_table: RecordedTable, ledger: Ledger
) -> None:
    loop = NASBench201(nas_table)
    env = ExperimentEnv(
        loop,
        ledger,
        runner=loop.run,
        config={"architecture_index": 0},
        environment_hash="c" * 64,
        budget=412,
    )
    screen = asyncio.run(env.step(LoopAction(777)))
    confirm = asyncio.run(env.step(LoopAction(777, "200", "confirm")))
    cached = asyncio.run(env.step(LoopAction(777, "200", "confirm")))
    other = asyncio.run(env.step(LoopAction(888, "200", "confirm")))
    assert screen.work_units == 12
    assert confirm.work_units == other.work_units == 200
    assert not cached.executed
    assert cached.work_units == 0
    assert cached.attempt_id == confirm.attempt_id
    assert loop.spec.reproducibility is Reproducibility.DETERMINISTIC
    assert loop.seeds(0) == (777,)
    assert loop.seeds(0, fidelity="200") == (777, 888)
    assert confirm.result is not None
    assert other.result is not None
    metric = confirm.result.metrics["validation_score"]
    assert metric.value == 80
    assert metric.direction is Direction.MAXIMIZE
    assert metric.sample_count == 1
    assert other.result.metrics["validation_score"].value == 81
    assert confirm.result.observations["recorded_training_seconds"].value == 2000
    assert confirm.result.observations["lookup_seconds"].value < 1


def test_hpob_load_and_contract(hpob_data: Path, ledger: Ledger) -> None:
    loop = HPOB.load(hpob_data, "space", "task")
    env = ExperimentEnv(
        loop,
        ledger,
        runner=loop.run,
        config={"configuration_index": 1},
        environment_hash="c" * 64,
        budget=2,
    )
    first = asyncio.run(env.step(LoopAction(0)))
    second = asyncio.run(env.step(LoopAction(0)))
    assert first.result is not None
    assert first.result.metrics["validation_score"].value == pytest.approx(0.5)
    assert not second.executed
    assert "recorded_training_seconds" not in first.result.observations
    assert loop.candidates[1] == (0.25, 0.5)
    assert loop.spec.budget_unit == "lookups"
    task = loop.spec.execution["task"]
    assert isinstance(task, Mapping)
    assert task["meta_split"] == "validation"
    inputs = loop.spec.inputs(
        loop.normalize_config({"configuration_index": 1}), LoopAction(1), "c" * 64
    )
    with pytest.raises(ContractError, match="No recorded outcome"):
        loop.build_job(inputs)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"architecture_index": True},
        {"architecture_index": -1},
        {"architecture_index": 2},
        {"architecture_index": 0.5},
        {"architecture_index": 0, "dataset": "cifar10"},
        {"score": 100},
    ],
)
def test_invalid_nas_candidates_leave_ledger_unchanged(
    nas_table: RecordedTable, ledger: Ledger, config
) -> None:
    with pytest.raises(ContractError):
        ExperimentEnv(
            NASBench201(nas_table),
            ledger,
            runner=NASBench201(nas_table).run,
            config=config,
            environment_hash="c" * 64,
            budget=12,
        )
    assert ledger.query_all() == ()


def test_evaluator_rejects_forged_score_program_and_frozen_inputs(
    nas_table: RecordedTable,
) -> None:
    loop = NASBench201(nas_table)
    inputs = loop.spec.inputs(
        loop.normalize_config({"architecture_index": 0}), LoopAction(777), "c" * 64
    )
    job = loop.build_job(inputs)
    with pytest.raises(ContractError):
        loop.evaluate(job, {"work_units": 12, "lookup_seconds": 0.1, "score": 100})
    with pytest.raises(ContractError):
        loop.evaluate(job, {"work_units": 200, "lookup_seconds": 0.1})
    with pytest.raises(ContractError):
        loop.evaluate(job, {"work_units": 12, "lookup_seconds": -1})
    with pytest.raises(ContractError):
        asyncio.run(
            loop.run(Job(inputs, "print('score')"), RunContext("run", "attempt"))
        )
    with pytest.raises(ContractError):
        loop.build_job(replace(inputs, data_hash="f" * 64))


def test_failure_is_recorded_not_scored(
    nas_table: RecordedTable, ledger: Ledger
) -> None:
    async def forged(job, context):
        return {"work_units": 12, "lookup_seconds": 0.1, "score": 100}

    loop = NASBench201(nas_table)
    env = ExperimentEnv(
        loop,
        ledger,
        runner=forged,
        config={"architecture_index": 0},
        environment_hash="c" * 64,
        budget=12,
    )
    with pytest.raises(ContractError):
        asyncio.run(env.step(LoopAction(777)))
    runs = ledger.query_all()
    assert len(runs) == 1
    assert runs[0].status is Status.FAILED
    assert runs[0].result is None


def test_task_identity_and_data_drift(hpob_data: Path) -> None:
    first = HPOB.load(hpob_data, "space", "task")
    other = HPOB.load(hpob_data, "space", "other")
    train = HPOB.load(hpob_data, "space", "task", split="train")
    config = {"configuration_index": 0}
    assert first.spec.candidate_hash(config) != other.spec.candidate_hash(config)
    assert first.spec.hash != train.spec.hash
    path = hpob_data / "hpob/hpob-data/meta-validation-dataset.json"
    path.write_text("changed")
    with pytest.raises(ContractError, match="artifact changed"):
        HPOB.load(hpob_data, "space", "task")


def test_table_is_frozen_and_order_does_not_change_identity(
    nas_table: RecordedTable,
) -> None:
    reversed_table = replace(nas_table, outcomes=tuple(reversed(nas_table.outcomes)))
    assert nas_table.hash == reversed_table.hash
    changed = replace(
        nas_table,
        outcomes=(replace(nas_table.outcomes[0], score=60), *nas_table.outcomes[1:]),
    )
    assert changed.hash != nas_table.hash
    with pytest.raises(ContractError):
        replace(nas_table, outcomes=(*nas_table.outcomes, nas_table.outcomes[0]))
    with pytest.raises(ContractError):
        HPOB(nas_table)
    with pytest.raises(TypeError):
        nas_table.provenance["dataset"] = "test"  # pyright: ignore[reportIndexIssue]


def test_heldout_and_missing_data_are_rejected(
    hpob_data: Path, nas_table: RecordedTable
) -> None:
    with pytest.raises(InvalidInputError, match="NAS validation dataset"):
        nas201_table(
            TinyNAS(),
            dataset="cifar10",
            artifact_hash="a" * 64,
            dependencies_hash="b" * 64,
        )
    with pytest.raises(InvalidInputError, match="HPO-B meta-split"):
        HPOB.load(hpob_data, "space", "task", split="test")
    with pytest.raises(ContractError):
        HPOB.load(hpob_data, "missing", "task")
    loop = NASBench201(nas_table)
    inputs = loop.spec.inputs(
        loop.normalize_config({"architecture_index": 0}), LoopAction(888), "c" * 64
    )
    with pytest.raises(ContractError):
        loop.build_job(inputs)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_nonfinite_recorded_values_are_rejected(score) -> None:
    with pytest.raises(InvalidInputError, match="finite number"):
        RecordedOutcome(0, "lookup", 0, score)


@given(order=st.permutations(tuple(range(6))))
def test_row_order_hash_invariance(order: tuple[int, ...]) -> None:
    table = nas201_table(
        TinyNAS(),
        dataset="cifar10-valid",
        artifact_hash="a" * 64,
        dependencies_hash="b" * 64,
    )
    permuted = replace(table, outcomes=tuple(table.outcomes[index] for index in order))
    assert permuted.hash == table.hash


def test_unverified_nas_pickle_is_rejected_before_import(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "NAS-Bench-201-v1_1-096897.pth"
    path.write_bytes(b"untrusted pickle")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "downloads": [
                    {
                        "path": f"data/tabular/raw/{path.name}",
                        "size_bytes": path.stat().st_size,
                        "sha256": file_digest(path),
                    }
                ]
            }
        )
    )
    with pytest.raises(ContractError, match="known official artifact"):
        load_nas201(tmp_path)


@pytest.mark.parametrize(
    "task",
    [
        {"X": [], "y": []},
        {"X": [[0], [0, 1]], "y": [[0], [1]]},
        {"X": [[0], [1]], "y": [[1], [1]]},
        {"X": [[0], [1]], "y": [[0, 1], [1]]},
    ],
)
def test_invalid_hpob_tables_are_rejected(tmp_path: Path, task) -> None:
    directory = tmp_path / "hpob/hpob-data"
    directory.mkdir(parents=True)
    path = directory / "meta-validation-dataset.json"
    path.write_text(json.dumps({"space": {"task": task}}))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "hpob_extracted_files": [
                    {
                        "path": f"data/tabular/hpob/hpob-data/{path.name}",
                        "size_bytes": path.stat().st_size,
                        "sha256": file_digest(path),
                    }
                ]
            }
        )
    )
    with pytest.raises(ContractError):
        HPOB.load(tmp_path, "space", "task")


@pytest.mark.parametrize(
    ("dataset", "held_out"),
    [
        ("cifar10-valid", "ori-test"),
        ("cifar100", "x-test"),
        ("ImageNet16-120", "x-test"),
    ],
)
def test_nas_held_out_set_per_dataset_and_selection_rejects_it(
    dataset: str, held_out: str
) -> None:
    table = nas201_held_out_table(
        TinyHeldOutNAS(dataset, held_out),
        dataset=dataset,
        artifact_hash="a" * 64,
        dependencies_hash="b" * 64,
    )
    assert table.held_out
    assert table.metric_split == "held_out"
    assert table.provenance["score_split"] == held_out
    assert table.lookup(0, "12", 777).score == 60 - 0.12
    with pytest.raises(ContractError, match="HeldOutEvaluator"):
        NASBench201(table)
    with pytest.raises(ContractError, match="HeldOutEvaluator"):
        TabularLoop(table)


def test_validation_table_never_requests_held_out_sets(
    nas_table: RecordedTable,
) -> None:
    assert not nas_table.held_out
    assert nas_table.provenance["score_split"] == "x-valid"
    assert nas_table.metric_split == "validation"


def _held_out_setup(ledger: Ledger, nas_table: RecordedTable):
    held = nas201_held_out_table(
        TinyHeldOutNAS("cifar10-valid", "ori-test"),
        dataset="cifar10-valid",
        artifact_hash="a" * 64,
        dependencies_hash="b" * 64,
    )
    selection = NASBench201(nas_table)
    baseline = {"architecture_index": 0}
    chosen = {"architecture_index": 1}
    evaluator = HeldOutEvaluator(
        held,
        selection,
        ledger,
        baseline=baseline,
        candidates={selection.spec.candidate_hash(chosen): chosen},
    )
    plan = StagePlan(
        evaluator.baseline_template(environment_hash="c" * 64, fidelity="200"),
        Metric(0, "percent", Direction.MAXIMIZE, split="held_out"),
        (777, 888),
        metric_name="validation_score",
    )
    return selection, evaluator, plan, chosen


def test_held_out_evaluator_scores_once_as_final_evaluator(
    nas_table: RecordedTable, ledger: Ledger
) -> None:
    selection, evaluator, plan, chosen = _held_out_setup(ledger, nas_table)
    candidate = selection.spec.candidate_hash(chosen)
    # The decider's candidate identity is unchanged by the split.
    assert plan.template.candidate_hash == selection.spec.candidate_hash(
        {"architecture_index": 0}
    )
    pairs = asyncio.run(evaluator(candidate, plan))
    assert len(pairs) == 2
    assert all(isinstance(pair, Pair) for pair in pairs)
    assert [pair.baseline.inputs.seed for pair in pairs] == [777, 888]
    # Held-out scores: 60 + index - 200 / 100; the selection table is untouched.
    assert [pair.baseline.metric.value for pair in pairs] == [58, 58]
    assert [pair.candidate.metric.value for pair in pairs] == [59, 59]
    assert {pair.candidate.metric.split for pair in pairs} == {"held_out"}
    assert pairs[0].candidate.inputs.candidate_hash == candidate
    runs = ledger.query_all()
    assert len(runs) == 4
    assert {run.inputs.phase for run in runs} == {"held_out"}
    assert {run.inputs.data_hash for run in runs} == {evaluator.spec.data_hash}
    assert evaluator.spec.data_hash != selection.spec.data_hash
    with pytest.raises(InvalidTransitionError, match="read once"):
        asyncio.run(evaluator(candidate, plan))


def test_held_out_evaluator_rejects_mismatched_inputs(
    nas_table: RecordedTable, ledger: Ledger
) -> None:
    selection, evaluator, plan, chosen = _held_out_setup(ledger, nas_table)
    candidate = selection.spec.candidate_hash(chosen)
    with pytest.raises(ContractError, match="requires a held-out table"):
        HeldOutEvaluator(
            nas_table,
            selection,
            ledger,
            baseline={"architecture_index": 0},
            candidates={},
        )
    with pytest.raises(ContractError, match="does not match its configuration"):
        HeldOutEvaluator(
            evaluator._table,
            selection,
            ledger,
            baseline={"architecture_index": 0},
            candidates={"f" * 64: chosen},
        )
    selection_data = replace(plan.template, data_hash=selection.spec.data_hash)
    forged = replace(plan, template=selection_data)
    with pytest.raises(ContractError, match="differs from the frozen held-out task"):
        asyncio.run(evaluator(candidate, forged))
    assert ledger.query_all() == ()


def test_held_out_evaluator_requires_the_same_candidates(
    hpob_data: Path, nas_table: RecordedTable, ledger: Ledger
) -> None:
    selection = NASBench201(nas_table)
    fewer = nas201_held_out_table(
        TinyHeldOutNAS("cifar10-valid", "ori-test"),
        dataset="cifar10-valid",
        artifact_hash="a" * 64,
        dependencies_hash="b" * 64,
    )
    other = replace(
        fewer,
        candidates=("changed", "architecture 1"),
        outcomes=fewer.outcomes,
    )
    with pytest.raises(ContractError, match="candidates differ"):
        HeldOutEvaluator(
            other, selection, ledger, baseline={"architecture_index": 0}, candidates={}
        )


def _write_hpob(directory: Path, names: tuple[str, ...]) -> Path:
    data = directory / "hpob/hpob-data"
    data.mkdir(parents=True)
    task = {"X": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]], "y": [[0.2], [0.5], [0.8]]}
    files = []
    for name in names:
        path = data / name
        path.write_text(json.dumps({"space": {"task": task}}))
        files.append(
            {
                "path": f"data/tabular/hpob/hpob-data/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps({"hpob_extracted_files": files})
    )
    return directory


def test_hpob_meta_test_only_through_the_held_out_loader(
    tmp_path: Path, ledger: Ledger
) -> None:
    root = _write_hpob(
        tmp_path, ("meta-validation-dataset.json", "meta-test-dataset.json")
    )
    with pytest.raises(InvalidInputError, match="HPO-B meta-split"):
        HPOB.load(root, "space", "task", split="test")
    held = load_hpob_held_out(root, "space", "task")
    assert held.held_out
    assert held.provenance["score_split"] == "meta-test"
    with pytest.raises(ContractError, match="HeldOutEvaluator"):
        HPOB(held)
    selection = HPOB.load(root, "space", "task")
    assert selection.spec.data_hash != held.hash
    config = {"configuration_index": 2}
    evaluator = HeldOutEvaluator(
        held,
        selection,
        ledger,
        baseline={"configuration_index": 0},
        candidates={selection.spec.candidate_hash(config): config},
    )
    plan = StagePlan(
        evaluator.baseline_template(environment_hash="c" * 64),
        Metric(0, "normalized response", Direction.MAXIMIZE, split="held_out"),
        (0,),
        metric_name="validation_score",
    )
    (pair,) = asyncio.run(evaluator(selection.spec.candidate_hash(config), plan))
    assert pair.baseline.metric.value == 0
    assert pair.candidate.metric.value == 1


def test_loaders_cache_by_artifact_and_reread_after_change(
    hpob_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parses = 0
    real = json.loads

    def counting(data, *args, **kwargs):
        nonlocal parses
        parses += isinstance(data, bytes)  # split files; manifests are read as text
        return real(data, *args, **kwargs)

    hashes = 0
    real_digest = loaders.file_digest

    def counting_digest(path: Path) -> str:
        nonlocal hashes
        hashes += 1
        return real_digest(path)

    monkeypatch.setattr(loaders.json, "loads", counting)
    monkeypatch.setattr(loaders, "file_digest", counting_digest)
    first = HPOB.load(hpob_data, "space", "task")
    again = HPOB.load(hpob_data, "space", "task")
    other = HPOB.load(hpob_data, "space", "other")
    assert first.spec.data_hash == again.spec.data_hash
    assert other.spec.data_hash != first.spec.data_hash
    # The split parsed once for three loads, and the file hashed once.
    assert parses == 1
    assert hashes == 1
    path = hpob_data / "hpob/hpob-data/meta-validation-dataset.json"
    path.write_text("changed")
    with pytest.raises(ContractError, match="artifact changed"):
        HPOB.load(hpob_data, "space", "task")


def test_seeds_use_a_prebuilt_map_and_fidelities_are_not_seed_paired(
    nas_table: RecordedTable,
) -> None:
    assert nas_table.seeds(0, "12") == (777,)
    assert nas_table.seeds(0, "200") == (777, 888)
    assert nas_table.seeds(5, "12") == ()
    # Recorded seed sets differ by fidelity, so a seed is not a matched pair.
    assert set(nas_table.seeds(0, "12")) != set(nas_table.seeds(0, "200"))
