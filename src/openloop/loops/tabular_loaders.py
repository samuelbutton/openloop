"""Read verified local benchmark artifacts into immutable validation-only tables.

NAS legacy pickle loading is restricted to the known official artifact hash.
HPO-B reads one meta-split JSON. Nothing downloads data or starts model training.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, runtime_checkable

from openloop.ledger import JSONValue, content_hash
from openloop.ledger.validation import (
    check_digest,
    check_text,
    coerce_enum,
    finite_number,
    plain,
)

from .files import file_digest
from .models import ContractError
from .tabular import (
    NAS_FIDELITIES,
    Benchmark,
    MetaSplit,
    NASDataset,
    RecordedOutcome,
    RecordedTable,
)

NAS_ARTIFACT = "NAS-Bench-201-v1_1-096897.pth"
NAS_SHA256 = "2317a7dc911654322f4f932730a821cf38ecc5b5f1adde0aff26ff6b3f736666"


def _object(value: object) -> Mapping[str, JSONValue]:
    value = plain(value)
    if not isinstance(value, Mapping):
        raise ContractError("Benchmark metadata must be a JSON object")
    return value


def _artifact(
    root: Path, section: str, name: str
) -> tuple[Path, str, Mapping[str, JSONValue]]:
    manifest = _object(json.loads((root / "manifest.json").read_text()))
    entries = manifest.get(section)
    if not isinstance(entries, (tuple, list)):
        raise ContractError("Benchmark manifest must list its artifacts")
    matches = []
    for entry in entries:
        item = _object(entry)
        location = item.get("path")
        if not isinstance(location, str):
            raise ContractError("Benchmark artifact path must be text")
        if Path(location).name == name:
            matches.append((location, item))
    if len(matches) != 1:
        raise ContractError("Benchmark manifest must identify one artifact")
    location, item = matches[0]
    try:
        relative = Path(location).relative_to("data/tabular")
    except ValueError:
        raise ContractError(
            "Benchmark artifact path must be under data/tabular"
        ) from None
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ContractError("Benchmark artifact must stay inside its data root")
    expected = item.get("sha256")
    check_digest(expected)
    assert isinstance(expected, str)
    if path.stat().st_size != item["size_bytes"] or file_digest(path) != expected:
        raise ContractError("Frozen benchmark artifact changed")
    return path, expected, manifest


def load_hpob(
    data_root: Path,
    search_space: str,
    dataset: str,
    *,
    split: MetaSplit | str = MetaSplit.VALIDATION,
) -> RecordedTable:
    """Load one task, preserving the original meta-split and row ordering."""
    split = coerce_enum(MetaSplit, split, "HPO-B meta-split")
    for label in (search_space, dataset):
        check_text(label)
    name = f"meta-{'train' if split is MetaSplit.TRAIN else 'validation'}-dataset.json"
    path, digest, _ = _artifact(data_root, "hpob_extracted_files", name)
    tables = json.loads(path.read_bytes())
    try:
        task = tables[search_space][dataset]
    except KeyError:
        raise ContractError(
            "HPO-B task is absent from the selected meta-split"
        ) from None
    x, y = task["X"], task["y"]
    if not isinstance(x, list) or not x or not isinstance(y, list) or len(x) != len(y):
        raise ContractError("HPO-B task must have aligned nonempty X and y rows")
    candidates: list[JSONValue] = []
    scores = []
    width = None
    for features, response in zip(x, y, strict=True):
        if (
            not isinstance(features, list)
            or not features
            or not isinstance(response, list)
            or len(response) != 1
        ):
            raise ContractError("HPO-B requires feature vectors and scalar responses")
        if width is not None and len(features) != width:
            raise ContractError("HPO-B feature dimensions must agree")
        width = len(features)
        candidates.append(
            tuple(finite_number(value, "HPO-B feature") for value in features)
        )
        scores.append(finite_number(response[0], "HPO-B response"))
    lower, upper = min(scores), max(scores)
    if lower == upper:
        raise ContractError("HPO-B response normalization needs a nonconstant task")
    if file_digest(path) != digest:
        raise ContractError("HPO-B data changed during loading")
    return RecordedTable(
        Benchmark.HPOB,
        {
            "artifact_sha256": digest,
            "meta_split": str(split),
            "search_space": search_space,
            "dataset": dataset,
            "normalization": "(y - task_min) / (task_max - task_min)",
        },
        content_hash({"dependencies": "Python standard library"}),
        tuple(candidates),
        tuple(
            RecordedOutcome(index, "lookup", 0, (value - lower) / (upper - lower))
            for index, value in enumerate(scores)
        ),
    )


@runtime_checkable
class NASArchitecture(Protocol):
    def get_dataset_seeds(self, dataset: str) -> list[int]: ...
    def get_metrics(
        self, dataset: str, setname: str, *, is_random: int
    ) -> Mapping[str, object]: ...


@runtime_checkable
class NASReader(Protocol):
    def __len__(self) -> int: ...
    def arch(self, index: int) -> str: ...
    def query_meta_info_by_index(
        self, arch_index: int, *, hp: str
    ) -> NASArchitecture: ...


def nas201_table(
    api: NASReader,
    *,
    dataset: NASDataset | str,
    artifact_hash: str,
    dependencies_hash: str,
) -> RecordedTable:
    """Snapshot validation outcomes from an already loaded trusted official API."""
    dataset = coerce_enum(NASDataset, dataset, "NAS validation dataset")
    check_digest(artifact_hash)
    candidates: list[JSONValue] = []
    outcomes: list[RecordedOutcome] = []
    for index in range(len(api)):
        candidates.append(api.arch(index))
        for fidelity in NAS_FIDELITIES:
            architecture = api.query_meta_info_by_index(index, hp=fidelity.name)
            for seed in architecture.get_dataset_seeds(str(dataset)):
                metrics = architecture.get_metrics(
                    str(dataset), "x-valid", is_random=seed
                )
                training = architecture.get_metrics(
                    str(dataset), "train", is_random=seed
                )
                seconds = training.get("all_time")
                outcomes.append(
                    RecordedOutcome(
                        index,
                        fidelity.name,
                        seed,
                        finite_number(metrics["accuracy"], "NAS accuracy"),
                        None
                        if seconds is None
                        else finite_number(seconds, "NAS training seconds"),
                    )
                )
    return RecordedTable(
        Benchmark.NAS201,
        {
            "artifact_sha256": artifact_hash,
            "dataset": str(dataset),
            "score_split": "x-valid",
            "seed_selection": "exact recorded seed",
        },
        dependencies_hash,
        tuple(candidates),
        tuple(outcomes),
    )


def load_nas201(
    data_root: Path, *, dataset: NASDataset | str = NASDataset.CIFAR10
) -> RecordedTable:
    """Load x-valid scores and train timing for each exact recorded seed."""
    dataset = coerce_enum(NASDataset, dataset, "NAS validation dataset")
    path, digest, manifest = _artifact(data_root, "downloads", NAS_ARTIFACT)
    if digest != NAS_SHA256:
        raise ContractError("Legacy NAS loading requires the known official artifact")
    packages = {}
    for name in ("nas-bench-201", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            raise ContractError(
                "Install the optional benchmarks dependency group"
            ) from None
        runtime = _object(manifest.get("runtime"))
        expected_packages = _object(runtime.get("packages"))
        if packages[name] != expected_packages.get(name):
            raise ContractError(f"Frozen benchmark dependency changed: {name}")
    # This optional API is absent from the default environment.
    from nas_201_api import NASBench201API  # pyright: ignore[reportMissingImports]

    directory = Path(inspect.getfile(NASBench201API)).parent
    api_files = {
        item.name: file_digest(item) for item in sorted(directory.glob("*.py"))
    }
    if not api_files:
        raise ContractError("NAS API source snapshot is unavailable")
    dependencies = content_hash({"packages": packages, "api_files": api_files})

    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        api = NASBench201API(str(path), verbose=False)
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
    if not isinstance(api, NASReader) or len(api) != 15625:
        raise ContractError("NAS-Bench-201 must contain 15,625 architectures")
    if file_digest(path) != digest:
        raise ContractError("NAS data changed during loading")
    return nas201_table(
        api,
        dataset=dataset,
        artifact_hash=digest,
        dependencies_hash=dependencies,
    )
