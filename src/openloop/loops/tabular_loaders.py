"""Read verified local benchmark artifacts into immutable tables.

`load_nas201` and `load_hpob` read validation scores only. The held-out scores
come from the separately named `load_nas201_held_out` and `load_hpob_held_out`;
`TabularLoop` rejects their tables, and `HeldOutEvaluator` is their only consumer.

NAS legacy pickle loading is restricted to the known official artifact hash.
HPO-B reads one meta-split JSON. Nothing downloads data or starts model training.
Verified tables and parsed splits are cached per artifact digest, so repeated
loads of the same task do not re-read the large files. The cache lives only in
memory for the life of the process; `clear_caches` empties it.
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
    HPOB_HELD_OUT_SPLIT,
    NAS_FIDELITIES,
    NAS_HELD_OUT_SETS,
    NAS_SELECTION_SET,
    Benchmark,
    MetaSplit,
    NASDataset,
    RecordedOutcome,
    RecordedTable,
)

NAS_ARTIFACT = "NAS-Bench-201-v1_1-096897.pth"
NAS_SHA256 = "2317a7dc911654322f4f932730a821cf38ecc5b5f1adde0aff26ff6b3f736666"

# A file whose size, modification time and inode are unchanged since it was last
# hashed is not hashed again. This avoids re-reading 5 GB per load; it detects
# any in-place edit that changes one of those three attributes.
type _FileStamp = tuple[str, int, int, int]
_VERIFIED: dict[_FileStamp, str] = {}
_NAS_TABLES: dict[tuple[str, str, str, str], RecordedTable] = {}
_HPOB_TABLES: dict[tuple[str, str, str, str], RecordedTable] = {}
_HPOB_SPLIT: dict[
    tuple[str, str], Mapping[str, Mapping[str, Mapping[str, object]]]
] = {}


def clear_caches() -> None:
    """Forget verified digests, parsed splits, and loaded tables."""
    _VERIFIED.clear()
    _NAS_TABLES.clear()
    _HPOB_TABLES.clear()
    _HPOB_SPLIT.clear()


def _digest(path: Path) -> str:
    status = path.stat()
    stamp = (str(path), status.st_size, status.st_mtime_ns, status.st_ino)
    cached = _VERIFIED.get(stamp)
    if cached is None:
        cached = _VERIFIED[stamp] = file_digest(path)
    return cached


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
    if path.stat().st_size != item["size_bytes"] or _digest(path) != expected:
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
    name = f"meta-{'train' if split is MetaSplit.TRAIN else 'validation'}-dataset.json"
    return _hpob_table(data_root, name, str(split), search_space, dataset)


def load_hpob_held_out(
    data_root: Path, search_space: str, dataset: str
) -> RecordedTable:
    """Load one meta-test task. Only `HeldOutEvaluator` may consume the result.

    This is the sole function that opens the meta-test file.
    """
    return _hpob_table(
        data_root, "meta-test-dataset.json", "test", search_space, dataset
    )


def _hpob_table(
    data_root: Path, name: str, split: str, search_space: str, dataset: str
) -> RecordedTable:
    for label in (search_space, dataset):
        check_text(label)
    path, digest, _ = _artifact(data_root, "hpob_extracted_files", name)
    key = (str(path), digest, search_space, dataset)
    cached = _HPOB_TABLES.get(key)
    if cached is not None:
        return cached
    # Only the most recent split stays parsed: a 346 MB file is several GB in memory.
    split_key = (str(path), digest)
    tables = _HPOB_SPLIT.get(split_key)
    if tables is None:
        tables = json.loads(path.read_bytes())
        if _digest(path) != digest:
            raise ContractError("HPO-B data changed during loading")
        _HPOB_SPLIT.clear()
        _HPOB_SPLIT[split_key] = tables
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
    provenance: dict[str, JSONValue] = {
        "artifact_sha256": digest,
        "meta_split": split,
        "search_space": search_space,
        "dataset": dataset,
        "normalization": "(y - task_min) / (task_max - task_min)",
    }
    if split == "test":
        provenance["score_split"] = HPOB_HELD_OUT_SPLIT
    table = RecordedTable(
        Benchmark.HPOB,
        provenance,
        content_hash({"dependencies": "Python standard library"}),
        tuple(candidates),
        tuple(
            RecordedOutcome(index, "lookup", 0, (value - lower) / (upper - lower))
            for index, value in enumerate(scores)
        ),
    )
    _HPOB_TABLES[key] = table
    return table


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
    return _nas_table(api, dataset, artifact_hash, dependencies_hash, held_out=False)


def nas201_held_out_table(
    api: NASReader,
    *,
    dataset: NASDataset | str,
    artifact_hash: str,
    dependencies_hash: str,
) -> RecordedTable:
    """Snapshot held-out outcomes: `ori-test` for cifar10-valid, else `x-test`."""
    return _nas_table(api, dataset, artifact_hash, dependencies_hash, held_out=True)


def _nas_table(
    api: NASReader,
    dataset: NASDataset | str,
    artifact_hash: str,
    dependencies_hash: str,
    *,
    held_out: bool,
) -> RecordedTable:
    dataset = coerce_enum(NASDataset, dataset, "NAS validation dataset")
    score_set = NAS_HELD_OUT_SETS[dataset] if held_out else NAS_SELECTION_SET
    check_digest(artifact_hash)
    candidates: list[JSONValue] = []
    outcomes: list[RecordedOutcome] = []
    for index in range(len(api)):
        candidates.append(api.arch(index))
        for fidelity in NAS_FIDELITIES:
            architecture = api.query_meta_info_by_index(index, hp=fidelity.name)
            for seed in architecture.get_dataset_seeds(str(dataset)):
                metrics = architecture.get_metrics(
                    str(dataset), score_set, is_random=seed
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
            "score_split": score_set,
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
    return _load_nas(data_root, dataset, held_out=False)


def load_nas201_held_out(
    data_root: Path, *, dataset: NASDataset | str = NASDataset.CIFAR10
) -> RecordedTable:
    """Load held-out scores. Only `HeldOutEvaluator` may consume the result.

    Uses `ori-test` for cifar10-valid and `x-test` for cifar100 and ImageNet16-120.
    """
    return _load_nas(data_root, dataset, held_out=True)


def _load_nas(
    data_root: Path, dataset: NASDataset | str, *, held_out: bool
) -> RecordedTable:
    dataset = coerce_enum(NASDataset, dataset, "NAS validation dataset")
    path, digest, manifest = _artifact(data_root, "downloads", NAS_ARTIFACT)
    if digest != NAS_SHA256:
        raise ContractError("Legacy NAS loading requires the known official artifact")
    key = (str(path), digest, str(dataset), "held-out" if held_out else "validation")
    cached = _NAS_TABLES.get(key)
    if cached is not None:
        return cached
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
    build = nas201_held_out_table if held_out else nas201_table
    table = build(
        api,
        dataset=dataset,
        artifact_hash=digest,
        dependencies_hash=dependencies,
    )
    _NAS_TABLES[key] = table
    return table
