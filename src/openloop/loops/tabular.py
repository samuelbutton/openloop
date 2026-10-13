"""Immutable tabular outcomes. No files, optional libraries, or oracle access.

Rows describe recorded benchmark measurements, not new model training.
A table whose scores come from a held-out split marks itself in its provenance.
`TabularLoop` refuses such tables, so selection code cannot read them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from openloop.ledger import JSONValue, content_hash
from openloop.ledger.validation import (
    check_digest,
    coerce_enum,
    finite_number,
    freeze,
    freeze_object,
)

from .models import ContractError, Fidelity


class Benchmark(StrEnum):
    NAS201 = "nas201"
    HPOB = "hpob"


class TabularSetting(StrEnum):
    ARCHITECTURE = "architecture_index"
    CONFIGURATION = "configuration_index"


class NASDataset(StrEnum):
    CIFAR10 = "cifar10-valid"
    CIFAR100 = "cifar100"
    IMAGENET = "ImageNet16-120"


class MetaSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


NAS_FIDELITIES = (Fidelity("12", 12), Fidelity("200", 200))
HPOB_FIDELITIES = (Fidelity("lookup", 1),)

NAS_SELECTION_SET = "x-valid"
# For cifar100 and ImageNet16-120 the x-valid and x-test sets are the two halves
# of the original test set, so these held-out scores are less separate than
# cifar10-valid's, whose ori-test set is disjoint from its x-valid training split.
NAS_HELD_OUT_SETS: Mapping[NASDataset, str] = MappingProxyType(
    {
        NASDataset.CIFAR10: "ori-test",
        NASDataset.CIFAR100: "x-test",
        NASDataset.IMAGENET: "x-test",
    }
)
HPOB_HELD_OUT_SPLIT = "meta-test"
HELD_OUT_SCORE_SPLITS = frozenset({*NAS_HELD_OUT_SETS.values(), HPOB_HELD_OUT_SPLIT})


@dataclass(frozen=True)
class RecordedOutcome:
    index: int
    fidelity: str
    seed: int
    score: float
    training_seconds: float | None = None

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in (self.index, self.seed)
        ):
            raise ContractError(
                "Recorded indices and seeds must be nonnegative integers"
            )
        score = finite_number(self.score, "Recorded score")
        object.__setattr__(self, "score", score)
        if self.training_seconds is not None:
            seconds = finite_number(self.training_seconds, "Recorded training seconds")
            if seconds < 0:
                raise ContractError("Recorded training seconds must be nonnegative")
            object.__setattr__(self, "training_seconds", seconds)

    @property
    def key(self) -> tuple[int, str, int]:
        return self.index, self.fidelity, self.seed


@dataclass(frozen=True)
class RecordedTable:
    """One immutable task, including its configurations and all selectable outcomes."""

    benchmark: Benchmark
    provenance: Mapping[str, JSONValue]
    dependencies_hash: str
    candidates: tuple[JSONValue, ...]
    outcomes: tuple[RecordedOutcome, ...]
    _lookup: Mapping[tuple[int, str, int], RecordedOutcome] = field(
        init=False, repr=False, compare=False
    )
    _seeds: Mapping[tuple[int, str], tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )
    _hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "benchmark", coerce_enum(Benchmark, self.benchmark, "benchmark")
        )
        check_digest(self.dependencies_hash)
        candidates = tuple(freeze(value) for value in self.candidates)
        if not candidates:
            raise ContractError("A benchmark task needs candidates")
        if not all(isinstance(row, RecordedOutcome) for row in self.outcomes):
            raise ContractError("Benchmark outcomes must be typed records")
        outcomes = tuple(sorted(self.outcomes, key=lambda row: row.key))
        lookup = {row.key: row for row in outcomes}
        if not outcomes or len(lookup) != len(outcomes):
            raise ContractError("Recorded outcomes must be nonempty and unique")
        for row in outcomes:
            if row.index >= len(candidates) or row.fidelity not in {
                f.name for f in self.fidelities
            }:
                raise ContractError(
                    "Recorded outcome has an unknown candidate or fidelity"
                )
            maximum = 100 if self.benchmark is Benchmark.NAS201 else 1
            if not 0 <= row.score <= maximum:
                raise ContractError("Recorded score is outside the benchmark range")
            if self.benchmark is Benchmark.HPOB and row.seed != 0:
                raise ContractError("HPO-B lookup outcomes use seed 0")
        if {(row.index, row.fidelity) for row in outcomes} != {
            (index, fidelity.name)
            for index in range(len(candidates))
            for fidelity in self.fidelities
        }:
            raise ContractError("Every candidate must have each declared fidelity")
        provenance = freeze_object(self.provenance)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))
        seeds: dict[tuple[int, str], list[int]] = {}
        for row in outcomes:
            seeds.setdefault((row.index, row.fidelity), []).append(row.seed)
        object.__setattr__(
            self,
            "_seeds",
            MappingProxyType({key: tuple(value) for key, value in seeds.items()}),
        )
        object.__setattr__(
            self,
            "_hash",
            content_hash(
                {
                    "benchmark": self.benchmark,
                    "provenance": provenance,
                    "candidates": candidates,
                    "outcomes": outcomes,
                }
            ),
        )

    @property
    def fidelities(self) -> tuple[Fidelity, ...]:
        return NAS_FIDELITIES if self.benchmark is Benchmark.NAS201 else HPOB_FIDELITIES

    @property
    def setting(self) -> TabularSetting:
        return (
            TabularSetting.ARCHITECTURE
            if self.benchmark is Benchmark.NAS201
            else TabularSetting.CONFIGURATION
        )

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def held_out(self) -> bool:
        """Whether the scores come from a split that selection must never read."""
        return self.provenance.get("score_split") in HELD_OUT_SCORE_SPLITS

    @property
    def metric_split(self) -> str:
        return "held_out" if self.held_out else "validation"

    def seeds(self, index: int, fidelity: str) -> tuple[int, ...]:
        """Return exact recorded seeds, in ascending order; never remap a seed.

        Fidelities have their own seed sets, so NAS 12- and 200-epoch rows are
        not seed-paired.
        """
        return self._seeds.get((index, fidelity), ())

    def lookup(self, index: int, fidelity: str, seed: int) -> RecordedOutcome:
        """Reject missing measurements instead of drawing or averaging seeds."""
        try:
            return self._lookup[index, fidelity, seed]
        except KeyError:
            raise ContractError(
                "No recorded outcome for this candidate, fidelity, and seed"
            ) from None
