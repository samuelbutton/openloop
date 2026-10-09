"""Pure content-addressed identity for experiment inputs."""

# Runtime isinstance checks validate untyped callers at the trust boundary.
# pyright: reportUnnecessaryIsInstance=false

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .errors import InvalidInputError
from .validation import (
    JSONValue,
    check_digest,
    check_positive,
    check_text,
    coerce_enum,
    freeze_object,
    plain,
)

# Bump only when the canonical encoding or the meaning of a field changes.
# That changes every hash, so it must not follow storage schema changes.
IDENTITY_VERSION = 2


class Reproducibility(StrEnum):
    """Whether identical inputs are expected to give identical results."""

    DETERMINISTIC = "deterministic"
    NOISY = "noisy"


def canonical_json(value: object) -> str:
    """Encode declared inputs without key-order or whitespace differences."""
    return json.dumps(
        plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentInputs:
    """All declared result-affecting inputs. Hashes refer to complete snapshots."""

    source_hash: str
    dependencies_hash: str
    data_hash: str
    environment_hash: str
    evaluator_hash: str
    loop_hash: str
    seed: int
    fidelity: str
    budget_unit: str
    budget_amount: int
    reproducibility: Reproducibility
    config: Mapping[str, JSONValue] = field(default_factory=dict[str, JSONValue])
    execution: Mapping[str, JSONValue] = field(default_factory=dict[str, JSONValue])
    tokenizer_hash: str | None = None
    phase: str = "screen"

    def __post_init__(self) -> None:
        for name in (
            "source_hash",
            "dependencies_hash",
            "data_hash",
            "environment_hash",
            "evaluator_hash",
            "loop_hash",
        ):
            check_digest(getattr(self, name))
        if self.tokenizer_hash is not None:
            check_digest(self.tokenizer_hash)
        if type(self.seed) is not int or self.seed < 0:
            raise InvalidInputError("Seed must be a nonnegative integer")
        check_positive(self.budget_amount)
        for value in (self.fidelity, self.budget_unit, self.phase):
            check_text(value)
        object.__setattr__(
            self,
            "reproducibility",
            coerce_enum(Reproducibility, self.reproducibility, "reproducibility"),
        )
        for name, value in (("config", self.config), ("execution", self.execution)):
            if not isinstance(value, Mapping):
                raise InvalidInputError(f"{name} must be a JSON object")
            canonical_json(value)
            object.__setattr__(self, name, freeze_object(value))

    @property
    def manifest(self) -> Mapping[str, JSONValue]:
        return {"identity_version": IDENTITY_VERSION, "inputs": plain(self)}

    @property
    def hash(self) -> str:
        return content_hash(self.manifest)

    @property
    def candidate_hash(self) -> str:
        """Identify the code and configuration, ignoring seed, fidelity, and data."""
        return content_hash(
            {
                "identity_version": IDENTITY_VERSION,
                "source_hash": self.source_hash,
                "dependencies_hash": self.dependencies_hash,
                "config": self.config,
            }
        )
