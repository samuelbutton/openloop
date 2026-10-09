"""Pure content-addressed identity for experiment inputs."""

# Runtime isinstance checks validate untyped callers at the trust boundary.
# pyright: reportUnnecessaryIsInstance=false

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import TypeGuard

from .errors import InvalidInputError

# Bump only when the canonical encoding or the meaning of a field changes.
# That changes every hash, so it must not follow storage schema changes.
IDENTITY_VERSION = 2

type JSONValue = (
    None
    | bool
    | int
    | float
    | str
    | Mapping[str, JSONValue]
    | tuple[JSONValue, ...]
    | list[JSONValue]
)


class Reproducibility(StrEnum):
    """Whether identical inputs are expected to give identical results."""

    DETERMINISTIC = "deterministic"
    NOISY = "noisy"


def coerce_enum[E: StrEnum](kind: type[E], value: object, label: str) -> E:
    """Convert a string or member, rejecting anything outside the enum."""
    try:
        return kind(value)
    except ValueError:
        allowed = ", ".join(member.value for member in kind)
        raise InvalidInputError(
            f"Invalid {label}; expected one of: {allowed}"
        ) from None


# isinstance alone narrows `object` to Mapping[Unknown, Unknown]; these name the
# element type so strict checking can follow the recursion.
def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_sequence(value: object) -> TypeGuard[tuple[object, ...] | list[object]]:
    return isinstance(value, tuple | list)


def plain(value: object) -> JSONValue:
    """Convert dataclasses, enums, and sequences to JSON-compatible values."""
    if isinstance(value, Enum):
        return plain(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: plain(getattr(value, item.name)) for item in fields(value)}
    if _is_mapping(value):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidInputError("JSON object keys must be strings")
            result[key] = plain(item)
        return result
    if _is_sequence(value):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidInputError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise InvalidInputError(f"Unsupported JSON value: {type(value).__name__}")


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


def freeze(value: JSONValue) -> JSONValue:
    """Make nested mappings and sequences immutable."""
    if isinstance(value, Mapping):
        return freeze_object(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def freeze_object(value: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    """Make a JSON object's nested mappings and sequences immutable."""
    return MappingProxyType({key: freeze(item) for key, item in value.items()})


def check_digest(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise InvalidInputError("Content references must be lowercase SHA-256 digests")


def check_text(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError("Expected a nonempty string")


def check_positive(value: object) -> None:
    if type(value) is not int or value < 1:
        raise InvalidInputError("Expected a positive integer")


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
