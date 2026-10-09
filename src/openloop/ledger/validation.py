"""Public validators shared by the ledger and its callers.

These pure functions check untrusted values at a trust boundary and raise
`InvalidInputError`. They hold no state and touch no storage.
"""

# Runtime isinstance checks validate untyped callers at the trust boundary.
# pyright: reportUnnecessaryIsInstance=false

import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import TypeGuard

from .errors import InvalidInputError

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


def finite_number(value: object, label: str) -> float:
    """Require a real finite number, without accepting booleans or numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(f"{label} must be a finite number")
    if not math.isfinite(value):
        raise InvalidInputError(f"{label} must be a finite number")
    return float(value)


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
