"""Pure token-mixture allocation. No files, training, clocks, or ledger access."""

from collections.abc import Iterator, Mapping
from functools import reduce
from math import gcd
from types import MappingProxyType

from openloop.ledger import JSONValue, content_hash

from .models import ContractError


def normalize_weights(value: JSONValue, names: tuple[str, ...]) -> Mapping[str, int]:
    """Require all source names and nonnegative integer ratios; remove common scale."""
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ContractError("Mixture weights must name every frozen training source")
    weights: dict[str, int] = {}
    for name in sorted(names):
        weight = value[name]
        if type(weight) is not int or weight < 0:
            raise ContractError("Mixture weights must be nonnegative integers")
        weights[name] = weight
    divisor = reduce(gcd, weights.values(), 0)
    if not divisor:
        raise ContractError("At least one mixture weight must be positive")
    return MappingProxyType(
        {name: weight // divisor for name, weight in weights.items()}
    )


def batch_counts(
    weights: Mapping[str, JSONValue], *, batches: int, seed: int
) -> Mapping[str, int]:
    """Allocate a finite batch budget by largest remainder, with seeded hash ties."""
    normalized = normalize_weights(weights, tuple(weights))
    if type(batches) is not int or batches < 1 or type(seed) is not int or seed < 0:
        raise ContractError(
            "Mixture allocation needs positive batches and a valid seed"
        )
    integers = normalized
    total = sum(integers.values())
    counts = {name: batches * weight // total for name, weight in integers.items()}
    order = sorted(
        integers,
        key=lambda name: (
            -(batches * integers[name] % total),
            content_hash({"seed": seed, "source": name}),
        ),
    )
    for name in order[: batches - sum(counts.values())]:
        counts[name] += 1
    return MappingProxyType(counts)


def batch_schedule(counts: Mapping[str, int], *, seed: int) -> Iterator[str]:
    """Interleave exact quotas by smooth weighted round robin, without randomness IO."""
    if not counts or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ContractError("Mixture quotas must be nonnegative integers")
    if type(seed) is not int or seed < 0:
        raise ContractError("Mixture schedule needs a nonnegative integer seed")
    total = sum(counts.values())
    if not total:
        raise ContractError("Mixture quotas cannot all be zero")
    order = sorted(
        counts, key=lambda name: content_hash({"seed": seed, "source": name})
    )
    scores = dict.fromkeys(order, 0)
    remaining = dict(counts)
    for _ in range(total):
        for name in order:
            scores[name] += counts[name]
        selected = max(
            (name for name in order if remaining[name]), key=scores.__getitem__
        )
        scores[selected] -= total
        remaining[selected] -= 1
        yield selected
