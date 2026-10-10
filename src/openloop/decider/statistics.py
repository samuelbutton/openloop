"""Pure noise and exact paired statistics; no training or data access.

Inference requires independent, predeclared seed draws. The sign test concerns
median paired gain, not a mean. Ties count as non-wins for a conservative test.
"""

import math
from dataclasses import dataclass
from statistics import median

from openloop.ledger import InvalidInputError
from openloop.ledger.validation import coerce_enum, finite_number

from .models import NoiseKind, Sample, Verdict, unique_samples


@dataclass(frozen=True)
class Calibration:
    """Three to five same-code pairs; five pairs lock the noise estimate.

    Distinct seeds measure seed variation. Fixed-seed repeats measure only
    execution variation. They must never substitute for distinct-seed evidence.
    """

    pairs: tuple[tuple[Sample, Sample], ...]
    kind: NoiseKind = NoiseKind.DISTINCT_SEED

    def __post_init__(self) -> None:
        pairs = tuple(tuple(pair) for pair in self.pairs)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(
            self, "kind", coerce_enum(NoiseKind, self.kind, "noise kind")
        )
        if not 3 <= len(pairs) <= 5 or any(len(pair) != 2 for pair in pairs):
            raise InvalidInputError("Noise estimation requires three to five pairs")
        samples = self.samples
        if any(not isinstance(sample, Sample) for sample in samples):
            raise InvalidInputError("Noise calibration requires typed samples")
        if self.kind is NoiseKind.DISTINCT_SEED:
            unique_samples(samples)
            if len({sample.inputs.seed for sample in samples}) != len(samples):
                raise InvalidInputError("Noise pairs require distinct seeds")
        elif len({sample.inputs.hash for sample in samples}) != 1:
            raise InvalidInputError(
                "Fixed-seed calibration must repeat identical inputs"
            )
        if len({sample.attempt_id for sample in samples}) != len(samples) or len(
            {sample.reference for sample in samples}
        ) != len(samples):
            raise InvalidInputError("Noise pairs require distinct executions")
        if (
            len(
                {
                    (sample.context_hash, sample.inputs.candidate_hash)
                    for sample in samples
                }
            )
            != 1
        ):
            raise InvalidInputError(
                "Noise calibration requires one candidate and context"
            )
        _ = self.sigma  # Reject overflow at the boundary.

    @property
    def samples(self) -> tuple[Sample, ...]:
        return tuple(sample for pair in self.pairs for sample in pair)

    @property
    def sigma(self) -> float:
        """AutoScientists' per-run sqrt(sum(pair difference squared) / (2n))."""
        differences = [
            finite_number(left.metric.value - right.metric.value, "Noise difference")
            for left, right in self.pairs
        ]
        # hypot avoids overflow from squaring a large finite score difference.
        return finite_number(
            math.hypot(*differences) / math.sqrt(2 * len(self.pairs)), "Noise sigma"
        )

    @property
    def locked(self) -> bool:
        return len(self.pairs) == 5


def autoscientists_gate(
    gain: float, sigma: float, *, second_gain: float | None = None
) -> Verdict:
    """Published A.6 gate; both gains use the SAME frozen champion score.

    Large gains pass directly. A small positive gain needs a new candidate
    seed that also beats the anchor. This gate alone has no multiplicity bound.
    """
    gain = finite_number(gain, "Gain")
    sigma = finite_number(sigma, "Noise sigma")
    if sigma < 0:
        raise InvalidInputError("Noise sigma must be nonnegative")
    if second_gain is not None:
        second_gain = finite_number(second_gain, "Second gain")
    if gain <= 0:
        return Verdict.DISCARD
    if gain > 2 * sigma:
        return Verdict.PROMOTE
    if second_gain is None:
        return Verdict.INCONCLUSIVE
    return Verdict.PROMOTE if second_gain > 0 else Verdict.DISCARD


def sign_p_value(gains: tuple[float, ...], *, minimum_effect: float = 0.0) -> float:
    """Exact upper binomial tail under P(gain > minimum_effect) <= 1/2."""
    minimum_effect = finite_number(minimum_effect, "Minimum effect")
    values = tuple(finite_number(gain, "Gain") for gain in gains)
    n = len(values)
    wins = sum(gain > minimum_effect for gain in values)
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def median_interval(
    gains: tuple[float, ...], *, alpha: float = 0.05
) -> tuple[float | None, float | None]:
    """Exact conservative two-sided order-statistic interval for median gain.

    None is an unbounded endpoint when evidence cannot give finite bounds.
    """
    alpha = finite_number(alpha, "Alpha")
    if not 0 < alpha < 1:
        raise InvalidInputError("Require 0 < alpha < 1")
    values = sorted(finite_number(gain, "Gain") for gain in gains)
    n = len(values)
    k = 0
    for rank in range(1, n // 2 + 1):
        tail = sum(math.comb(n, j) for j in range(rank)) / 2**n
        if 2 * tail <= alpha:
            k = rank
        else:
            break
    return (values[k - 1], values[n - k]) if k else (None, None)


def median_gain(gains: tuple[float, ...]) -> float | None:
    """Return the tested effect estimate, absent if there are no seed pairs."""
    return float(median(gains)) if gains else None
