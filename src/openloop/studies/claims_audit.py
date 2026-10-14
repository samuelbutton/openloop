"""Pure S1 part 2 public-claims audit: source transforms, run order and analysis.

No file, network, GPU or training access happens here. The audit replicates
published claims on this machine; it judges measurements, not the authors.
Orientation: a gain is parent BPB minus child BPB, so positive means improved.
"""

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean, median, stdev

from openloop.decider import Verdict, autoscientists_gate, sign_p_value
from openloop.ledger import InvalidInputError
from openloop.ledger.validation import check_text, finite_number

SEED_CALL = "mx.random.seed(42)"
MIN_PAIRS_FOR_INFERENCE = 3

# Memory guard policy. These constants enter the frozen plan, so they are
# pre-registered. macOS reports pressure level 1 normal, 2 warning, 4 critical.
# Warning alone never trips: ordinary training runs at warning.
PRESSURE_NORMAL = 1
PRESSURE_WARNING = 2
PRESSURE_CRITICAL = 4
MEMORY_SAMPLE_SECONDS = 5.0
FREE_PERCENT_FLOOR = 15
FREE_PERCENT_CONSECUTIVE_SAMPLES = 2


class ClaimedOutcome(StrEnum):
    """What the publication said about a child relative to its parent."""

    WIN = "win"
    NOT_WIN = "not_win"


class Finding(StrEnum):
    """Replication finding. Wins and not-wins use different subsets."""

    REPLICATED = "replicated"
    NOT_REPLICATED = "not_replicated"
    REVERSED = "reversed"
    INDISTINGUISHABLE = "indistinguishable"
    ACTUALLY_WORSE = "actually_worse"
    ACTUALLY_BETTER = "actually_better"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class MemoryReading:
    """One system-wide memory observation, timestamped by the sampler."""

    observed_at: str
    pressure_level: int
    free_percent: int

    def __post_init__(self) -> None:
        check_text(self.observed_at)
        if self.pressure_level not in (
            PRESSURE_NORMAL,
            PRESSURE_WARNING,
            PRESSURE_CRITICAL,
        ):
            raise InvalidInputError("Unknown memory pressure level")
        if type(self.free_percent) is not int or not 0 <= self.free_percent <= 100:
            raise InvalidInputError("Free memory percentage must be 0 to 100")


@dataclass(frozen=True)
class GuardTrip:
    """Why the guard stopped the study, with the readings that caused it."""

    reason: str
    readings: tuple[MemoryReading, ...]


def memory_guard_policy() -> dict[str, int | float]:
    """The thresholds as recorded in the plan."""
    return {
        "sample_seconds": MEMORY_SAMPLE_SECONDS,
        "critical_pressure_level": PRESSURE_CRITICAL,
        "free_percent_floor": FREE_PERCENT_FLOOR,
        "free_percent_consecutive_samples": FREE_PERCENT_CONSECUTIVE_SAMPLES,
    }


class MemoryGuard:
    """Trip on critical pressure, or free memory below the floor repeatedly.

    Stateful across one job's samples. A warning level, or a single low free
    reading, never trips.
    """

    def __init__(self) -> None:
        self._low_run = 0
        self._recent: list[MemoryReading] = []

    def observe(self, reading: MemoryReading) -> GuardTrip | None:
        self._recent = [*self._recent, reading][-FREE_PERCENT_CONSECUTIVE_SAMPLES - 1 :]
        if reading.free_percent < FREE_PERCENT_FLOOR:
            self._low_run += 1
        else:
            self._low_run = 0
        if reading.pressure_level >= PRESSURE_CRITICAL:
            return GuardTrip("critical memory pressure", tuple(self._recent))
        if self._low_run >= FREE_PERCENT_CONSECUTIVE_SAMPLES:
            return GuardTrip(
                f"system free memory below {FREE_PERCENT_FLOOR}% for "
                f"{FREE_PERCENT_CONSECUTIVE_SAMPLES} consecutive samples",
                tuple(self._recent),
            )
        return None


@dataclass(frozen=True)
class Comparison:
    """One pre-registered child-versus-parent claim with published scores."""

    name: str
    child: str
    parent: str
    claimed: ClaimedOutcome
    published_child_bpb: float
    published_parent_bpb: float

    def __post_init__(self) -> None:
        check_text(self.name)
        check_text(self.child)
        check_text(self.parent)
        if self.child == self.parent:
            raise InvalidInputError("A comparison needs two different arms")
        object.__setattr__(self, "claimed", ClaimedOutcome(self.claimed))
        finite_number(self.published_child_bpb, "Published child BPB")
        finite_number(self.published_parent_bpb, "Published parent BPB")

    @property
    def published_gain(self) -> float:
        return self.published_parent_bpb - self.published_child_bpb


@dataclass(frozen=True)
class Observation:
    """One successful training run's final summary."""

    arm: str
    seed: int
    val_bpb: float
    num_steps: int

    def __post_init__(self) -> None:
        check_text(self.arm)
        if type(self.seed) is not int or self.seed < 0:
            raise InvalidInputError("Seed must be a nonnegative integer")
        if finite_number(self.val_bpb, "val_bpb") <= 0:
            raise InvalidInputError("val_bpb must be positive")
        if type(self.num_steps) is not int or self.num_steps <= 0:
            raise InvalidInputError("Step count must be a positive integer")


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    seeds: tuple[int, ...]
    mean_bpb: float | None
    sd_bpb: float | None
    mean_steps: float | None


@dataclass(frozen=True)
class ComparisonResult:
    comparison: Comparison
    seeds: tuple[int, ...]
    gains: tuple[float, ...]
    mean_gain: float | None
    median_gain: float | None
    sd_gain: float | None
    sign_p: float | None
    sign_p_bonferroni: float | None
    t_p: float | None
    t_p_bonferroni: float | None
    interval: tuple[float, float] | None
    adjusted_interval: tuple[float, float] | None
    parent_sd_bpb: float | None
    published_gain_gate: Verdict | None
    finding: Finding


@dataclass(frozen=True)
class AuditReport:
    comparison_count: int
    alpha: float
    arms: tuple[ArmSummary, ...]
    comparisons: tuple[ComparisonResult, ...]
    unmatched: Mapping[str, tuple[int, ...]]


def inject_seed(source: str, seed: int) -> str:
    """Replace the single upstream seed call. Fail unless exactly one match."""
    if type(seed) is not int or seed < 0:
        raise InvalidInputError("Seed must be a nonnegative integer")
    if source.count(SEED_CALL) != 1 or len(re.findall(r"\.seed\(", source)) != 1:
        raise InvalidInputError(
            f"Expected exactly one seed call, {SEED_CALL}, in the source"
        )
    return source.replace(SEED_CALL, f"mx.random.seed({seed})")


def substitute_once(source: str, old: str, new: str) -> str:
    """Replace one exact text occurrence. Fail unless it occurs exactly once."""
    if not old or old == new or source.count(old) != 1:
        raise InvalidInputError(f"Expected exactly one occurrence of {old!r}")
    return source.replace(old, new)


def interleaved_order(
    arms: Sequence[str], seeds: Sequence[int], shuffle_seed: int
) -> tuple[tuple[str, int], ...]:
    """Shuffle every arm-seed pair with a fixed seed so arms interleave."""
    if len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds):
        raise InvalidInputError("Arms and seeds must be unique")
    pairs = [(arm, seed) for arm in sorted(arms) for seed in sorted(seeds)]
    random.Random(shuffle_seed).shuffle(pairs)
    return tuple(pairs)


def parse_summary(log: str) -> dict[str, float | int | str]:
    """Parse only the final summary block. Reject missing or nonfinite values."""
    parts = log.rsplit("\n---\n", 1)
    if len(parts) != 2:
        raise InvalidInputError("Training omitted its final summary")
    integers = {"num_steps", "depth"}
    text = {"backend", "chip"}
    summary: dict[str, float | int | str] = {}
    for line in parts[1].splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.strip()
        if key in integers:
            if not value.isdecimal() or int(value) <= 0:
                raise InvalidInputError(f"Invalid integer summary: {key}")
            summary[key] = int(value)
        elif key in text:
            summary[key] = value
        else:
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                raise InvalidInputError(f"Invalid numeric summary: {key}")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise InvalidInputError(f"Nonfinite summary: {key}")
            summary[key] = numeric
    if "val_bpb" not in summary or "num_steps" not in summary:
        raise InvalidInputError("Training omitted BPB or step count")
    return summary


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def t_upper_tail(t: float, df: int) -> float:
    """P(T > |t|) for Student's t with df degrees of freedom."""
    if df < 1:
        raise InvalidInputError("Degrees of freedom must be positive")
    t = abs(finite_number(t, "t statistic"))
    return 0.5 * _incomplete_beta(df / 2, 0.5, df / (df + t * t))


def t_quantile(upper_tail: float, df: int) -> float:
    """Positive t with P(T > t) equal to upper_tail, by bisection."""
    if not 0 < upper_tail < 0.5:
        raise InvalidInputError("Upper tail must be between 0 and 0.5")
    low, high = 0.0, 1.0
    while t_upper_tail(high, df) > upper_tail:
        high *= 2
        if high > 1e12:
            raise InvalidInputError("t quantile out of range")
    for _ in range(200):
        mid = (low + high) / 2
        if t_upper_tail(mid, df) > upper_tail:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def mean_interval(gains: Sequence[float], confidence: float) -> tuple[float, float]:
    """Two-sided t interval for the mean of at least two paired gains."""
    if len(gains) < 2 or not 0 < confidence < 1:
        raise InvalidInputError("An interval needs two gains and 0 < confidence < 1")
    mean = fmean(gains)
    half = t_quantile((1 - confidence) / 2, len(gains) - 1) * (
        stdev(gains) / math.sqrt(len(gains))
    )
    return mean - half, mean + half


def paired_t_p(gains: Sequence[float]) -> float:
    """Two-sided paired t-test p-value for mean gain zero."""
    if len(gains) < 2:
        raise InvalidInputError("A t-test needs two gains")
    mean, sd = fmean(gains), stdev(gains)
    if sd == 0:
        return 1.0 if mean == 0 else 0.0
    return min(
        1.0, 2 * t_upper_tail(mean / (sd / math.sqrt(len(gains))), len(gains) - 1)
    )


def _classify(
    claimed: ClaimedOutcome,
    interval: tuple[float, float],
    sign_p: float,
    alpha: float,
) -> Finding:
    low, high = interval
    if claimed is ClaimedOutcome.WIN:
        if low > 0 and sign_p <= alpha:
            return Finding.REPLICATED
        return Finding.REVERSED if high < 0 else Finding.NOT_REPLICATED
    if high < 0:
        return Finding.ACTUALLY_WORSE
    return Finding.ACTUALLY_BETTER if low > 0 else Finding.INDISTINGUISHABLE


def _arm_summary(arm: str, runs: Mapping[int, Observation]) -> ArmSummary:
    scores = [run.val_bpb for run in runs.values()]
    steps = [run.num_steps for run in runs.values()]
    return ArmSummary(
        arm,
        tuple(sorted(runs)),
        fmean(scores) if scores else None,
        stdev(scores) if len(scores) > 1 else None,
        fmean(steps) if steps else None,
    )


def analyze_audit(
    comparisons: Sequence[Comparison],
    observations: Sequence[Observation],
    *,
    alpha: float = 0.05,
) -> AuditReport:
    """Paired per-seed analysis. Only seeds present in both arms are paired.

    Intervals use the Bonferroni level 1 - alpha / m across all m registered
    comparisons for classification. The exact sign test needs every pair to
    improve; with five pairs its smallest possible p-value is 1/32, so its
    Bonferroni-adjusted value can never fall below alpha at m = 5. It is
    reported, and a win additionally needs all pairs positive via its raw p.
    """
    alpha = finite_number(alpha, "Alpha")
    if not 0 < alpha < 1 or not comparisons:
        raise InvalidInputError("Require 0 < alpha < 1 and some comparisons")
    runs: dict[str, dict[int, Observation]] = {}
    for item in observations:
        arm_runs = runs.setdefault(item.arm, {})
        if item.seed in arm_runs:
            raise InvalidInputError(f"Duplicate run for {item.arm} seed {item.seed}")
        arm_runs[item.seed] = item
    count = len(comparisons)
    results: list[ComparisonResult] = []
    unmatched: dict[str, tuple[int, ...]] = {}
    for comparison in comparisons:
        child = runs.get(comparison.child, {})
        parent = runs.get(comparison.parent, {})
        seeds = tuple(sorted(set(child) & set(parent)))
        extra = tuple(sorted(set(child) ^ set(parent)))
        if extra:
            unmatched[comparison.name] = extra
        gains = tuple(parent[s].val_bpb - child[s].val_bpb for s in seeds)
        parent_sd = (
            stdev([run.val_bpb for run in parent.values()]) if len(parent) > 1 else None
        )
        gate = (
            autoscientists_gate(comparison.published_gain, parent_sd)
            if parent_sd is not None
            else None
        )
        if len(gains) < MIN_PAIRS_FOR_INFERENCE:
            results.append(
                ComparisonResult(
                    comparison,
                    seeds,
                    gains,
                    fmean(gains) if gains else None,
                    median(gains) if gains else None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    parent_sd,
                    gate,
                    Finding.INSUFFICIENT_DATA,
                )
            )
            continue
        sign_p = sign_p_value(gains)
        adjusted = mean_interval(gains, 1 - alpha / count)
        t_p = paired_t_p(gains)
        results.append(
            ComparisonResult(
                comparison,
                seeds,
                gains,
                fmean(gains),
                median(gains),
                stdev(gains),
                sign_p,
                min(1.0, sign_p * count),
                t_p,
                min(1.0, t_p * count),
                mean_interval(gains, 1 - alpha),
                adjusted,
                parent_sd,
                gate,
                _classify(comparison.claimed, adjusted, sign_p, alpha),
            )
        )
    arms = tuple(_arm_summary(arm, arm_runs) for arm, arm_runs in sorted(runs.items()))
    return AuditReport(count, alpha, arms, tuple(results), unmatched)
