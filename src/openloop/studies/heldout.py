"""Pure S1 part 2 follow-ups: held-out evaluation transform, parsing and analysis.

No file, network, GPU or training access happens here. Boundary: a candidate
program computes both scores, but the held-out score is read once, from a
shard that no run trained on or selected on. Orientation matches
`openloop.studies.claims_audit`: gain is parent BPB minus child BPB.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import StatisticsError, correlation, fmean, median, stdev

from openloop.decider import sign_p_value
from openloop.ledger import InvalidInputError
from openloop.ledger.validation import check_text, finite_number
from openloop.studies.claims_audit import (
    MIN_PAIRS_FOR_INFERENCE,
    Observation,
    mean_interval,
    paired_t_p,
)

# The published final validation evaluation. The held-out evaluation is
# appended directly after it, so it runs after training and after the
# published score exists. The anchor is identical in both audited programs.
EVAL_ANCHOR = (
    'val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, backend="mlx")\n'
)
HELDOUT_DIR_ENV = "OPENLOOP_HELDOUT_DIR"
HELDOUT_SHARD = "shard_05000.parquet"
HELDOUT_KEY = "heldout_bpb"

# Evaluation only. `evaluate_bpb` reads `DATA_DIR` and `VAL_FILENAME` from the
# `prepare` module at call time, so pointing those two names at the held-out
# shard reuses the arm's own loader, tokenizer, batch size and token count.
# No training state is touched: training has ended and the model is not
# updated. A missing environment variable raises KeyError and the run fails.
HELDOUT_EVAL = (
    "import prepare as _heldout_prepare\n"
    f'_heldout_prepare.DATA_DIR = os.environ["{HELDOUT_DIR_ENV}"]\n'
    f'_heldout_prepare.VAL_FILENAME = "{HELDOUT_SHARD}"\n'
    f"{HELDOUT_KEY} = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, "
    'backend="mlx")\n'
    f'print(f"{HELDOUT_KEY}:      {{{HELDOUT_KEY}:.6f}}")\n'
)

CONFIDENCE = 0.95
_HELDOUT_LINE = re.compile(rf"^{HELDOUT_KEY}: +([0-9]+(?:\.[0-9]+)?)$", re.MULTILINE)


class HeldoutFinding(StrEnum):
    """Pre-registered outcome of the single held-out test."""

    REPLICATED = "replicated"
    REVERSED = "reversed"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_DATA = "insufficient_data"


def inject_heldout_eval(source: str) -> str:
    """Append the held-out evaluation after the published final evaluation.

    Fails unless the anchor occurs exactly once, so a changed program cannot be
    evaluated silently with the transformation misplaced or duplicated.
    """
    if source.count(EVAL_ANCHOR) != 1 or HELDOUT_KEY in source:
        raise InvalidInputError(
            "Expected the final evaluation line exactly once and no held-out code"
        )
    return source.replace(EVAL_ANCHOR, EVAL_ANCHOR + HELDOUT_EVAL)


def parse_heldout_bpb(log: str) -> float:
    """Return the single held-out score. Reject missing, repeated or nonpositive."""
    found = _HELDOUT_LINE.findall(log)
    if len(found) != 1:
        raise InvalidInputError("Expected exactly one held-out BPB line")
    value = float(found[0])
    if not math.isfinite(value) or value <= 0:
        raise InvalidInputError("Held-out BPB must be finite and positive")
    return value


@dataclass(frozen=True)
class HeldoutObservation:
    """One successful run's published validation score and held-out score."""

    arm: str
    seed: int
    val_bpb: float
    heldout_bpb: float
    num_steps: int

    def __post_init__(self) -> None:
        check_text(self.arm)
        if type(self.seed) is not int or self.seed < 0:
            raise InvalidInputError("Seed must be a nonnegative integer")
        for name, value in (("val_bpb", self.val_bpb), (HELDOUT_KEY, self.heldout_bpb)):
            if finite_number(value, name) <= 0:
                raise InvalidInputError(f"{name} must be positive")
        if type(self.num_steps) is not int or self.num_steps <= 0:
            raise InvalidInputError("Step count must be a positive integer")


@dataclass(frozen=True)
class GainStats:
    """Paired gain statistics for one score, unadjusted (one test, alpha 0.05)."""

    seeds: tuple[int, ...]
    gains: tuple[float, ...]
    wins: int
    mean_gain: float | None
    median_gain: float | None
    sd_gain: float | None
    interval: tuple[float, float] | None
    sign_p: float | None
    t_p: float | None


@dataclass(frozen=True)
class Agreement:
    """How closely the validation and held-out views of the same pairs agree."""

    seeds: tuple[int, ...]
    same_sign_seeds: int
    gain_correlation: float | None
    mean_shift_child: float | None
    mean_shift_parent: float | None


@dataclass(frozen=True)
class HeldoutReport:
    child: str
    parent: str
    confidence: float
    heldout: GainStats
    validation: GainStats
    agreement: Agreement
    finding: HeldoutFinding
    unmatched: tuple[int, ...]


def _stats(seeds: tuple[int, ...], gains: tuple[float, ...]) -> GainStats:
    wins = sum(gain > 0 for gain in gains)
    if len(gains) < MIN_PAIRS_FOR_INFERENCE:
        return GainStats(
            seeds,
            gains,
            wins,
            fmean(gains) if gains else None,
            median(gains) if gains else None,
            None,
            None,
            None,
            None,
        )
    return GainStats(
        seeds,
        gains,
        wins,
        fmean(gains),
        median(gains),
        stdev(gains),
        mean_interval(gains, CONFIDENCE),
        sign_p_value(gains),
        paired_t_p(gains),
    )


def _classify(stats: GainStats) -> HeldoutFinding:
    if stats.interval is None:
        return HeldoutFinding.INSUFFICIENT_DATA
    low, high = stats.interval
    if low > 0:
        return HeldoutFinding.REPLICATED
    return HeldoutFinding.REVERSED if high < 0 else HeldoutFinding.INCONCLUSIVE


def analyze_heldout(
    observations: Sequence[HeldoutObservation], *, child: str, parent: str
) -> HeldoutReport:
    """Paired analysis of one pre-registered claim on the held-out score.

    This is a single pre-specified test, so no multiplicity adjustment applies:
    the finding uses the unadjusted 95% t interval of the held-out gain.
    Validation scores in this phase are additional samples on the selection
    shard, reported for comparison. They never decide the finding.
    """
    check_text(child)
    check_text(parent)
    if child == parent:
        raise InvalidInputError("A comparison needs two different arms")
    runs: dict[str, dict[int, HeldoutObservation]] = {child: {}, parent: {}}
    for item in observations:
        if item.arm not in runs:
            raise InvalidInputError(f"Unexpected arm: {item.arm}")
        if item.seed in runs[item.arm]:
            raise InvalidInputError(f"Duplicate run for {item.arm} seed {item.seed}")
        runs[item.arm][item.seed] = item
    kid, par = runs[child], runs[parent]
    seeds = tuple(sorted(set(kid) & set(par)))
    held = tuple(par[s].heldout_bpb - kid[s].heldout_bpb for s in seeds)
    valid = tuple(par[s].val_bpb - kid[s].val_bpb for s in seeds)
    try:
        corr = (
            correlation(held, valid) if len(seeds) >= MIN_PAIRS_FOR_INFERENCE else None
        )
    except StatisticsError:
        corr = None  # A constant series has no correlation.
    held_stats = _stats(seeds, held)
    agreement = Agreement(
        seeds,
        sum((h > 0) == (v > 0) for h, v in zip(held, valid, strict=True)),
        corr,
        fmean(kid[s].heldout_bpb - kid[s].val_bpb for s in seeds) if seeds else None,
        fmean(par[s].heldout_bpb - par[s].val_bpb for s in seeds) if seeds else None,
    )
    return HeldoutReport(
        child,
        parent,
        CONFIDENCE,
        held_stats,
        _stats(seeds, valid),
        agreement,
        _classify(held_stats),
        tuple(sorted(set(kid) ^ set(par))),
    )


def _integer(value: object) -> int:
    if type(value) is not int:
        raise InvalidInputError("Receipt field must be an integer")
    return value


def heldout_run_observations(
    records: Sequence[Mapping[str, object]],
) -> list[HeldoutObservation]:
    """Build observations from run receipts of kind `heldout` that succeeded."""
    result: list[HeldoutObservation] = []
    for record in records:
        summary = record.get("summary")
        if record.get("kind") != "heldout" or not isinstance(summary, Mapping):
            continue
        result.append(
            HeldoutObservation(
                str(record["arm"]),
                _integer(record["seed"]),
                float(summary["val_bpb"]),
                float(summary[HELDOUT_KEY]),
                _integer(summary["num_steps"]),
            )
        )
    return result


def combined_observations(
    original: Sequence[Mapping[str, object]],
    completion: Sequence[Mapping[str, object]],
) -> list[Observation]:
    """Original October 13 receipts plus seed-completion receipts, as observations.

    Failed receipts (no summary) are skipped. A completion receipt must be of
    kind `seed_completion`, so held-out runs can never enter the audit report.
    """
    for record in completion:
        if record.get("kind") != "seed_completion":
            raise InvalidInputError("Only seed-completion receipts may be combined")
    result: list[Observation] = []
    for record in [*original, *completion]:
        summary = record.get("summary")
        if not isinstance(summary, Mapping):
            continue
        result.append(
            Observation(
                str(record["arm"]),
                _integer(record["seed"]),
                float(summary["val_bpb"]),
                _integer(summary["num_steps"]),
            )
        )
    return result
