"""Pure S1 calibration plan and analysis over trusted T1 ledger scores.

No data access or training occurs here. Fixed-seed execution variation and
distinct-seed variation are separate. Candidate ranks use matched seed means,
not the ordering of noisy baseline repeats. No held-out scores are permitted.
"""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from statistics import fmean, stdev

from openloop.ledger import (
    Direction,
    ExperimentInputs,
    InvalidInputError,
    JSONValue,
    Purpose,
    Run,
    Status,
    content_hash,
)
from openloop.ledger.validation import (
    check_text,
    coerce_enum,
    finite_number,
    freeze_object,
)
from openloop.loops.models import LoopAction, LoopSpec, Phase


class Cohort(StrEnum):
    FIXED_SEED = "fixed_seed"
    DISTINCT_SEED = "distinct_seed"
    RANK = "rank"


@dataclass(frozen=True)
class Trial:
    slot: str
    candidate: str
    cohort: Cohort
    inputs: ExperimentInputs

    def __post_init__(self) -> None:
        check_text(self.slot)
        check_text(self.candidate)
        object.__setattr__(self, "cohort", coerce_enum(Cohort, self.cohort, "cohort"))
        if not isinstance(self.inputs, ExperimentInputs):
            raise InvalidInputError("Trials require typed inputs")


@dataclass(frozen=True)
class StudyPlan:
    """Frozen jobs, execution order, and three seeds for candidate ranks."""

    trials: tuple[Trial, ...]
    rank_seeds: tuple[int, ...]
    wall_limit_seconds: int = 43_200

    def __post_init__(self) -> None:
        trials = tuple(self.trials)
        seeds = tuple(self.rank_seeds)
        if not trials or any(not isinstance(trial, Trial) for trial in trials):
            raise InvalidInputError("Declare typed study trials")
        if len({trial.slot for trial in trials}) != len(trials):
            raise InvalidInputError("Study slots must be unique")
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise InvalidInputError("Rank seeds must be unique nonnegative integers")
        if type(self.wall_limit_seconds) is not int or self.wall_limit_seconds < 1:
            raise InvalidInputError("Wall limit must be a positive integer")
        conditions = {
            content_hash(
                replace(
                    trial.inputs,
                    config={},
                    seed=0,
                    phase=Phase.SCREEN,
                    fidelity="study",
                    budget_amount=1,
                )
            )
            for trial in trials
        }
        if len(conditions) != 1:
            raise InvalidInputError(
                "Study trials must share frozen source, data and environment"
            )
        candidates = {trial.candidate for trial in trials}
        identities = [
            {
                trial.inputs.candidate_hash
                for trial in trials
                if trial.candidate == candidate
            }
            for candidate in candidates
        ]
        if any(len(group) != 1 for group in identities) or len(
            set.union(*identities)
        ) != len(candidates):
            raise InvalidInputError("Each candidate label requires one unique identity")
        levels = {trial.inputs.fidelity for trial in trials}
        if len(levels) != 2:
            raise InvalidInputError("Study trials require two fidelities")
        for level in levels:
            group = [trial for trial in trials if trial.inputs.fidelity == level]
            if (
                len(
                    {
                        (trial.inputs.budget_amount, trial.inputs.phase)
                        for trial in group
                    }
                )
                != 1
            ):
                raise InvalidInputError("Each fidelity requires one budget and phase")
            for cohort in (Cohort.FIXED_SEED, Cohort.DISTINCT_SEED):
                baseline = [
                    trial
                    for trial in group
                    if trial.candidate == "baseline" and trial.cohort is cohort
                ]
                expected = 1 if cohort is Cohort.FIXED_SEED else 10
                if (
                    len(baseline) != 10
                    or len({trial.inputs.seed for trial in baseline}) != expected
                ):
                    raise InvalidInputError(
                        "Each fidelity needs ten fixed and ten distinct baseline runs"
                    )
            fixed_seeds = {
                trial.inputs.seed
                for trial in group
                if trial.cohort is Cohort.FIXED_SEED
            }
            distinct_seeds = {
                trial.inputs.seed
                for trial in group
                if trial.candidate == "baseline"
                and trial.cohort is Cohort.DISTINCT_SEED
            }
            if fixed_seeds & distinct_seeds or not set(seeds) <= distinct_seeds:
                raise InvalidInputError(
                    "Rank seeds need distinct baseline runs, separate from fixed seeds"
                )
            if any(
                trial.candidate == "baseline" and trial.cohort is Cohort.RANK
                for trial in group
            ):
                raise InvalidInputError(
                    "Baseline rank scores must reuse the distinct-seed runs"
                )
            for candidate in candidates - {"baseline"}:
                panel = [trial for trial in group if trial.candidate == candidate]
                if (
                    len(panel) != len(seeds)
                    or {trial.inputs.seed for trial in panel} != set(seeds)
                    or any(trial.cohort is not Cohort.RANK for trial in panel)
                ):
                    raise InvalidInputError(
                        "Each candidate needs exactly the matched rank seeds"
                    )
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "rank_seeds", seeds)

    @property
    def hash(self) -> str:
        return content_hash({"s1_version": 1, "plan": self})

    @property
    def training_tokens(self) -> int:
        return sum(trial.inputs.budget_amount for trial in self.trials)


def build_plan(
    spec: LoopSpec,
    configs: Mapping[str, Mapping[str, JSONValue]],
    environment_hash: str,
) -> StudyPlan:
    """Declare 10 fixed and 10 distinct baseline seeds per fidelity.

    The baseline's first three distinct seeds also enter the rank panel.
    Each other candidate gets those same three seeds at both fidelities.
    """
    if "baseline" not in configs or len(configs) < 3 or len(spec.fidelities) != 2:
        raise InvalidInputError(
            "S1 needs a baseline, a candidate panel and two fidelities"
        )
    if len({spec.candidate_hash(config) for config in configs.values()}) != len(
        configs
    ):
        raise InvalidInputError("Rank candidates must have distinct identities")
    rank_seeds = (1000, 1001, 1002)
    trials: list[Trial] = []
    for candidate, config in configs.items():
        cohorts = (
            (
                (Cohort.FIXED_SEED, (42,) * 10),
                (Cohort.DISTINCT_SEED, tuple(range(1000, 1010))),
            )
            if candidate == "baseline"
            else ((Cohort.RANK, rank_seeds),)
        )
        for cohort, seeds in cohorts:
            for i, seed in enumerate(seeds):
                for level, phase in zip(spec.fidelities, Phase, strict=True):
                    inputs = spec.inputs(
                        config, LoopAction(seed, level.name, phase), environment_hash
                    )
                    trials.append(
                        Trial(
                            f"{candidate}:{cohort.value}:{i}:{level.name}",
                            candidate,
                            cohort,
                            inputs,
                        )
                    )
    # A paired baseline pilot runs first. Shuffle the rest to spread time drift.
    pilot = [
        trial
        for trial in trials
        if trial.candidate == "baseline"
        and trial.cohort is Cohort.DISTINCT_SEED
        and trial.inputs.seed == rank_seeds[0]
    ]
    remaining = [trial for trial in trials if trial not in pilot]
    random.Random(20261010).shuffle(remaining)
    return StudyPlan(tuple(pilot + remaining), rank_seeds)


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for i in ordered[start:end]:
            ranks[i] = (start + 1 + end) / 2
        start = end
    return tuple(ranks)


def spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    """Average tied ranks. Constant rankings return None, not a false zero."""
    if len(left) != len(right) or len(left) < 2:
        raise InvalidInputError("Correlation needs matched observations")
    ranks = [
        _ranks(tuple(finite_number(value, "Rank value") for value in values))
        for values in (left, right)
    ]
    centered = [tuple(value - fmean(values) for value in values) for values in ranks]
    denominator = math.sqrt(
        sum(value**2 for value in centered[0]) * sum(value**2 for value in centered[1])
    )
    if denominator == 0:
        return None
    return max(
        -1.0, min(1.0, sum(a * b for a, b in zip(*centered, strict=True)) / denominator)
    )


def _interval(values: list[float]) -> tuple[float, float]:
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[
        math.ceil(0.975 * (len(values) - 1))
    ]


def _noise(values: tuple[float, ...]) -> dict[str, JSONValue]:
    if len(values) < 2:
        raise InvalidInputError("Noise estimation needs at least two scores")
    rng = random.Random(20261010)
    bootstrap = [stdev(rng.choices(values, k=len(values))) for _ in range(2000)]
    return {
        "sample_count": len(values),
        "mean_val_bpb": fmean(values),
        "sample_std_bpb": stdev(values),
        "std_bootstrap_95_percentile_interval": list(_interval(bootstrap)),
        "scores": values,
    }


def analyze(plan: StudyPlan, runs: Mapping[str, Run]) -> Mapping[str, JSONValue]:
    """Require every planned success; never treat failures or aliases as samples."""
    if set(runs) != {trial.slot for trial in plan.trials}:
        raise InvalidInputError("Analysis requires the complete planned trial set")
    attempts: set[str] = set()
    scores: dict[str, float] = {}
    references: dict[str, JSONValue] = {}
    for trial in plan.trials:
        run = runs[trial.slot]
        if (
            run.inputs != trial.inputs
            or run.status is not Status.SUCCEEDED
            or run.result is None
            or run.reused_from is not None
            or run.purpose is not Purpose.REPLICATION
        ):
            raise InvalidInputError(
                "Study evidence must be a matching successful replication"
            )
        if run.attempt_id in attempts:
            raise InvalidInputError("An attempt cannot count twice")
        attempts.add(run.attempt_id)
        metric = run.result.metrics.get("val_bpb")
        if (
            metric is None
            or metric.split != "validation"
            or metric.unit != "BPB"
            or metric.direction is not Direction.MINIMIZE
        ):
            raise InvalidInputError("S1 requires trusted validation BPB")
        scores[trial.slot] = metric.value
        references[trial.slot] = {
            "run_id": run.id,
            "attempt_id": run.attempt_id,
            "input_hash": run.inputs.hash,
            "val_bpb": metric.value,
            "observations": {
                name: value.value for name, value in run.result.observations.items()
            },
        }
    fidelities = tuple(dict.fromkeys(trial.inputs.fidelity for trial in plan.trials))
    if len(fidelities) != 2:
        raise InvalidInputError("S1 requires exactly two fidelities")
    fidelities = tuple(
        sorted(
            fidelities,
            key=lambda name: next(
                trial.inputs.budget_amount
                for trial in plan.trials
                if trial.inputs.fidelity == name
            ),
        )
    )
    noise: dict[str, JSONValue] = {}
    for fidelity in fidelities:
        noise[fidelity] = {
            cohort.value: _noise(
                tuple(
                    scores[trial.slot]
                    for trial in plan.trials
                    if trial.inputs.fidelity == fidelity
                    and trial.candidate == "baseline"
                    and trial.cohort is cohort
                )
            )
            for cohort in (Cohort.FIXED_SEED, Cohort.DISTINCT_SEED)
        }
    candidates = tuple(sorted({trial.candidate for trial in plan.trials}))
    panel: dict[str, tuple[tuple[float, ...], ...]] = {}
    for candidate in candidates:
        rows: list[tuple[float, ...]] = []
        for fidelity in fidelities:
            values: list[float] = []
            for seed in plan.rank_seeds:
                slots = [
                    trial.slot
                    for trial in plan.trials
                    if trial.candidate == candidate
                    and trial.inputs.fidelity == fidelity
                    and trial.inputs.seed == seed
                    and trial.cohort in (Cohort.RANK, Cohort.DISTINCT_SEED)
                ]
                if len(slots) != 1:
                    raise InvalidInputError(
                        "Rank panel needs one score per candidate, fidelity and seed"
                    )
                values.append(scores[slots[0]])
            rows.append(tuple(values))
        panel[candidate] = tuple(rows)
    means = [
        tuple(fmean(panel[candidate][i]) for candidate in candidates) for i in range(2)
    ]
    rho = spearman(means[0], means[1])
    rng = random.Random(20261010)
    bootstrap_rhos: list[float] = []
    undefined = 0
    for _ in range(2000):
        indices = rng.choices(range(len(plan.rank_seeds)), k=len(plan.rank_seeds))
        resampled = [
            tuple(
                fmean(panel[candidate][i][index] for index in indices)
                for candidate in candidates
            )
            for i in range(2)
        ]
        value = spearman(resampled[0], resampled[1])
        if value is None:
            undefined += 1
        else:
            bootstrap_rhos.append(value)
    return freeze_object(
        {
            "study": "S1 part 1",
            "plan_hash": plan.hash,
            "completed_runs": len(runs),
            "training_tokens": plan.training_tokens,
            "noise": noise,
            "rank_correlation": {
                "method": "Spearman over mean validation BPB by candidate",
                "fidelities": fidelities,
                "candidate_count": len(candidates),
                "seeds": plan.rank_seeds,
                "rho": rho,
                "matched_seed_bootstrap_95_percentile_interval": list(
                    _interval(bootstrap_rhos)
                )
                if bootstrap_rhos and not undefined
                else None,
                "undefined_bootstrap_count": undefined,
                "bootstrap_draws": 2000,
                "candidate_scores": {
                    candidate: {
                        fidelity: {
                            "mean_val_bpb": fmean(panel[candidate][i]),
                            "scores": panel[candidate][i],
                        }
                        for i, fidelity in enumerate(fidelities)
                    }
                    for candidate in candidates
                },
            },
            "evidence": references,
        }
    )
