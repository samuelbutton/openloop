"""Pure fixed-fidelity policies. UCB rewards require declared finite bounds."""

import math
from dataclasses import dataclass, field
from statistics import fmean

from openloop.ledger import Direction, content_hash
from openloop.ledger.validation import check_text, finite_number
from openloop.loops import ContractError

from .models import Allocation, AllocationSnapshot, check_seed, select_jobs, utility


def _samples(snapshot: AllocationSnapshot, fidelity: str) -> dict[str, list[float]]:
    snapshot.spec.fidelity(fidelity)
    candidates = {
        job.inputs.candidate_hash
        for job in snapshot.jobs
        if job.inputs.fidelity == fidelity
    }
    samples: dict[str, list[float]] = {}
    for run in snapshot.evidence:
        if run.inputs.fidelity == fidelity and run.inputs.candidate_hash in candidates:
            samples.setdefault(run.inputs.candidate_hash, []).append(
                utility(snapshot, run)
            )
    return samples


@dataclass(frozen=True)
class Greedy:
    """Explore unseen candidates first, then prefer the largest observed mean.

    Compare one declared fidelity only. Input hash breaks score ties.
    """

    fidelity: str

    def __post_init__(self) -> None:
        check_text(self.fidelity)

    def select(self, snapshot: AllocationSnapshot) -> Allocation:
        samples = _samples(snapshot, self.fidelity)
        jobs = tuple(
            job for job in snapshot.eligible if job.inputs.fidelity == self.fidelity
        )
        ordered = tuple(
            sorted(
                jobs,
                key=lambda job: (
                    -fmean(samples[job.inputs.candidate_hash])
                    if job.inputs.candidate_hash in samples
                    else -math.inf,
                    job.inputs.hash,
                ),
            )
        )
        return select_jobs(snapshot, ordered, self)


@dataclass(frozen=True)
class Random:
    """Seeded random priorities for fresh jobs at one declared fidelity.

    Priorities depend on full input hashes, not caller order or process state.
    """

    fidelity: str
    seed: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        check_text(self.fidelity)
        check_seed(self.seed)

    def select(self, snapshot: AllocationSnapshot) -> Allocation:
        snapshot.spec.fidelity(self.fidelity)
        jobs = tuple(
            job for job in snapshot.eligible if job.inputs.fidelity == self.fidelity
        )
        ordered = tuple(
            sorted(
                jobs,
                key=lambda job: content_hash(
                    {
                        "seed": self.seed,
                        "inputs": job.inputs.hash,
                    }
                ),
            )
        )
        return select_jobs(snapshot, ordered, self)


@dataclass(frozen=True)
class Bandit:
    """UCB1 at one fidelity: mean reward + sqrt(2 log(total pulls) / pulls).

    Raw metric bounds are frozen in advance. Out-of-range evidence is an error,
    never clipped. Unseen candidates precede measured candidates.
    """

    fidelity: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        check_text(self.fidelity)
        lower = finite_number(self.lower, "Reward lower bound")
        upper = finite_number(self.upper, "Reward upper bound")
        if lower >= upper or not math.isfinite(upper - lower):
            raise ContractError("Reward bounds must have a positive finite range")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def select(self, snapshot: AllocationSnapshot) -> Allocation:
        samples = _samples(snapshot, self.fidelity)
        # Samples already have the direction applied; these bounds must match.
        lower, upper = (
            (self.lower, self.upper)
            if snapshot.spec.direction is Direction.MAXIMIZE
            else (-self.upper, -self.lower)
        )
        rewards: dict[str, list[float]] = {}
        for candidate, values in samples.items():
            if any(value < lower or value > upper for value in values):
                raise ContractError(
                    "Validation metric is outside the frozen reward bounds"
                )
            rewards[candidate] = [(value - lower) / (upper - lower) for value in values]
        total = sum(len(values) for values in rewards.values())
        indices = {
            candidate: fmean(values)
            + math.sqrt(2 * math.log(max(1, total)) / len(values))
            for candidate, values in rewards.items()
        }
        jobs = tuple(
            job for job in snapshot.eligible if job.inputs.fidelity == self.fidelity
        )
        ordered = tuple(
            sorted(
                jobs,
                key=lambda job: (
                    -indices.get(job.inputs.candidate_hash, math.inf),
                    job.inputs.hash,
                ),
            )
        )
        return select_jobs(snapshot, ordered, self)
