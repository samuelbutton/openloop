"""Rung declarations and per-input state for halving brackets; no I/O."""

from dataclasses import dataclass
from enum import StrEnum

from openloop.ledger import Run, Status
from openloop.ledger.validation import check_text
from openloop.loops import ContractError, Job, LoopSpec, Phase

from .models import AllocationSnapshot


class OnFailure(StrEnum):
    """What a terminally failed candidate does to its rung.

    BLOCK holds the rung until the coordinator retries the input. RANK_WORST
    ranks the candidate below every scored candidate and never promotes it; the
    failure stays in the ledger and is never a sample.
    """

    BLOCK = "block"
    RANK_WORST = "rank_worst"


@dataclass(frozen=True)
class Rung:
    """One fidelity and the distinct seeds whose mean score promotes a candidate."""

    fidelity: str | None
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.fidelity is not None:
            check_text(self.fidelity)
        seeds = tuple(self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ContractError("A rung requires one or more distinct seeds")
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ContractError("Rung seeds must be nonnegative integers")
        object.__setattr__(self, "seeds", seeds)


def check_rungs(spec: LoopSpec, rungs: tuple[Rung, ...]) -> None:
    """Require increasing declared fidelities."""
    previous = 0
    for rung in rungs:
        amount = spec.fidelity(rung.fidelity).amount
        if amount <= previous:
            raise ContractError("Rung fidelities must strictly increase")
        previous = amount


type Key = tuple[str, int, str]


@dataclass(frozen=True)
class InputIndex:
    """Screen inputs by candidate, seed, and fidelity name, from public evidence.

    A completed attempt closes its input: the first one wins, and later
    replications cannot revise a promotion. A queued or running attempt makes
    the input pending even after an earlier failure (a retry). A failure with no
    pending or completed attempt is terminal until the coordinator retries it.
    """

    measured: dict[Key, Run]
    pending: frozenset[Key]
    failed: frozenset[Key]
    fresh: dict[Key, Job]

    @classmethod
    def build(cls, snapshot: AllocationSnapshot) -> "InputIndex":
        measured: dict[Key, Run] = {}
        for run in snapshot.evidence:
            measured.setdefault(_key(run), run)
        pending: set[Key] = set()
        failed: set[Key] = set()
        for run in snapshot.runs:
            if run.inputs.phase != Phase.SCREEN:
                continue
            if run.status in (Status.QUEUED, Status.RUNNING):
                pending.add(_key(run))
            elif run.status is Status.FAILED:
                failed.add(_key(run))
        fresh = {_key(job): job for job in snapshot.eligible}
        return cls(measured, frozenset(pending), frozenset(failed), fresh)


def _key(item: Run | Job) -> Key:
    inputs = item.inputs
    return inputs.candidate_hash, inputs.seed, inputs.fidelity
