"""Pure synchronous halving brackets. Failed work never supplies a loss score.

Hyperband uses a frozen finite candidate pool and the loop's declared fidelity
grid. It never interpolates an unsupported work amount or assumes resumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from openloop.ledger import Purpose, Run
from openloop.ledger.validation import check_digest
from openloop.loops import ContractError, Job, LoopAction, LoopSpec, Phase

from .models import (
    Allocation,
    AllocationSnapshot,
    check_rungs,
    check_seed,
    select_jobs,
    utility,
)


def _parameters(
    candidates: tuple[str, ...], rungs: tuple[LoopAction, ...], eta: int
) -> None:
    if not candidates or len(set(candidates)) != len(candidates):
        raise ContractError("A cohort requires unique candidate hashes")
    for candidate in candidates:
        check_digest(candidate)
    if type(eta) is not int or eta < 2:
        raise ContractError("Halving eta must be an integer of at least two")
    if not rungs:
        raise ContractError("Declare at least one rung")
    for rung in rungs:
        if not isinstance(rung, LoopAction) or (
            rung.phase != Phase.SCREEN
            or rung.purpose != Purpose.RUN
            or rung.request_key is not None
        ):
            raise ContractError(
                "Rungs require ordinary screen actions without request keys"
            )


def _matches(run: Run, action: LoopAction, spec: LoopSpec) -> bool:
    return (
        run.inputs.seed == action.seed
        and run.inputs.fidelity == spec.fidelity(action.fidelity).name
    )


def _bracket_jobs(
    snapshot: AllocationSnapshot, bracket: SuccessiveHalving
) -> tuple[tuple[Job, ...], bool]:
    check_rungs(snapshot.spec, bracket.rungs)
    active = bracket.candidates
    for index, action in enumerate(bracket.rungs):
        measured: dict[str, Run] = {}
        for run in snapshot.evidence:
            if _matches(run, action, snapshot.spec):
                # The first completed attempt closes this candidate's rung.
                # Later replications cannot revise an earlier promotion.
                measured.setdefault(run.inputs.candidate_hash, run)
        missing = set(active) - measured.keys()
        if missing:
            jobs = tuple(
                sorted(
                    (
                        job
                        for job in snapshot.eligible
                        if job.inputs.candidate_hash in missing
                        and job.inputs.seed == action.seed
                        and job.inputs.fidelity
                        == snapshot.spec.fidelity(action.fidelity).name
                    ),
                    key=lambda job: job.inputs.hash,
                )
            )
            return jobs, False
        if index + 1 < len(bracket.rungs):
            active = tuple(
                sorted(
                    active,
                    key=lambda candidate: (
                        -utility(snapshot, measured[candidate]),
                        candidate,
                    ),
                )
            )[: max(1, len(active) // bracket.eta)]
    return (), True


@dataclass(frozen=True)
class SuccessiveHalving:
    """Freeze a cohort and seed per rung; wait for all active measurements.

    Keep floor(n / eta), with at least one survivor, at each promotion. Each
    higher fidelity costs its full declared amount, not an incremental amount.
    """

    candidates: tuple[str, ...]
    rungs: tuple[LoopAction, ...]
    eta: int = field(default=3, kw_only=True)

    def __post_init__(self) -> None:
        candidates, rungs = tuple(self.candidates), tuple(self.rungs)
        _parameters(candidates, rungs, self.eta)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rungs", rungs)

    def select(self, snapshot: AllocationSnapshot) -> Allocation:
        jobs, _ = _bracket_jobs(snapshot, self)
        return select_jobs(snapshot, jobs, self)


@dataclass(frozen=True)
class Hyperband:
    """Random, disjoint synchronous brackets over a declared fidelity grid.

    For L rungs, bracket s starts at rung L-1-s with
    ceil(L * eta**s / (s+1)) candidates. This is a grid variant of Hyperband;
    geometric grids recover its bracket counts and resource ratios.
    """

    candidates: tuple[str, ...]
    rungs: tuple[LoopAction, ...]
    eta: int = field(default=3, kw_only=True)
    seed: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        candidates, rungs = tuple(self.candidates), tuple(self.rungs)
        _parameters(candidates, rungs, self.eta)
        check_seed(self.seed)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rungs", rungs)
        if len(candidates) < sum(self.bracket_sizes):
            raise ContractError(
                "Candidate pool is too small for disjoint Hyperband brackets"
            )

    @property
    def bracket_sizes(self) -> tuple[int, ...]:
        length = len(self.rungs)
        return tuple(
            (length * self.eta**s + s) // (s + 1) for s in reversed(range(length))
        )

    def brackets(self, spec: LoopSpec) -> tuple[SuccessiveHalving, ...]:
        """Return the frozen plan; caller order does not change random cohorts."""
        check_rungs(spec, self.rungs)
        pool = sorted(self.candidates)
        Random(self.seed).shuffle(pool)
        offset = 0
        brackets: list[SuccessiveHalving] = []
        for start, size in enumerate(self.bracket_sizes):
            brackets.append(
                SuccessiveHalving(
                    tuple(pool[offset : offset + size]),
                    self.rungs[start:],
                    eta=self.eta,
                )
            )
            offset += size
        return tuple(brackets)

    def select(self, snapshot: AllocationSnapshot) -> Allocation:
        for bracket in self.brackets(snapshot.spec):
            jobs, finished = _bracket_jobs(snapshot, bracket)
            if not finished:
                return select_jobs(snapshot, jobs, self)
        return select_jobs(snapshot, (), self)
