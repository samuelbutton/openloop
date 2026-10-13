"""Pure synchronous halving brackets. Failed work never supplies a loss score.

Hyperband uses a frozen finite candidate pool and the loop's declared fidelity
grid. It never interpolates an unsupported work amount or assumes resumption.
Rungs start only when the remaining budget covers every job of the rung that
has not started, so a rung is never funded halfway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from statistics import fmean

from openloop.ledger.validation import check_digest, coerce_enum
from openloop.loops import ContractError, Job, LoopSpec

from .models import (
    Allocation,
    AllocationSnapshot,
    AllocationStatus,
    ScreenOutcome,
    check_seed,
    policy_hash,
    select_jobs,
    utility,
)
from .rungs import InputIndex, OnFailure, Rung, check_rungs


def _parameters(
    candidates: tuple[str, ...], rungs: tuple[Rung, ...], eta: int, on_failure: object
) -> OnFailure:
    if not candidates or len(set(candidates)) != len(candidates):
        raise ContractError("A cohort requires unique candidate hashes")
    for candidate in candidates:
        check_digest(candidate)
    if type(eta) is not int or eta < 2:
        raise ContractError("Halving eta must be an integer of at least two")
    if not rungs or not all(isinstance(rung, Rung) for rung in rungs):
        raise ContractError("Declare at least one rung, each a Rung")
    return coerce_enum(OnFailure, on_failure, "failure policy")


@dataclass(frozen=True)
class _Progress:
    """Where one bracket stands, from public evidence alone."""

    finished: bool
    status: AllocationStatus
    reason: str
    jobs: tuple[Job, ...] = ()
    finalists: dict[str, float] = field(default_factory=dict[str, float])
    evidence: dict[str, list[str]] = field(default_factory=dict[str, list[str]])


def _progress(snapshot: AllocationSnapshot, bracket: SuccessiveHalving) -> _Progress:
    check_rungs(snapshot.spec, bracket.rungs)
    index = InputIndex.build(snapshot)
    active = bracket.candidates
    evidence: dict[str, list[str]] = {item: [] for item in bracket.candidates}
    for position, rung in enumerate(bracket.rungs):
        fidelity = snapshot.spec.fidelity(rung.fidelity).name
        scores: dict[str, float] = {}
        start: list[Job] = []
        pending = blocked = missing = 0
        for candidate in active:
            values: list[float] = []
            fresh: list[Job] = []
            waiting = broken = absent = 0
            for seed in rung.seeds:
                key = candidate, seed, fidelity
                run = index.measured.get(key)
                if run is not None:
                    values.append(utility(snapshot, run))
                    evidence[candidate].append(run.id)
                elif key in index.pending:
                    waiting += 1
                elif key in index.failed:
                    broken += 1
                elif key in index.fresh:
                    fresh.append(index.fresh[key])
                else:
                    absent += 1
            if broken and bracket.on_failure is OnFailure.RANK_WORST:
                # Ranked last and never promoted. Its other seeds are not worth
                # starting, and the failure is not a sample.
                continue
            start += fresh
            pending, blocked, missing = (
                pending + waiting,
                blocked + broken,
                missing + absent,
            )
            if len(values) == len(rung.seeds):
                scores[candidate] = fmean(values)
        where = f"rung {position + 1} of {len(bracket.rungs)} ({fidelity})"
        if missing:
            return _Progress(
                False,
                AllocationStatus.MISSING_JOBS,
                f"{missing} fresh inputs at {where} were not offered; none started",
            )
        if start:
            ordered = tuple(sorted(start, key=lambda job: job.inputs.hash))
            return _Progress(
                False,
                AllocationStatus.SELECTED,
                f"Starting {len(ordered)} inputs at {where}",
                ordered,
            )
        if blocked:
            return _Progress(
                False,
                AllocationStatus.BLOCKED,
                f"{blocked} failed inputs at {where} need a retry",
            )
        if pending:
            return _Progress(
                False,
                AllocationStatus.WAITING,
                f"{pending} inputs at {where} are not complete",
            )
        if position + 1 == len(bracket.rungs):
            return _Progress(
                True,
                AllocationStatus.DONE,
                f"Bracket complete with {len(scores)} finalists",
                finalists=scores,
                evidence=evidence,
            )
        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        active = tuple(ranked[: max(1, len(active) // bracket.eta)])
        if not active:
            return _Progress(
                True,
                AllocationStatus.DONE,
                f"Every candidate failed at {where}",
                evidence=evidence,
            )
    raise AssertionError("A bracket always has a final rung")


def _allocate(
    snapshot: AllocationSnapshot, progress: _Progress, policy: object
) -> Allocation:
    return select_jobs(
        snapshot,
        progress.jobs,
        policy,
        atomic=True,
        empty=progress.status,
        note=progress.reason,
    )


def _outcome(
    snapshot: AllocationSnapshot,
    policy: object,
    pool: tuple[str, ...],
    brackets: tuple[SuccessiveHalving, ...],
    keep: int | None,
) -> ScreenOutcome:
    if keep is not None and (type(keep) is not int or keep < 1):
        raise ContractError("Survivor count must be a positive integer")
    finalists: dict[str, float] = {}
    evidence: dict[str, tuple[str, ...]] = dict.fromkeys(pool, ())
    for bracket in brackets:
        progress = _progress(snapshot, bracket)
        if not progress.finished:
            raise ContractError("An unfinished allocation cannot be a screen")
        # Every bracket ends at the same final rung, so these scores compare.
        finalists.update(progress.finalists)
        for candidate, runs in progress.evidence.items():
            evidence[candidate] = tuple(runs)
    ranked = sorted(finalists, key=lambda item: (-finalists[item], item))
    return ScreenOutcome(
        policy_hash(policy),
        pool,
        tuple(ranked if keep is None else ranked[:keep]),
        evidence,
        tuple(
            seed
            for bracket in brackets
            for rung in bracket.rungs
            for seed in rung.seeds
        ),
    )


@dataclass(frozen=True)
class SuccessiveHalving:
    """Freeze a cohort and the seeds per rung; wait for all active measurements.

    Keep floor(n / eta), with at least one survivor, at each promotion. Each
    higher fidelity costs its full declared amount, not an incremental amount.
    A rung declares one or more seeds; a candidate's rung score is its mean over
    them, and every seed must complete. More seeds shrink the winner's curse in
    a promotion (the best of n noisy scores is biased high) but multiply rung
    cost.

    A rung starts only if the remaining budget covers all of its unstarted jobs.
    Otherwise nothing starts and the status is BUDGET_EXHAUSTED. Work already
    queued or running is the coordinator's reservation and is not priced again.
    `on_failure` declares the response to a terminal failure; see `OnFailure`.
    """

    candidates: tuple[str, ...]
    rungs: tuple[Rung, ...]
    eta: int = field(default=3, kw_only=True)
    on_failure: OnFailure = field(default=OnFailure.BLOCK, kw_only=True)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        rungs = tuple(self.rungs)
        policy = _parameters(candidates, rungs, self.eta, self.on_failure)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rungs", rungs)
        object.__setattr__(self, "on_failure", policy)

    def select(self, snapshot: AllocationSnapshot) -> Allocation:
        return _allocate(snapshot, _progress(snapshot, self), self)

    def outcome(
        self, snapshot: AllocationSnapshot, *, survivors: int | None = None
    ) -> ScreenOutcome:
        """State a finished run as a screen; keep the best `survivors` finalists."""
        return _outcome(snapshot, self, self.candidates, (self,), survivors)


@dataclass(frozen=True)
class Hyperband:
    """Random, disjoint synchronous brackets over a declared fidelity grid.

    For L rungs, bracket s starts at rung L-1-s with
    ceil(L * eta**s / (s+1)) candidates. This is a grid variant of Hyperband;
    geometric grids recover its bracket counts and resource ratios.

    Brackets run in order. If the next rung of the current bracket is
    unaffordable, the allocator stops with BUDGET_EXHAUSTED and never skips to a
    cheaper later bracket. Skipping would spend the budget on a different mix
    of brackets than the declared plan and make equal-compute comparisons
    between runs uninterpretable. Pool members outside every bracket are never
    run.
    """

    candidates: tuple[str, ...]
    rungs: tuple[Rung, ...]
    eta: int = field(default=3, kw_only=True)
    seed: int = field(default=0, kw_only=True)
    on_failure: OnFailure = field(default=OnFailure.BLOCK, kw_only=True)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        rungs = tuple(self.rungs)
        policy = _parameters(candidates, rungs, self.eta, self.on_failure)
        check_seed(self.seed)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rungs", rungs)
        object.__setattr__(self, "on_failure", policy)
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
                    on_failure=self.on_failure,
                )
            )
            offset += size
        return tuple(brackets)

    def select(self, snapshot: AllocationSnapshot) -> Allocation:
        for bracket in self.brackets(snapshot.spec):
            progress = _progress(snapshot, bracket)
            if not progress.finished:
                return _allocate(snapshot, progress, self)
        return select_jobs(
            snapshot,
            (),
            self,
            empty=AllocationStatus.DONE,
            note="Every bracket is complete",
        )

    def outcome(
        self, snapshot: AllocationSnapshot, *, survivors: int | None = None
    ) -> ScreenOutcome:
        """State a finished run as a screen; keep the best `survivors` finalists.

        Finalists are scored at the shared final rung, so they rank across
        brackets. Pool members outside every bracket have no evidence.
        """
        return _outcome(
            snapshot,
            self,
            tuple(sorted(self.candidates)),
            self.brackets(snapshot.spec),
            survivors,
        )
