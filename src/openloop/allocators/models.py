"""Pure allocation contracts. Only trusted, scoped ledger records are evidence.

Allocation selects validation work. It neither reserves durable budget nor
executes jobs or authorizes confirmation and held-out evaluation.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from openloop.ledger import Direction, EventKind, Run, Status, content_hash
from openloop.ledger.validation import check_digest, check_text, coerce_enum
from openloop.loops import ContractError, Job, LoopSpec, Phase


@dataclass(frozen=True)
class AllocationSnapshot:
    """One loop and environment, offered jobs, and current public ledger evidence.

    Remaining budget excludes all work the coordinator has already committed:
    jobs that are queued or running, and the cost of any retry it has queued.
    Allocators therefore price only jobs that have not started. Slots are free
    worker slots; zero is valid and yields a NO_SLOTS allocation. Jobs must come
    from the trusted workload adapter. Failed inputs need an explicit repair or
    retry outside this interface.
    """

    spec: LoopSpec
    environment_hash: str
    jobs: tuple[Job, ...]
    runs: tuple[Run, ...]
    remaining_budget: int
    slots: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, LoopSpec):
            raise ContractError("Allocation requires a typed loop specification")
        check_digest(self.environment_hash)
        if type(self.remaining_budget) is not int or self.remaining_budget < 0:
            raise ContractError("Remaining budget must be a nonnegative integer")
        if type(self.slots) is not int or self.slots < 0:
            raise ContractError("Free slots must be a nonnegative integer")
        jobs, runs = tuple(self.jobs), tuple(self.runs)
        for job in jobs:
            if not isinstance(job, Job):
                raise ContractError("Allocation requires typed jobs")
            self.spec.validate_inputs(job.inputs)
            if job.inputs.environment_hash != self.environment_hash:
                raise ContractError("Job environment differs from the snapshot")
            if job.inputs.phase != Phase.SCREEN:
                raise ContractError("Allocation selects screen jobs only")
        if len({job.inputs.hash for job in jobs}) != len(jobs):
            raise ContractError("Offered input hashes must be unique")
        attempts: dict[str, Run] = {}
        for run in runs:
            if not isinstance(run, Run):
                raise ContractError("Allocation requires typed ledger runs")
            self.spec.validate_inputs(run.inputs)
            if run.inputs.environment_hash != self.environment_hash:
                raise ContractError("Evidence environment differs from the snapshot")
            previous = attempts.get(run.attempt_id)
            if previous is not None and (
                previous.inputs != run.inputs
                or previous.status != run.status
                or previous.result != run.result
            ):
                raise ContractError("Conflicting records for one attempt")
            attempts[run.attempt_id] = run
            if run.status is Status.SUCCEEDED:
                if not run.events or run.events[-1].kind is not EventKind.SUCCEEDED:
                    raise ContractError(
                        "Successful evidence requires a completion event"
                    )
                metric = (
                    run.result.metrics.get(self.spec.metric_name)
                    if run.result is not None
                    else None
                )
                if metric is None or (
                    metric.unit != self.spec.metric_unit
                    or metric.direction != self.spec.direction
                    or metric.split != "validation"
                ):
                    raise ContractError(
                        "Evidence requires the declared validation metric"
                    )
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "runs", runs)

    @property
    def evidence(self) -> tuple[Run, ...]:
        """Successful screen attempts, counted once even with cache aliases."""
        by_attempt: dict[str, Run] = {}
        completed = (
            run
            for run in self.runs
            if run.status is Status.SUCCEEDED and run.inputs.phase == Phase.SCREEN
        )
        for run in sorted(
            completed,
            key=lambda run: (
                run.events[-1].timestamp,
                run.attempt_id,
                run.created_at,
                run.id,
            ),
        ):
            by_attempt.setdefault(run.attempt_id, run)
        return tuple(by_attempt.values())

    @property
    def eligible(self) -> tuple[Job, ...]:
        """Fresh inputs only; cached, pending, and failed inputs stay out."""
        seen = {run.inputs.hash for run in self.runs}
        return tuple(job for job in self.jobs if job.inputs.hash not in seen)


class AllocationStatus(StrEnum):
    """Why an allocation holds the jobs it holds, or none."""

    SELECTED = "selected"
    DONE = "done"
    WAITING = "waiting"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_SLOTS = "no_slots"
    MISSING_JOBS = "missing_jobs"


@dataclass(frozen=True)
class Allocation:
    """Ordered jobs, their full declared work cost, and why this is the answer.

    SELECTED holds jobs; every other status holds none. DONE: nothing left for
    this policy. WAITING: results are pending. BLOCKED: a failed input needs a
    retry. BUDGET_EXHAUSTED: the remaining budget cannot fund the next step.
    NO_SLOTS: no free worker slot. MISSING_JOBS: the caller did not offer a job
    the policy needs. Only DONE is final.
    """

    policy_hash: str
    jobs: tuple[Job, ...]
    status: AllocationStatus
    reason: str

    def __post_init__(self) -> None:
        check_digest(self.policy_hash)
        check_text(self.reason)
        jobs = tuple(self.jobs)
        if not all(isinstance(job, Job) for job in jobs):
            raise ContractError("Allocation requires typed jobs")
        if len({job.inputs.hash for job in jobs}) != len(jobs):
            raise ContractError("Allocation cannot repeat inputs")
        status = coerce_enum(AllocationStatus, self.status, "allocation status")
        if (status is AllocationStatus.SELECTED) != bool(jobs):
            raise ContractError("Only a SELECTED allocation holds jobs")
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "status", status)

    @property
    def work_units(self) -> int:
        return sum(job.inputs.budget_amount for job in self.jobs)


@dataclass(frozen=True)
class ScreenOutcome:
    """A finished allocator run, stated for use as a decider screen.

    `candidates` is the whole declared pool, `survivors` the subset it kept, and
    `evidence` the ledger run ids that scored each candidate (empty if never
    run). `seeds` is every seed the allocator may use. A promotion here is a
    selection, not a test: confirmation needs fresh seeds outside `seeds`.
    """

    policy_hash: str
    candidates: tuple[str, ...]
    survivors: tuple[str, ...]
    evidence: Mapping[str, tuple[str, ...]]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        check_digest(self.policy_hash)
        candidates, survivors = tuple(self.candidates), tuple(self.survivors)
        if not set(survivors) <= set(candidates) or len(set(survivors)) != len(
            survivors
        ):
            raise ContractError("Survivors must be unique members of the pool")
        if set(self.evidence) != set(candidates):
            raise ContractError("Evidence must cover every pool member")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "survivors", survivors)
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(
                {key: tuple(value) for key, value in self.evidence.items()}
            ),
        )
        object.__setattr__(self, "seeds", tuple(sorted(set(self.seeds))))


class Allocator(Protocol):
    """Select eligible screen jobs within the snapshot budget and free slots."""

    def select(self, snapshot: AllocationSnapshot) -> Allocation: ...


def policy_hash(policy: object) -> str:
    """Identify a policy by type, version, and every declared parameter."""
    return content_hash(
        {
            "allocator_version": 2,
            "policy": type(policy).__name__,
            "parameters": policy,
        }
    )


def select_jobs(
    snapshot: AllocationSnapshot,
    ordered: tuple[Job, ...],
    policy: object,
    *,
    atomic: bool = False,
    empty: AllocationStatus = AllocationStatus.DONE,
    note: str = "",
) -> Allocation:
    """Apply shared admission limits to a policy's ordered subset.

    A non-atomic policy funds each job that fits, in order. An atomic policy
    funds all of `ordered` or none of it, so a unit of work that is useful only
    when complete never starts without the budget to finish it. `empty` and
    `note` explain an empty `ordered` and add detail to any reason.
    """
    eligible = {job.inputs.hash: job for job in snapshot.eligible}
    for job in ordered:
        if eligible.pop(job.inputs.hash, None) != job:
            raise ContractError("Policy returned an ineligible or duplicate job")
    identity = policy_hash(policy)
    detail = f" {note}" if note else ""
    if not ordered:
        return Allocation(identity, (), empty, (note or f"No work: {empty.value}"))
    if snapshot.slots == 0:
        return Allocation(
            identity, (), AllocationStatus.NO_SLOTS, f"No free slots.{detail}"
        )
    remaining = snapshot.remaining_budget
    if atomic:
        cost = sum(job.inputs.budget_amount for job in ordered)
        if cost > remaining:
            return Allocation(
                identity,
                (),
                AllocationStatus.BUDGET_EXHAUSTED,
                f"The step needs {cost} work units and {remaining} remain; "
                f"none started.{detail}",
            )
    selected: list[Job] = []
    for job in ordered:
        if job.inputs.budget_amount <= remaining:
            selected.append(job)
            remaining -= job.inputs.budget_amount
            if len(selected) == snapshot.slots:
                break
    if not selected:
        return Allocation(
            identity,
            (),
            AllocationStatus.BUDGET_EXHAUSTED,
            f"No offered job fits the {snapshot.remaining_budget} remaining "
            f"work units.{detail}",
        )
    return Allocation(
        identity,
        tuple(selected),
        AllocationStatus.SELECTED,
        f"Selected {len(selected)} of {len(ordered)} ranked jobs.{detail}",
    )


def utility(snapshot: AllocationSnapshot, run: Run) -> float:
    """Orient a validated metric so larger is better; do not mix fidelities."""
    if run.result is None:
        raise ContractError("A score requires a completed result")
    value = run.result.metrics[snapshot.spec.metric_name].value
    return value if snapshot.spec.direction is Direction.MAXIMIZE else -value


def check_seed(seed: int) -> None:
    """Reject implicit, negative, or boolean random seeds."""
    if type(seed) is not int or seed < 0:
        raise ContractError("Policy seed must be a nonnegative integer")
