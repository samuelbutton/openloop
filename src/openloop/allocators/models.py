"""Pure allocation contracts. Only trusted, scoped ledger records are evidence.

Allocation selects validation work. It neither reserves durable budget nor
executes jobs or authorizes confirmation and held-out evaluation.
"""

from dataclasses import dataclass, field
from typing import Protocol

from openloop.ledger import Direction, EventKind, Run, Status, content_hash
from openloop.ledger.validation import check_digest, check_positive
from openloop.loops import ContractError, Job, LoopAction, LoopSpec, Phase


@dataclass(frozen=True)
class AllocationSnapshot:
    """One loop and environment, offered jobs, and current public ledger evidence.

    Remaining budget excludes work already reserved by the coordinator. Slots
    are free worker slots. Jobs must come from the trusted workload adapter.
    Failed inputs need an explicit repair or retry outside this interface.
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
        check_positive(self.slots)
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


@dataclass(frozen=True)
class Allocation:
    """Ordered jobs and their full declared work cost, in the loop budget unit."""

    policy_hash: str
    jobs: tuple[Job, ...]

    def __post_init__(self) -> None:
        check_digest(self.policy_hash)
        jobs = tuple(self.jobs)
        if not all(isinstance(job, Job) for job in jobs):
            raise ContractError("Allocation requires typed jobs")
        if len({job.inputs.hash for job in jobs}) != len(jobs):
            raise ContractError("Allocation cannot repeat inputs")
        object.__setattr__(self, "jobs", jobs)

    @property
    def work_units(self) -> int:
        return sum(job.inputs.budget_amount for job in self.jobs)


class Allocator(Protocol):
    """Select eligible screen jobs within the snapshot budget and free slots."""

    def select(self, snapshot: AllocationSnapshot) -> Allocation: ...


def select_jobs(
    snapshot: AllocationSnapshot, ordered: tuple[Job, ...], policy: object
) -> Allocation:
    """Apply shared admission limits to a policy's ordered subset."""
    eligible = {job.inputs.hash: job for job in snapshot.eligible}
    remaining = snapshot.remaining_budget
    selected: list[Job] = []
    for job in ordered:
        if eligible.pop(job.inputs.hash, None) != job:
            raise ContractError("Policy returned an ineligible or duplicate job")
        if job.inputs.budget_amount <= remaining:
            selected.append(job)
            remaining -= job.inputs.budget_amount
            if len(selected) == snapshot.slots:
                break
    return Allocation(
        content_hash(
            {
                "allocator_version": 1,
                "policy": type(policy).__name__,
                "parameters": policy,
            }
        ),
        tuple(selected),
    )


def utility(snapshot: AllocationSnapshot, run: Run) -> float:
    """Orient a validated metric so larger is better; do not mix fidelities."""
    if run.result is None:
        raise ContractError("A score requires a completed result")
    value = run.result.metrics[snapshot.spec.metric_name].value
    return value if snapshot.spec.direction is Direction.MAXIMIZE else -value


def check_rungs(spec: LoopSpec, rungs: tuple[LoopAction, ...]) -> None:
    """Require increasing declared fidelities with explicit screen seeds."""
    previous = 0
    for rung in rungs:
        amount = spec.fidelity(rung.fidelity).amount
        if amount <= previous:
            raise ContractError("Rung fidelities must strictly increase")
        previous = amount


def check_seed(seed: int) -> None:
    """Reject implicit, negative, or boolean random seeds."""
    if type(seed) is not int or seed < 0:
        raise ContractError("Policy seed must be a nonnegative integer")
