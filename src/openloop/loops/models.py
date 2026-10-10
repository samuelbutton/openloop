"""Pure loop contracts and records. Candidate settings cannot change frozen inputs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

from openloop.ledger import (
    Direction,
    ExperimentInputs,
    InvalidInputError,
    JSONValue,
    Purpose,
    Reproducibility,
    Result,
    Status,
    canonical_json,
    content_hash,
)
from openloop.ledger.validation import (
    check_digest,
    check_positive,
    check_text,
    coerce_enum,
    freeze_object,
)


class ContractError(InvalidInputError):
    """A loop action, candidate, or output violates the declared contract."""


class Phase(StrEnum):
    """Verification phase of a sample. Held-out evaluation is not selectable."""

    SCREEN = "screen"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Fidelity:
    """A named amount of work, in the loop's budget unit."""

    name: str
    amount: int

    def __post_init__(self) -> None:
        check_text(self.name)
        check_positive(self.amount)


@dataclass(frozen=True)
class LoopSpec:
    name: str
    source_hash: str
    dependencies_hash: str
    data_hash: str
    evaluator_hash: str
    adapter_hash: str
    budget_unit: str
    reproducibility: Reproducibility
    fidelities: tuple[Fidelity, ...]
    mutable_keys: tuple[str, ...]
    metric_name: str
    metric_unit: str
    direction: Direction = Direction.MINIMIZE
    tokenizer_hash: str | None = None
    execution: Mapping[str, JSONValue] = field(default_factory=dict[str, JSONValue])

    def __post_init__(self) -> None:
        for digest in (
            self.source_hash,
            self.dependencies_hash,
            self.data_hash,
            self.evaluator_hash,
            self.adapter_hash,
        ):
            check_digest(digest)
        if self.tokenizer_hash is not None:
            check_digest(self.tokenizer_hash)
        for text in (self.name, self.budget_unit, self.metric_name, self.metric_unit):
            check_text(text)
        object.__setattr__(
            self, "direction", coerce_enum(Direction, self.direction, "direction")
        )
        object.__setattr__(
            self,
            "reproducibility",
            coerce_enum(
                Reproducibility,
                self.reproducibility,
                "reproducibility",
            ),
        )
        fidelities = tuple(self.fidelities)
        if not fidelities:
            raise ContractError("Declare at least one fidelity")
        if not all(isinstance(item, Fidelity) for item in fidelities):
            raise ContractError("Fidelities must be typed")
        if len({item.name for item in fidelities}) != len(fidelities):
            raise ContractError("Fidelity names must be unique")
        if any(lower.amount >= higher.amount for lower, higher in pairwise(fidelities)):
            raise ContractError("Fidelity amounts must strictly increase")
        if len(set(self.mutable_keys)) != len(self.mutable_keys):
            raise ContractError("Mutable keys must be unique")
        for key in self.mutable_keys:
            check_text(key)
        canonical_json(self.execution)
        object.__setattr__(self, "execution", freeze_object(self.execution))
        object.__setattr__(self, "fidelities", fidelities)
        object.__setattr__(self, "mutable_keys", tuple(self.mutable_keys))

    @property
    def hash(self) -> str:
        return content_hash(self)

    def fidelity(self, name: str | None = None) -> Fidelity:
        """Return the named fidelity, or the lowest when no name is given."""
        if name is None:
            return self.fidelities[0]
        for item in self.fidelities:
            if item.name == name:
                return item
        raise ContractError(f"Unknown fidelity: {name!r}")

    def candidate_hash(self, config: Mapping[str, JSONValue]) -> str:
        """Identify a candidate for this loop without choosing a seed or fidelity."""
        return self._inputs(
            config, 0, Phase.SCREEN.value, self.fidelity().name, "0" * 64
        ).candidate_hash

    def inputs(
        self,
        config: Mapping[str, JSONValue],
        action: LoopAction,
        environment_hash: str,
    ) -> ExperimentInputs:
        return self._inputs(
            config,
            action.seed,
            str(action.phase),
            self.fidelity(action.fidelity).name,
            environment_hash,
        )

    def _inputs(
        self,
        config: Mapping[str, JSONValue],
        seed: int,
        phase: str,
        fidelity: str,
        environment_hash: str,
    ) -> ExperimentInputs:
        return ExperimentInputs(
            source_hash=self.source_hash,
            dependencies_hash=self.dependencies_hash,
            data_hash=self.data_hash,
            environment_hash=environment_hash,
            evaluator_hash=self.evaluator_hash,
            loop_hash=self.hash,
            tokenizer_hash=self.tokenizer_hash,
            seed=seed,
            fidelity=fidelity,
            phase=phase,
            budget_unit=self.budget_unit,
            budget_amount=self.fidelity(fidelity).amount,
            reproducibility=self.reproducibility,
            config=config,
            execution=self.execution,
        )

    def validate_inputs(self, inputs: ExperimentInputs) -> None:
        expected = self._inputs(
            inputs.config,
            inputs.seed,
            inputs.phase,
            self.fidelity(inputs.fidelity).name,
            inputs.environment_hash,
        )
        if inputs != expected or inputs.phase not in {item.value for item in Phase}:
            raise ContractError("Input manifest differs from the frozen loop contract")


@dataclass(frozen=True)
class LoopAction:
    """One sample request. `fidelity` None selects the lowest declared fidelity."""

    seed: int
    fidelity: str | None = None
    phase: Phase | str = Phase.SCREEN
    request_key: str | None = None
    purpose: Purpose | str = Purpose.RUN
    retry_of: str | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("Seed must be a nonnegative integer")
        if self.fidelity is not None:
            check_text(self.fidelity)
        object.__setattr__(self, "phase", coerce_enum(Phase, self.phase, "phase"))
        object.__setattr__(
            self, "purpose", coerce_enum(Purpose, self.purpose, "purpose")
        )
        if (self.purpose is Purpose.RETRY) != (self.retry_of is not None):
            raise ContractError("Only retries require retry_of")
        if self.request_key is not None:
            check_text(self.request_key)


@dataclass(frozen=True)
class Job:
    inputs: ExperimentInputs
    program: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, ExperimentInputs):
            raise ContractError("Jobs require typed experiment inputs")
        if self.program is not None:
            check_text(self.program)


type JobOutput = Mapping[str, JSONValue]


@dataclass(frozen=True)
class RunContext:
    """Identity of the admitted attempt that a runner executes.

    A replication has the same inputs as its baseline, so runners that need a
    unique key per execution use `attempt_id`, never the input hash.
    """

    run_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        check_text(self.run_id)
        check_text(self.attempt_id)
        if not self.run_id or not self.attempt_id:
            raise ContractError("Run context needs run and attempt identifiers")


type Runner = Callable[[Job, RunContext], Awaitable[JobOutput]]


class Workload(Protocol):
    @property
    def spec(self) -> LoopSpec: ...

    def normalize_config(
        self, config: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]: ...

    def build_job(self, inputs: ExperimentInputs) -> Job: ...

    def evaluate(self, job: Job, output: JobOutput) -> Result: ...


@dataclass(frozen=True)
class Observation:
    candidate_hash: str
    evidence: tuple[str, ...]
    budget_unit: str
    remaining_budget: int
    permitted_fidelities: tuple[str, ...]


@dataclass(frozen=True)
class StepResult:
    """Outcome of one step.

    `status` is the ledger status of the run: SUCCEEDED for a completed run
    (new or reused), QUEUED or RUNNING for a reference to pending work with no
    result yet, and FAILED when a request key replays a run that failed.
    """

    observation: Observation
    run_id: str
    attempt_id: str
    status: Status
    result: Result | None
    executed: bool
    work_units: int
    episode_done: bool


class LoopEnv(Protocol):
    @property
    def spec(self) -> LoopSpec: ...

    async def initial_observation(self) -> Observation: ...

    async def step(self, action: LoopAction) -> StepResult: ...
