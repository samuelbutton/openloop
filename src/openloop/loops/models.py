"""Pure loop contracts and records. Candidate settings cannot change frozen inputs."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from openloop.ledger.errors import InvalidInputError
from openloop.ledger.identity import (
    ExperimentInputs,
    JSONValue,
    Reproducibility,
    canonical_json,
    check_digest,
    check_positive,
    check_text,
    coerce_enum,
    content_hash,
    freeze_object,
)
from openloop.ledger.records import Direction, Purpose, Result


class ContractError(InvalidInputError):
    """A loop action, candidate, or output violates the declared contract."""


class FidelityName(StrEnum):
    SCREEN = "screen"
    CONFIRM = "confirm"


def finite_number(value: JSONValue, label: str) -> float:
    """Require a real finite number, without accepting booleans or numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite number")
    if not math.isfinite(value):
        raise ContractError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class Fidelity:
    name: FidelityName
    amount: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", coerce_enum(FidelityName, self.name, "fidelity")
        )
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
        if len(self.fidelities) != len(FidelityName) or {
            item.name for item in self.fidelities
        } != set(FidelityName):
            raise ContractError("Declare exactly one screen and one confirm fidelity")
        if (
            self.fidelity(FidelityName.SCREEN).amount
            > self.fidelity(FidelityName.CONFIRM).amount
        ):
            raise ContractError("Confirmation cannot cost less than screening")
        if len(set(self.mutable_keys)) != len(self.mutable_keys):
            raise ContractError("Mutable keys must be unique")
        for key in self.mutable_keys:
            check_text(key)
        canonical_json(self.execution)
        object.__setattr__(self, "execution", freeze_object(self.execution))
        object.__setattr__(self, "fidelities", tuple(self.fidelities))
        object.__setattr__(self, "mutable_keys", tuple(self.mutable_keys))

    @property
    def hash(self) -> str:
        return content_hash(self)

    def fidelity(self, name: FidelityName | str) -> Fidelity:
        selected = coerce_enum(FidelityName, name, "fidelity")
        return next(item for item in self.fidelities if item.name is selected)

    def inputs(
        self,
        config: Mapping[str, JSONValue],
        action: LoopAction,
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
            seed=action.seed,
            fidelity=str(action.fidelity),
            phase=str(action.fidelity),
            budget_unit=self.budget_unit,
            budget_amount=self.fidelity(action.fidelity).amount,
            reproducibility=self.reproducibility,
            config=config,
            execution=self.execution,
        )

    def validate_inputs(self, inputs: ExperimentInputs) -> None:
        expected = self.inputs(
            inputs.config,
            LoopAction(inputs.seed, self.fidelity(inputs.fidelity).name),
            inputs.environment_hash,
        )
        if inputs != expected:
            raise ContractError("Input manifest differs from the frozen loop contract")


@dataclass(frozen=True)
class LoopAction:
    seed: int
    fidelity: FidelityName | str = FidelityName.SCREEN
    request_key: str | None = None
    purpose: Purpose | str = Purpose.RUN
    retry_of: str | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("Seed must be a nonnegative integer")
        object.__setattr__(
            self, "fidelity", coerce_enum(FidelityName, self.fidelity, "fidelity")
        )
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
type Runner = Callable[[Job], Awaitable[JobOutput]]


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
    permitted_fidelities: tuple[FidelityName, ...]


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    run_id: str
    attempt_id: str
    result: Result | None
    executed: bool
    work_units: int
    episode_done: bool


class LoopEnv(Protocol):
    @property
    def spec(self) -> LoopSpec: ...

    async def initial_observation(self) -> Observation: ...

    async def step(self, action: LoopAction) -> StepResult: ...
