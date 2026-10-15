"""Pure local training contracts. No framework imports, jobs, or model downloads."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Protocol

from openloop.ledger import InvalidInputError, content_hash
from openloop.ledger.validation import check_digest


class TrainingError(InvalidInputError):
    """A training request or numerical result violates the backend contract."""


class Objective(StrEnum):
    CROSS_ENTROPY = "cross_entropy"
    GRPO = "grpo"


def sampling_seed(seed: int, sample: int) -> int:
    """Namespace request/sample randomness separately from model initialization."""
    return int(
        content_hash(
            {
                "domain": "openloop.training.sample.v1",
                "request_seed": seed,
                "sample": sample,
            }
        )[:16],
        16,
    )


def finite(value: float, name: str) -> float:
    """Require a finite real value, excluding booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise TrainingError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class Datum:
    """Causal token sequence; only tokens at completion_start onward carry loss."""

    tokens: tuple[int, ...]
    completion_start: int
    advantage: float = 0.0
    old_logprobs: tuple[float, ...] = ()
    reference_logprobs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        tokens = tuple(self.tokens)
        if any(type(token) is not int or token < 0 for token in tokens):
            raise TrainingError("Tokens must be nonnegative integers")
        if type(
            self.completion_start
        ) is not int or not 1 <= self.completion_start < len(tokens):
            raise TrainingError("A datum needs a prompt and a nonempty completion")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "advantage", finite(self.advantage, "Advantage"))
        for name in ("old_logprobs", "reference_logprobs"):
            values = tuple(finite(value, name) for value in getattr(self, name))
            if values and len(values) != self.completion_tokens:
                raise TrainingError(f"{name} must match completion length")
            if any(value > 0 for value in values):
                raise TrainingError("Log probabilities cannot be positive")
            object.__setattr__(self, name, values)

    @property
    def completion_tokens(self) -> int:
        return len(self.tokens) - self.completion_start


@dataclass(frozen=True)
class LossConfig:
    objective: Objective | str = Objective.GRPO
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "objective", Objective(self.objective))
        except ValueError as error:
            raise TrainingError("Unknown training objective") from error
        clip = finite(self.clip_epsilon, "Clip epsilon")
        beta = finite(self.kl_beta, "KL beta")
        if not 0 < clip < 1 or beta < 0:
            raise TrainingError("Clip epsilon must be in (0,1) and KL beta nonnegative")

    def validate(self, data: tuple[Datum, ...]) -> None:
        """Check the entire batch before any backend accumulates gradients."""
        if not data or not all(isinstance(row, Datum) for row in data):
            raise TrainingError("Training requires a nonempty batch of Datum records")
        if self.objective is Objective.GRPO:
            for row in data:
                if not row.old_logprobs or (
                    self.kl_beta and not row.reference_logprobs
                ):
                    raise TrainingError(
                        "GRPO needs old and, with KL, reference logprobs"
                    )


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite(getattr(self, name), name)
            if value < 0:
                raise TrainingError(f"{name} must be nonnegative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise TrainingError("Adam betas must be in [0,1)")
        if min(self.learning_rate, self.epsilon, self.max_grad_norm) <= 0:
            raise TrainingError(
                "Learning rate, epsilon, and gradient limit must be positive"
            )


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    num_samples: int = 4
    temperature: float = 1.0
    seed: int = 0
    stop_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 256:
            raise TrainingError("Generation limit must be 1-256 tokens")
        if type(self.num_samples) is not int or not 1 <= self.num_samples <= 16:
            raise TrainingError("Sample count must be 1-16")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - 8192:
            raise TrainingError("Sampling seed must fit uint32 with room for the group")
        if self.temperature not in (0.0, 1.0) or isinstance(self.temperature, bool):
            raise TrainingError(
                "Use temperature 1 for policy sampling or 0 for greedy evaluation"
            )
        stops = tuple(self.stop_tokens)
        if any(type(token) is not int or token < 0 for token in stops):
            raise TrainingError("Stop tokens must be nonnegative integers")
        object.__setattr__(self, "stop_tokens", stops)


@dataclass(frozen=True)
class Sample:
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]

    def __post_init__(self) -> None:
        Datum((0, *self.tokens), 1, old_logprobs=self.logprobs)
        if len(self.tokens) != len(self.logprobs):
            raise TrainingError("Sample tokens and logprobs must match")
        object.__setattr__(self, "tokens", tuple(self.tokens))
        object.__setattr__(self, "logprobs", tuple(self.logprobs))


@dataclass(frozen=True)
class ForwardBackwardResult:
    loss: float
    completion_tokens: int

    def __post_init__(self) -> None:
        finite(self.loss, "Loss")
        if type(self.completion_tokens) is not int or self.completion_tokens < 1:
            raise TrainingError(
                "Training result needs a positive completion-token count"
            )


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        check_digest(self.sha256)


class TrainingClient(Protocol):
    """Single-owner, synchronous Tinker-shaped operations on an already loaded model."""

    def logprobs(self, datum: Datum) -> tuple[float, ...]: ...
    def forward_backward(
        self, data: tuple[Datum, ...], config: LossConfig
    ) -> ForwardBackwardResult: ...
    def optim_step(self) -> Mapping[str, float]: ...
    def sample(
        self, prompt: tuple[int, ...], params: SamplingParams
    ) -> tuple[Sample, ...]: ...
    def save_state(self, path: Path) -> Checkpoint: ...
    def load_state(self, path: Path) -> None: ...
