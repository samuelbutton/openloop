"""Bounded GRPO orchestration over the training contract; rewards stay host-owned."""

from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from time import perf_counter
from typing import Protocol

from openloop.ledger import content_hash
from openloop.ledger.validation import check_digest

from .contracts import (
    Datum,
    LossConfig,
    Objective,
    SamplingParams,
    TrainingClient,
    TrainingError,
    finite,
)
from .loss import group_advantages


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class ArithmeticOperator(StrEnum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"


@dataclass(frozen=True)
class ArithmeticTask:
    left: int
    right: int
    operator: ArithmeticOperator | str

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or not 0 <= value < 20
            for value in (self.left, self.right)
        ):
            raise TrainingError("Arithmetic operands must be integers in [0,20)")
        try:
            object.__setattr__(self, "operator", ArithmeticOperator(self.operator))
        except ValueError as error:
            raise TrainingError("Unknown arithmetic operation") from error

    @property
    def answer(self) -> int:
        if self.operator is ArithmeticOperator.ADD:
            return self.left + self.right
        if self.operator is ArithmeticOperator.SUBTRACT:
            return self.left - self.right
        return self.left * self.right

    @property
    def prompt(self) -> str:
        return (
            f"What is {self.left} {self.operator} {self.right}? "
            "Reply with just the integer result."
        )


def reward(task: ArithmeticTask, text: str) -> float:
    """Accept only the complete exact integer response; execute no generated text."""
    import re

    match = re.fullmatch(r"\s*(-?\d{1,4})\s*", text)
    return float(match is not None and int(match.group(1)) == task.answer)


@dataclass(frozen=True)
class ArithmeticData:
    train: tuple[ArithmeticTask, ...]
    validation: tuple[ArithmeticTask, ...]

    def __post_init__(self) -> None:
        for name in ("train", "validation"):
            tasks = tuple(getattr(self, name))
            if not tasks or not all(isinstance(task, ArithmeticTask) for task in tasks):
                raise TrainingError("Arithmetic splits need typed tasks")
            hashes = [content_hash(task) for task in tasks]
            if len(set(hashes)) != len(tasks):
                raise TrainingError("Arithmetic split contains duplicates")
            object.__setattr__(self, name, tasks)
        if {content_hash(task) for task in self.train} & {
            content_hash(task) for task in self.validation
        }:
            raise TrainingError("Arithmetic train and validation overlap")

    @classmethod
    def default(cls) -> "ArithmeticData":
        """Freeze 64 train / 16 validation expressions by content hash; reserve 20%."""
        tasks = sorted(
            (
                ArithmeticTask(a, b, op)
                for a in range(20)
                for b in range(20)
                for op in ArithmeticOperator
            ),
            key=content_hash,
        )
        train = tuple(
            task for task in tasks if int(content_hash(task)[:8], 16) % 10 < 7
        )[:64]
        validation = tuple(
            task for task in tasks if int(content_hash(task)[:8], 16) % 10 == 7
        )[:16]
        return cls(train, validation)


@dataclass(frozen=True)
class Rollout:
    task_hash: str
    text: str
    reward: float
    advantage: float
    tokens: int

    def __post_init__(self) -> None:
        check_digest(self.task_hash)
        if (
            not 0 <= finite(self.reward, "Reward") <= 1
            or type(self.tokens) is not int
            or self.tokens < 1
        ):
            raise TrainingError("Rollout reward or token count is invalid")
        finite(self.advantage, "Advantage")


@dataclass(frozen=True)
class GRPOResult:
    updates: int
    generated_tokens: int
    gradient_tokens: int
    uniform_groups: int
    mean_reward: float
    mean_loss: float
    validation_pass_rate: float
    validation_count: int
    training_seconds: float
    evaluation_seconds: float
    max_gradient_norm: float
    rollouts: tuple[Rollout, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.uniform_groups <= self.updates or self.updates < 1:
            raise TrainingError("GRPO update counts are invalid")
        if (
            not 0 < self.gradient_tokens < self.generated_tokens
            or self.validation_count < 1
        ):
            raise TrainingError("GRPO token or evaluation counts are invalid")
        for value in (self.mean_reward, self.validation_pass_rate):
            if not 0 <= finite(value, "Rate") <= 1:
                raise TrainingError("GRPO rates must be in [0,1]")
        finite(self.mean_loss, "Mean loss")
        for value in (
            self.training_seconds,
            self.evaluation_seconds,
            self.max_gradient_norm,
        ):
            if finite(value, "Measurement") < 0:
                raise TrainingError("GRPO times and norms must be nonnegative")
        object.__setattr__(self, "rollouts", tuple(self.rollouts))


def update_seed(seed: int, step: int) -> int:
    """Hash seed and update index; offset seeds would share keys across jobs."""
    digest = content_hash(
        {"domain": "openloop.grpo.update.v1", "seed": seed, "step": step}
    )
    return int(digest[:16], 16) % (2**32 - 8192)


def train_grpo(
    client: TrainingClient,
    reference: TrainingClient,
    tokenizer: Tokenizer,
    data: ArithmeticData,
    *,
    updates: int,
    sampling: SamplingParams,
    loss: LossConfig,
) -> GRPOResult:
    """One fresh group per update, one optimizer step, then fixed greedy validation.

    No held-out tasks enter this function. Validation never supplies gradients.
    Equal-reward groups have zero policy advantages, not fabricated learning signal.
    """
    if type(updates) is not int or updates < 1 or sampling.num_samples < 2:
        raise TrainingError("GRPO needs positive updates and groups of at least two")
    if sampling.temperature != 1:
        raise TrainingError("Training rollouts must sample the untempered policy")
    if loss.objective is not Objective.GRPO or client is reference:
        raise TrainingError(
            "GRPO needs its policy objective and a separate frozen reference"
        )
    generated = gradient_tokens = uniform = 0
    max_norm = 0.0
    rollouts: list[Rollout] = []
    rewards_seen, losses = [], []
    started = perf_counter()
    for step in range(updates):
        task = data.train[step % len(data.train)]
        prompt = tuple(tokenizer.encode(task.prompt))
        params = SamplingParams(
            sampling.max_tokens,
            sampling.num_samples,
            1.0,
            update_seed(sampling.seed, step),
            sampling.stop_tokens,
        )
        samples = client.sample(prompt, params)
        if len(samples) != params.num_samples:
            raise TrainingError("Backend returned the wrong rollout group size")
        rewards = tuple(
            reward(task, tokenizer.decode(list(sample.tokens))) for sample in samples
        )
        advantages = group_advantages(rewards)
        uniform += int(len(set(rewards)) == 1)
        batch = []
        for sample, advantage in zip(samples, advantages, strict=True):
            text = tokenizer.decode(list(sample.tokens))
            rollouts.append(
                Rollout(
                    content_hash(task),
                    text,
                    reward(task, text),
                    advantage,
                    len(sample.tokens),
                )
            )
            datum = Datum(
                (*prompt, *sample.tokens), len(prompt), advantage, sample.logprobs
            )
            references = reference.logprobs(datum) if loss.kl_beta else ()
            old_logprobs = client.logprobs(datum)
            batch.append(
                Datum(
                    datum.tokens,
                    datum.completion_start,
                    advantage,
                    old_logprobs,
                    references,
                )
            )
            generated += len(sample.tokens)
        result = client.forward_backward(tuple(batch), loss)
        update = client.optim_step()
        max_norm = max(max_norm, finite(update["gradient_norm"], "Gradient norm"))
        gradient_tokens += result.completion_tokens
        rewards_seen.extend(rewards)
        losses.append(result.loss)
    training_seconds = perf_counter() - started
    evaluation_started = perf_counter()
    passes = 0
    for index, task in enumerate(data.validation):
        sample = client.sample(
            tuple(tokenizer.encode(task.prompt)),
            SamplingParams(
                sampling.max_tokens, 1, 0.0, sampling.seed + index, sampling.stop_tokens
            ),
        )[0]
        generated += len(sample.tokens)
        passes += int(reward(task, tokenizer.decode(list(sample.tokens))))
    return GRPOResult(
        updates,
        generated,
        gradient_tokens,
        uniform,
        fmean(rewards_seen),
        fmean(losses),
        passes / len(data.validation),
        len(data.validation),
        training_seconds,
        perf_counter() - evaluation_started,
        max_norm,
        tuple(rollouts),
    )
