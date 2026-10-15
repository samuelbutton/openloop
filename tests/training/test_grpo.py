"""GRPO controls groups and update counts; verifiable rewards never execute text."""

from pathlib import Path

import pytest

from openloop.training import (
    Checkpoint,
    Datum,
    ForwardBackwardResult,
    LossConfig,
    Sample,
    SamplingParams,
)
from openloop.training.grpo import ArithmeticData, ArithmeticTask, reward, train_grpo


class Tokenizer:
    def __init__(self, data: ArithmeticData) -> None:
        self.tasks = {task.prompt: task for task in (*data.train, *data.validation)}
        self.prompts = {text: index + 1 for index, text in enumerate(self.tasks)}
        self.outputs = {
            100 + self.prompts[text]: str(task.answer)
            for text, task in self.tasks.items()
        }
        self.outputs[999] = "9999"

    def encode(self, text: str) -> list[int]:
        return [self.prompts[text]]

    def decode(self, tokens: list[int]) -> str:
        return self.outputs[tokens[0]]


class Client:
    def __init__(self) -> None:
        self.updates = 0
        self.rows: list[Datum] = []
        self.sample_params: list[SamplingParams] = []

    def logprobs(self, datum: Datum) -> tuple[float, ...]:
        return (-0.8,) * datum.completion_tokens

    def sample(
        self, prompt: tuple[int, ...], params: SamplingParams
    ) -> tuple[Sample, ...]:
        self.sample_params.append(params)
        return tuple(
            Sample((100 + prompt[0] if i % 2 == 0 else 999,), (-0.7,))
            for i in range(params.num_samples)
        )

    def forward_backward(
        self, data: tuple[Datum, ...], config: LossConfig
    ) -> ForwardBackwardResult:
        config.validate(data)
        self.rows.extend(data)
        return ForwardBackwardResult(0.1, sum(row.completion_tokens for row in data))

    def optim_step(self) -> dict[str, float]:
        self.updates += 1
        return {"step": float(self.updates), "gradient_norm": 0.5}

    def save_state(self, path: Path) -> Checkpoint:
        raise AssertionError("The GRPO driver does not own checkpoint publication")

    def load_state(self, path: Path, sha256: str | None = None) -> None:
        raise AssertionError("The GRPO driver must not restore a different policy")


def test_groups_train_only_on_train_and_evaluate_once() -> None:
    data = ArithmeticData((ArithmeticTask(1, 2, "+"),), (ArithmeticTask(4, 3, "-"),))
    client, reference = Client(), Client()
    result = train_grpo(
        client,
        reference,
        Tokenizer(data),
        data,
        updates=2,
        sampling=SamplingParams(max_tokens=16, num_samples=4),
        loss=LossConfig(),
    )
    assert client.updates == 2
    assert len(client.rows) == 8
    assert {row.tokens[0] for row in client.rows} == {1}
    assert all(row.reference_logprobs == (-0.8,) for row in client.rows)
    assert [row.advantage for row in client.rows[:4]] == pytest.approx(
        [1, -1, 1, -1], abs=3e-6
    )
    assert [params.temperature for params in client.sample_params] == [1, 1, 0]
    assert result.generated_tokens == 9
    assert result.gradient_tokens == 8
    assert result.validation_pass_rate == 1


def test_offset_job_seeds_do_not_share_update_seeds() -> None:
    data = ArithmeticData((ArithmeticTask(1, 2, "+"),), (ArithmeticTask(4, 3, "-"),))

    def training_seeds(seed: int) -> set[int]:
        client = Client()
        train_grpo(
            client,
            Client(),
            Tokenizer(data),
            data,
            updates=4,
            sampling=SamplingParams(max_tokens=16, num_samples=4, seed=seed),
            loss=LossConfig(),
        )
        return {p.seed for p in client.sample_params if p.temperature == 1}

    first, second = training_seeds(42), training_seeds(58)
    assert len(first) == len(second) == 4
    assert not first & second


@pytest.mark.parametrize(
    "text",
    [
        "<answer>3</answer>",
        "3 + 0",
        "<answer>3</answer> extra",
        "<answer>1+2</answer>",
        "__import__('os')",
    ],
)
def test_reward_rejects_nonanswers(text: str) -> None:
    assert reward(ArithmeticTask(1, 2, "+"), text) == 0


def test_arithmetic_data_is_frozen_and_disjoint() -> None:
    data = ArithmeticData.default()
    assert data == ArithmeticData.default()
    assert len(data.train) == 64
    assert len(data.validation) == 16
    assert not set(data.train) & set(data.validation)
    assert reward(ArithmeticTask(1, 2, "+"), " 3\n") == 1
