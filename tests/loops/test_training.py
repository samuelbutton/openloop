"""Exercise actual adapted loop control with a CPU stand-in, independent of clocks."""

from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.loops import ContractError
from openloop.loops.training import baseline_config, training_program

from .conftest import TOY_TRAIN


def execute(tokens: int, seed: int, dt: float) -> dict[str, object]:
    class Tokenizer:
        @classmethod
        def from_directory(cls, tokenizer_dir: str) -> SimpleNamespace:
            return SimpleNamespace()

    program = training_program(
        TOY_TRAIN,
        baseline_config(TOY_TRAIN),
        seed=seed,
        token_budget=tokens,
        tokenizer_dir="toy tokenizer",
    )
    namespace: dict[str, object] = {"Tokenizer": Tokenizer, "dt": dt}
    exec(compile(program, "adapted.py", "exec"), namespace)
    return namespace


@given(
    steps=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_token_stop_and_schedule_ignore_machine_speed(steps: int, seed: int) -> None:
    tokens = steps * 65536
    fast = execute(tokens, seed, 0.01)
    slow = execute(tokens, seed, 100.0)
    assert fast["step"] == slow["step"] == steps
    assert fast["total_tokens"] == slow["total_tokens"] == tokens
    assert (
        fast["progresses"]
        == slow["progresses"]
        == [index / steps for index in range(steps)]
    )
    assert fast["STARTUP_EXCLUDE_STEPS"] == 0
    assert fast["seed_values"] == [seed]
    assert fast["total_training_time"] == pytest.approx(steps * 0.01)


def test_source_structure_change_is_rejected() -> None:
    source = TOY_TRAIN.replace("STARTUP_EXCLUDE_STEPS = 1", "OTHER_STARTUP = 1")
    with pytest.raises(ContractError, match="supported token adapter"):
        training_program(
            source,
            baseline_config(source),
            seed=42,
            token_budget=65536,
            tokenizer_dir="toy",
        )


def test_partial_batch_budget_is_rejected() -> None:
    with pytest.raises(ContractError, match="exact number"):
        training_program(
            TOY_TRAIN,
            baseline_config(TOY_TRAIN),
            seed=42,
            token_budget=65537,
            tokenizer_dir="toy",
        )
