"""Independent finite differences, optimizer/state, and causal-mask checks."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from openloop.training import (
    Datum,
    LossConfig,
    Objective,
    SamplingParams,
    TrainingError,
)
from openloop.training.loss import reference_loss

from .conftest import WEIGHTS


def numpy_logprobs(weights, row):
    selected = weights[np.array(row.tokens[row.completion_start - 1 : -1])].astype(
        float
    )
    logp = selected - np.logaddexp.reduce(selected, axis=-1, keepdims=True)
    return tuple(logp[np.arange(len(selected)), row.tokens[row.completion_start :]])


@pytest.mark.parametrize("objective", list(Objective))
def test_loss_and_first_adam_step_match_finite_differences(
    client_factory, objective
) -> None:
    client, weights = client_factory()
    data = (
        Datum((0, 1, 2, 3), 2, 1.2, (-1.7, -1.4), (-1.6, -1.8)),
        Datum((4, 2), 1, -0.7, (-1.0,), (-1.5,)),
    )
    config = LossConfig(objective)

    def score(w):
        return reference_loss(
            tuple(numpy_logprobs(w, row) for row in data), data, config
        )

    gradient = np.zeros_like(WEIGHTS, dtype=float)
    for index in np.ndindex(*WEIGHTS.shape):
        plus, minus = WEIGHTS.astype(float).copy(), WEIGHTS.astype(float).copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        gradient[index] = (score(plus) - score(minus)) / 2e-5
    result = client.forward_backward(data, config)
    assert result.loss == pytest.approx(score(WEIGHTS), abs=1e-6)
    assert result.completion_tokens == 3
    before = weights()
    step = client.optim_step()
    expected = before - 0.01 * gradient / (np.abs(gradient) + 1e-8)
    np.testing.assert_allclose(weights(), expected, atol=2e-6)
    np.testing.assert_array_equal(weights()[0], before[0])
    assert step["step"] == 1
    assert step["gradient_norm"] == pytest.approx(np.linalg.norm(gradient), abs=1e-6)


def test_accumulation_and_optimizer_boundary(client_factory) -> None:
    first, _ = client_factory()
    second, _ = client_factory()
    rows = (Datum((0, 1, 2), 1),)
    config = LossConfig(Objective.CROSS_ENTROPY)
    first.forward_backward(rows, config)
    first_norm = first.optim_step()["gradient_norm"]
    second.forward_backward(rows, config)
    second.forward_backward(rows, config)
    assert second.optim_step()["gradient_norm"] == pytest.approx(
        2 * first_norm, abs=1e-6
    )
    with pytest.raises(TrainingError, match="accumulated"):
        second.optim_step()


def test_checkpoint_restores_optimizer_for_next_update(
    client_factory, tmp_path: Path
) -> None:
    client, weights = client_factory()
    rows = (Datum((0, 2, 3), 1),)
    config = LossConfig(Objective.CROSS_ENTROPY)
    client.forward_backward(rows, config)
    with pytest.raises(TrainingError, match="boundary"):
        client.save_state(tmp_path / "pending")
    client.optim_step()
    checkpoint = client.save_state(tmp_path / "state")
    assert checkpoint.path.is_dir()
    client.forward_backward(rows, config)
    client.optim_step()
    expected = weights()
    client.load_state(checkpoint.path)
    client.forward_backward(rows, config)
    assert client.optim_step()["step"] == 2
    np.testing.assert_allclose(weights(), expected, atol=1e-7)
    with pytest.raises(TrainingError, match="already exists"):
        client.save_state(checkpoint.path)


def test_greedy_stop_and_sample_logprob_alignment(client_factory) -> None:
    client, _ = client_factory()
    samples = client.sample(
        (0, 1),
        SamplingParams(max_tokens=4, num_samples=2, temperature=0, stop_tokens=(4,)),
    )
    assert len(samples) == 2
    for sample in samples:
        assert sample.tokens == (4,)
        assert sample.logprobs == pytest.approx(
            client.logprobs(Datum((0, 1, 4), 2)), abs=1e-6
        )
    params = SamplingParams(max_tokens=4, num_samples=2, seed=23)
    assert client.sample((1,), params) == client.sample((1,), params)


def test_bad_batches_and_corrupt_checkpoints_fail_before_update(
    client_factory, tmp_path: Path
) -> None:
    client, weights = client_factory()
    before = weights()
    with pytest.raises(TrainingError, match="old"):
        client.forward_backward((Datum((0, 1), 1),), LossConfig())
    np.testing.assert_array_equal(weights(), before)
    checkpoint = client.save_state(tmp_path / "state")
    state = next(
        path for path in checkpoint.path.iterdir() if path.name != "manifest.json"
    )
    with state.open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(TrainingError, match="hash mismatch"):
        client.load_state(checkpoint.path)


def test_fresh_group_seed_does_not_reuse_a_shifted_token_stream(client_factory) -> None:
    client, _ = client_factory()
    # Every row of this fixture has the same categorical distribution. Fresh
    # groups must not reuse the preceding group's token-by-token random keys.
    first = client.sample((1,), SamplingParams(max_tokens=16, num_samples=1, seed=42))[
        0
    ]
    second = client.sample((1,), SamplingParams(max_tokens=16, num_samples=1, seed=58))[
        0
    ]
    assert first.tokens[1:] != second.tokens[:-1]


@pytest.mark.skipif(
    os.environ.get("OPENLOOP_TEST_MPS") != "1", reason="Explicit Metal check only"
)
def test_torch_mps_matches_cpu_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal/MPS is unavailable")
    from openloop.training.torch_backend import TorchBackend

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 5)
            self.embedding.weight.data.copy_(torch.tensor(WEIGHTS))

        def forward(self, tokens):
            return self.embedding(tokens)

    row = Datum((0, 1, 2), 1)
    client = TorchBackend(Model(), model_hash="a" * 64, device="mps")
    assert client.logprobs(row) == pytest.approx(numpy_logprobs(WEIGHTS, row), abs=1e-6)
    result = client.forward_backward((row,), LossConfig(Objective.CROSS_ENTROPY))
    assert math.isfinite(result.loss)
    assert client.optim_step()["gradient_norm"] > 0
