"""Tiny CPU causal models; optional backends never download pretrained weights."""

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from openloop.training import AdamParams, TrainingClient

WEIGHTS = np.arange(25, dtype=np.float32).reshape(5, 5) / 50


@pytest.fixture(params=["torch", "mlx"])
def client_factory(
    request,
) -> Iterator[Callable[[], tuple[TrainingClient, Callable[[], np.ndarray]]]]:
    params = AdamParams(learning_rate=0.01, max_grad_norm=10)
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from openloop.training.torch_backend import TorchBackend

        class TorchModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 5)
                self.embedding.weight.data.copy_(torch.tensor(WEIGHTS))

            def forward(self, tokens):
                return self.embedding(tokens)

        def torch_factory():
            model = TorchModel()
            client = TorchBackend(model, model_hash="a" * 64, params=params)
            return client, lambda: model.embedding.weight.detach().cpu().numpy().copy()

        yield torch_factory
    else:
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from openloop.training.mlx_backend import MLXBackend

        previous = mx.default_device()
        mx.set_default_device(mx.cpu)

        class MLXModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(5, 5)
                self.embedding.weight = mx.array(WEIGHTS)

            def __call__(self, tokens):
                return self.embedding(tokens)

        def mlx_factory():
            model = MLXModel()
            client = MLXBackend(model, model_hash="a" * 64, params=params)
            return client, lambda: np.array(model.embedding.weight)

        try:
            yield mlx_factory
        finally:
            mx.set_default_device(previous)
