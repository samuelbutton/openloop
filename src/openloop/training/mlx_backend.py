"""Thin MLX autograd/AdamW adapter; MLX imports and model loading are explicit."""

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol

from openloop.ledger.validation import check_digest

from .checkpoints import save_checkpoint, verify_checkpoint
from .contracts import (
    AdamParams,
    Checkpoint,
    Datum,
    ForwardBackwardResult,
    LossConfig,
    Objective,
    Sample,
    SamplingParams,
    TrainingError,
    finite,
    sampling_seed,
)


class MLXModel(Protocol):
    def __call__(self, tokens: object) -> object: ...
    def eval(self) -> object: ...
    def parameters(self) -> dict: ...
    def trainable_parameters(self) -> dict: ...
    def update(self, values: dict) -> object: ...


class MLXBackend:
    """Single-owner causal model with accumulated sequence-mean gradients."""

    def __init__(
        self,
        model: MLXModel,
        *,
        model_hash: str,
        params: AdamParams | None = None,
    ) -> None:
        check_digest(model_hash)
        params = AdamParams() if params is None else params
        self._mx = import_module("mlx.core")
        self._nn = import_module("mlx.nn")
        self._utils = import_module("mlx.utils")
        self._model, self._params, self._model_hash = model, params, model_hash
        model.eval()
        self._optimizer = import_module("mlx.optimizers").AdamW(
            learning_rate=params.learning_rate,
            betas=[params.beta1, params.beta2],
            eps=params.epsilon,
            weight_decay=params.weight_decay,
            bias_correction=True,
        )
        self._gradients: dict | None = None
        self._step = 0

    def _logprobs(self, model: MLXModel, row: Datum):
        mx = self._mx
        tokens = mx.array([row.tokens], dtype=mx.int32)
        logits = mx.array(model(tokens[:, :-1])).astype(mx.float32)
        if logits.ndim != 3 or logits.shape[:2] != tokens[:, :-1].shape:
            raise TrainingError(
                "Model must return [batch, sequence, vocabulary] logits"
            )
        scores = logits[0, row.completion_start - 1 :]
        scores = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        return mx.take_along_axis(
            scores, tokens[0, row.completion_start :, None], axis=-1
        ).squeeze(-1)

    def logprobs(self, datum: Datum) -> tuple[float, ...]:
        values = self._logprobs(self._model, datum)
        self._mx.eval(values)
        return tuple(finite(value, "Logprob") for value in values.tolist())

    def forward_backward(
        self,
        data: tuple[Datum, ...],
        config: LossConfig,
    ) -> ForwardBackwardResult:
        config.validate(data)
        mx = self._mx

        def loss_fn(model: MLXModel):
            losses = []
            for row in data:
                logp = self._logprobs(model, row)
                if config.objective is Objective.CROSS_ENTROPY:
                    loss = -mx.mean(logp)
                else:
                    ratio = mx.exp(logp - mx.array(row.old_logprobs))
                    clipped = mx.clip(
                        ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon
                    )
                    policy = mx.minimum(ratio * row.advantage, clipped * row.advantage)
                    if config.kl_beta:
                        delta = mx.array(row.reference_logprobs) - logp
                        kl = mx.exp(delta) - delta - 1
                    else:
                        kl = mx.zeros_like(logp)
                    loss = mx.mean(-policy + config.kl_beta * kl)
                losses.append(loss)
            return mx.mean(mx.stack(losses))

        loss, gradients = self._nn.value_and_grad(self._model, loss_fn)(self._model)
        mx.eval(loss, gradients)
        value = finite(loss.item(), "Loss")
        if self._gradients is not None:
            gradients = self._utils.tree_map(
                lambda a, b: a + b, self._gradients, gradients
            )
        if not all(
            mx.all(mx.isfinite(g)).item()
            for _, g in self._utils.tree_flatten(gradients)
        ):
            self._gradients = None
            raise TrainingError("Nonfinite MLX gradient; pending gradients cleared")
        self._gradients = gradients
        return ForwardBackwardResult(value, sum(row.completion_tokens for row in data))

    def optim_step(self) -> Mapping[str, float]:
        if self._gradients is None:
            raise TrainingError("optim_step needs accumulated gradients")
        mx = self._mx
        norm = mx.sqrt(
            sum(
                mx.sum(g.astype(mx.float32) ** 2)
                for _, g in self._utils.tree_flatten(self._gradients)
            )
        )
        mx.eval(norm)
        value = finite(norm.item(), "Gradient norm")
        scale = min(1.0, self._params.max_grad_norm / (value + 1e-6))
        gradients = self._utils.tree_map(
            lambda gradient: gradient * scale, self._gradients
        )
        self._optimizer.update(self._model, gradients)
        mx.eval(self._model.parameters(), self._optimizer.state)
        self._gradients = None
        self._step += 1
        return {"step": float(self._step), "gradient_norm": value}

    def sample(
        self, prompt: tuple[int, ...], params: SamplingParams
    ) -> tuple[Sample, ...]:
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise TrainingError("Sampling requires a nonempty token prompt")
        mx = self._mx
        samples = []
        for index in range(params.num_samples):
            sampler = self._sampler(params, index)
            completion, scores = [], []
            for _ in range(params.max_tokens):
                logits = mx.array(
                    self._model(mx.array([(*prompt, *completion)], dtype=mx.int32))
                ).astype(mx.float32)[0, -1]
                logp = logits - mx.logsumexp(logits)
                token = int(sampler(logp).item())
                completion.append(token)
                scores.append(float(logp[token].item()))
                if token in params.stop_tokens:
                    break
            samples.append(Sample(tuple(completion), tuple(scores)))
        return tuple(samples)

    def _sampler(self, params: SamplingParams, index: int):
        """Split request, sample, and token keys; never offset seeds along a stream."""
        mx = self._mx
        key = mx.random.key(sampling_seed(params.seed, index))

        def choose(logprobs: object):
            nonlocal key
            key, draw_key = mx.random.split(key)
            values = mx.array(logprobs)
            if params.temperature == 0:
                return mx.argmax(values, axis=-1)
            return mx.random.categorical(values, key=draw_key)

        return choose

    def save_state(self, path: Path) -> Checkpoint:
        if self._gradients is not None:
            raise TrainingError("Checkpoint only at an optimizer boundary")

        def write(directory: Path) -> None:
            arrays = {
                "model." + name: value
                for name, value in self._utils.tree_flatten(
                    self._model.trainable_parameters()
                )
            }
            arrays.update(
                {
                    "optimizer." + name: self._mx.array(value)
                    for name, value in self._utils.tree_flatten(self._optimizer.state)
                }
            )
            self._mx.save_safetensors(str(directory / "state.safetensors"), arrays)

        return save_checkpoint(
            path,
            backend="mlx",
            model_hash=self._model_hash,
            params=self._params,
            step=self._step,
            writer=write,
        )

    def load_state(self, path: Path, sha256: str | None = None) -> None:
        if self._gradients is not None:
            raise TrainingError("Restore only at an optimizer boundary")
        step = verify_checkpoint(
            path,
            backend="mlx",
            model_hash=self._model_hash,
            params=self._params,
            sha256=sha256,
        )
        arrays = self._mx.load(str(path / "state.safetensors"))
        parameters = self._utils.tree_unflatten(
            [
                (name.removeprefix("model."), value)
                for name, value in arrays.items()
                if name.startswith("model.")
            ]
        )
        expected = self._utils.tree_flatten(self._model.trainable_parameters())
        observed = self._utils.tree_flatten(parameters)
        if sorted((n, v.shape) for n, v in expected) != sorted(
            (n, v.shape) for n, v in observed
        ):
            raise TrainingError("Checkpoint trainable parameter shapes differ")
        self._model.update(parameters)
        self._optimizer.state = self._utils.tree_unflatten(
            [
                (name.removeprefix("optimizer."), value)
                for name, value in arrays.items()
                if name.startswith("optimizer.")
            ]
        )
        self._mx.eval(self._model.parameters(), self._optimizer.state)
        self._step = step
