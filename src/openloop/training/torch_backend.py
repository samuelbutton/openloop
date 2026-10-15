"""Thin PyTorch autograd/AdamW adapter. Importing does not load models or start jobs."""

from collections.abc import Iterable, Mapping
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


class TorchParameter(Protocol):
    @property
    def grad(self) -> object | None: ...


class TorchModel(Protocol):
    def __call__(self, tokens: object) -> object: ...
    def eval(self) -> object: ...
    def to(self, device: str) -> object: ...
    def parameters(self) -> Iterable[TorchParameter]: ...
    def named_parameters(self) -> Iterable[tuple[str, object]]: ...
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(
        self, values: Mapping[str, object], *, strict: bool
    ) -> object: ...


class TorchBackend:
    """Single-owner causal model. Mean sequence loss; calls accumulate gradient sums."""

    def __init__(
        self,
        model: TorchModel,
        *,
        model_hash: str,
        params: AdamParams | None = None,
        device: str = "cpu",
    ) -> None:
        check_digest(model_hash)
        params = AdamParams() if params is None else params
        if device not in ("cpu", "mps", "cuda"):
            raise TrainingError("Unsupported Torch device")
        self._torch = import_module("torch")
        self._model = model
        model.to(device)
        model.eval()
        self._params, self._model_hash, self._device = params, model_hash, device
        self._optimizer = self._torch.optim.AdamW(
            model.parameters(),
            lr=params.learning_rate,
            betas=(params.beta1, params.beta2),
            eps=params.epsilon,
            weight_decay=params.weight_decay,
        )
        self._pending = 0
        self._step = 0

    def _logprobs(self, row: Datum):
        torch = self._torch
        tokens = torch.tensor([row.tokens], dtype=torch.long, device=self._device)
        output = self._model(tokens[:, :-1])
        logits = torch.as_tensor(getattr(output, "logits", output)).float()
        if logits.ndim != 3 or logits.shape[:2] != tokens[:, :-1].shape:
            raise TrainingError(
                "Model must return [batch, sequence, vocabulary] logits"
            )
        scores = torch.log_softmax(logits[0, row.completion_start - 1 :], dim=-1)
        return scores.gather(-1, tokens[0, row.completion_start :, None]).squeeze(-1)

    def logprobs(self, datum: Datum) -> tuple[float, ...]:
        with self._torch.no_grad():
            values = self._logprobs(datum).cpu().tolist()
        return tuple(finite(value, "Logprob") for value in values)

    def forward_backward(
        self,
        data: tuple[Datum, ...],
        config: LossConfig,
    ) -> ForwardBackwardResult:
        config.validate(data)
        torch = self._torch
        losses = []
        for row in data:
            logp = self._logprobs(row)
            if config.objective is Objective.CROSS_ENTROPY:
                loss = -logp.mean()
            else:
                old = torch.tensor(row.old_logprobs, device=self._device)
                ratio = (logp - old).exp()
                clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
                policy = torch.minimum(ratio * row.advantage, clipped * row.advantage)
                if config.kl_beta:
                    delta = (
                        torch.tensor(row.reference_logprobs, device=self._device) - logp
                    )
                    kl = delta.exp() - delta - 1
                else:
                    kl = torch.zeros_like(logp)
                loss = (-policy + config.kl_beta * kl).mean()
            losses.append(loss)
        loss = torch.stack(losses).mean()
        value = finite(loss.detach().item(), "Loss")
        loss.backward()
        if any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all().item()
            for parameter in self._model.parameters()
        ):
            self._optimizer.zero_grad(set_to_none=True)
            self._pending = 0
            raise TrainingError("Nonfinite Torch gradient; pending gradients cleared")
        self._pending += 1
        return ForwardBackwardResult(value, sum(row.completion_tokens for row in data))

    def optim_step(self) -> Mapping[str, float]:
        if not self._pending:
            raise TrainingError("optim_step needs accumulated gradients")
        norm = self._torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self._params.max_grad_norm,
            error_if_nonfinite=True,
        )
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._pending = 0
        self._step += 1
        return {"step": float(self._step), "gradient_norm": float(norm.item())}

    def sample(
        self, prompt: tuple[int, ...], params: SamplingParams
    ) -> tuple[Sample, ...]:
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise TrainingError("Sampling requires a nonempty token prompt")
        torch = self._torch
        samples = []
        with torch.no_grad():
            for index in range(params.num_samples):
                generator = torch.Generator(device="cpu").manual_seed(
                    sampling_seed(params.seed, index)
                )
                completion, scores = [], []
                for _ in range(params.max_tokens):
                    tokens = torch.tensor([(*prompt, *completion)], device=self._device)
                    output = self._model(tokens)
                    logits = torch.as_tensor(getattr(output, "logits", output)).float()[
                        0, -1
                    ]
                    logp = torch.log_softmax(logits, dim=-1)
                    token = (
                        int(logp.argmax().item())
                        if params.temperature == 0
                        else int(
                            torch.multinomial(
                                logp.exp().cpu(), 1, generator=generator
                            ).item()
                        )
                    )
                    completion.append(token)
                    scores.append(float(logp[token].item()))
                    if token in params.stop_tokens:
                        break
                samples.append(Sample(tuple(completion), tuple(scores)))
        return tuple(samples)

    def save_state(self, path: Path) -> Checkpoint:
        if self._pending:
            raise TrainingError("Checkpoint only at an optimizer boundary")

        def write(directory: Path) -> None:
            self._torch.save(
                {
                    "model": self._model.state_dict(),
                    "optimizer": self._optimizer.state_dict(),
                },
                directory / "state.pt",
            )

        return save_checkpoint(
            path,
            backend="torch",
            model_hash=self._model_hash,
            params=self._params,
            step=self._step,
            writer=write,
        )

    def load_state(self, path: Path, sha256: str | None = None) -> None:
        if self._pending:
            raise TrainingError("Restore only at an optimizer boundary")
        step = verify_checkpoint(
            path,
            backend="torch",
            model_hash=self._model_hash,
            params=self._params,
            sha256=sha256,
        )
        state = self._torch.load(
            path / "state.pt", map_location=self._device, weights_only=True
        )
        self._model.load_state_dict(state["model"], strict=True)
        self._optimizer.load_state_dict(state["optimizer"])
        self._step = step
