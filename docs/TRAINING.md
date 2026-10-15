# Local training interface

Implemented on October 15, 2026. Supports N4 and N9 in [NEEDS.md](../NEEDS.md).
The [contracts](../src/openloop/training/contracts.py) use the operation names in
[Tinker's training API](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/).
These are synchronous local operations. They are not wire-compatible with the
Tinker SDK and do not call its service. No API key is needed.

| Operation | Local contract |
| --- | --- |
| `sample(prompt, params)` | Generate a bounded group from the current policy. Return token IDs and natural-log policy probabilities, including a generated stop token. |
| `forward_backward(data, config)` | Accumulate gradients without changing parameters. Return the loss and number of completion targets. |
| `optim_step()` | Clip the accumulated global gradient norm, apply AdamW, and clear gradients. Refuse a step without gradients. |
| `save_state(path)` | Publish a new checkpoint directory at an optimizer boundary. Include optimizer moments and step count. Refuse overwrite. |
| `load_state(path)` | Verify backend, model identity, optimizer settings, and payload hashes before loading. |
| `logprobs(datum)` | Score completion targets under the unchanged current policy without gradients. |

The caller owns one client and serializes its operations. There are no request
futures or concurrent updates. Importing the contracts loads no training library.

## Loss and sampling semantics

`Datum.tokens` contains the complete prompt and completion. `completion_start`
is the first completion token's index. It must follow at least one prompt token.
The model predicts token t from tokens before t. Loss excludes prompt positions.
Each sequence contributes its mean completion loss; sequences then receive equal
weight. Several `forward_backward` calls add their mean-batch gradients. They
do not implicitly average accumulated batches again.

Cross-entropy and clipped GRPO are supported. For each completion token, GRPO
minimizes the negative clipped policy surrogate plus beta times
`exp(reference_logp - policy_logp) - (reference_logp - policy_logp) - 1`.
The pure [scalar reference](../src/openloop/training/loss.py) is independent of
both numerical implementations. This follows the
[GRPO objective](https://arxiv.org/abs/2402.03300), with explicit local choices:
population reward SD plus 1e-6, equal sequence weighting, and one update per fresh
rollout group. It is not a reproduction of the full DeepSeekMath recipe.

Training sampling uses temperature 1. Greedy evaluation uses temperature 0.
Other temperatures are rejected to keep behavior-policy probabilities explicit.
Generation is capped at 256 tokens and 16 samples per request. Each request
has its own seed; both backends namespace sample seeds separately from model
initialization, and MLX splits token keys. Generation does
not depend on a hidden sampling RNG stream. A regression rejects shifted-stream
reuse across successive rollout groups.
Dropout stays disabled during scoring and gradient computation. Model-specific
initialization is separate from request sampling.

## Backends and checkpoint boundaries

[MLXBackend](../src/openloop/training/mlx_backend.py) wraps MLX autograd and
AdamW with bias correction. The [mlx-lm adapter](../src/openloop/training/mlx_lm_backend.py)
reuses its quantized models, LoRA conversion, and cached generation.
[TorchBackend](../src/openloop/training/torch_backend.py) wraps a supplied causal
PyTorch module and AdamW. It accepts CPU, MPS, and CUDA device names. CUDA has not
been checked in this task. Larger PyTorch/Transformers model loaders and LoRA
setup remain the caller's responsibility; the native T2 factory uses MLX.

```python
from openloop.training import AdamParams, Datum, LossConfig, Objective
from openloop.training.torch_backend import TorchBackend

client = TorchBackend(model, model_hash=frozen_model_hash, params=AdamParams())
row = Datum(tokens=(0, 1, 2, 3), completion_start=2)
result = client.forward_backward((row,), LossConfig(Objective.CROSS_ENTROPY))
update = client.optim_step()
checkpoint = client.save_state(checkpoint_path)
client.load_state(checkpoint.path)
```

The model must return `[batch, sequence, vocabulary]` logits. Transformers-style
Torch outputs with a `logits` field are also accepted. The host-supplied model
hash must bind all frozen base weights and model behavior. The native MLX factory
binds its LoRA layout and scale as well as the base snapshot.

Checkpoint publication uses a temporary directory and an exclusive writer lock,
then renames the complete directory. A killed writer can leave a lock that the
host must reconcile. This is separate from ledger recovery. MLX stores trainable
parameters and optimizer arrays in safetensors; the frozen base must already be
loaded. PyTorch stores its model and optimizer with `torch.save` and loads with
`weights_only=True`. Checkpoints require the same backend and optimizer settings.
They are not a format for MLX-to-PyTorch weight conversion. No pending gradients
may exist when saving or restoring. This boundary restriction differs from
Tinker's general training-state API.

## Checks on the Mac

Install optional PyTorch support with:

```sh
uv sync --locked --group training-torch
OPENLOOP_TEST_MPS=1 uv run --group training-torch pytest tests/training/test_backends.py
```

The [recorded correctness check](../research/t2/backend-correctness-namespaced-2026-10-15.json)
contains 15 passing tests on the local Mac, with source hashes. Tests use tiny causal models and
float32 arithmetic. Both CPU backends are checked against an independent scalar
loss and finite-difference gradients, including clipped GRPO and prompt masking.
They check the first AdamW update, accumulation, optimizer-boundary rules,
checkpoint continuation, corruption rejection, sampling seeds, and EOS alignment.
The explicit PyTorch MPS check covers log probabilities and a gradient update.
MLX CPU and PyTorch MPS checks do not establish parity of full quantized models.

Torch tests skip if its optional dependency is absent. MLX tests skip off Apple
silicon. The default suite does not opt into MPS. Portable contract, reward,
orchestration, and scalar checks remain offline and use no downloaded model.
Native Qwen training is a separate engineering smoke in [T2.md](T2.md).
