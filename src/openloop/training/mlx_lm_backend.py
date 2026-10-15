"""Use mlx-lm's quantized model, LoRA conversion, and cached generation unchanged."""

from importlib import import_module
from typing import Protocol

from openloop.ledger import content_hash
from openloop.ledger.validation import freeze_object

from .assets import ModelSnapshot
from .contracts import AdamParams, Sample, SamplingParams, TrainingError
from .mlx_backend import MLXBackend

LORA_LAYERS = 4
LORA_CONFIG = freeze_object(
    {
        "rank": 8,
        "scale": 1.0,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
)


class TemplateTokenizer(Protocol):
    eos_token_ids: set[int]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str: ...


class ChatTokenizer:
    """Use the frozen model's chat template; scoring strips special tokens only."""

    def __init__(self, tokenizer: TemplateTokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def stop_tokens(self) -> tuple[int, ...]:
        return tuple(sorted(self._tokenizer.eos_token_ids))

    def encode(self, text: str) -> list[int]:
        result = list(
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if not result or len(result) > 128:
            raise TrainingError("T2 prompt must fit 128 tokens")
        return result

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class MLXLMBackend(MLXBackend):
    """MLX backend with mlx-lm KV-cached rollouts for the native T2 recipe."""

    def sample(
        self, prompt: tuple[int, ...], params: SamplingParams
    ) -> tuple[Sample, ...]:
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise TrainingError("Sampling requires a nonempty token prompt")
        generate = import_module("mlx_lm.generate").generate_step
        mx = self._mx
        samples = []
        for index in range(params.num_samples):
            stream = generate(
                mx.array(prompt, dtype=mx.int32),
                self._model,
                max_tokens=params.max_tokens,
                sampler=self._sampler(params, index),
            )
            tokens, scores = [], []
            try:
                for token, logprobs in stream:
                    tokens.append(int(token))
                    scores.append(float(logprobs[token].item()))
                    if token in params.stop_tokens:
                        break
            finally:
                stream.close()
            samples.append(Sample(tuple(tokens), tuple(scores)))
        return tuple(samples)


def load_mlx_clients(
    snapshot: ModelSnapshot,
    *,
    params: AdamParams,
    seed: int,
) -> tuple[MLXBackend, MLXBackend, ChatTokenizer]:
    """Load local base/reference models; add rank-8 Q/V LoRA to four actor layers."""
    snapshot.verify()
    utils = import_module("mlx_lm.utils")
    mx = import_module("mlx.core")
    actor, tokenizer = utils.load(
        str(snapshot.root),
        tokenizer_config={"local_files_only": True},
        trust_remote_code=False,
    )
    reference_model, _ = utils.load(
        str(snapshot.root),
        tokenizer_config={"local_files_only": True},
        trust_remote_code=False,
    )
    actor.freeze()
    reference_model.freeze()
    mx.random.seed(seed)
    import_module("mlx_lm.tuner.utils").linear_to_lora_layers(
        actor,
        LORA_LAYERS,
        LORA_CONFIG,
    )
    mx.eval(actor.parameters(), reference_model.parameters())
    if not import_module("mlx.utils").tree_flatten(actor.trainable_parameters()):
        raise TrainingError("LoRA conversion produced no trainable parameters")
    return (
        MLXLMBackend(
            actor,
            model_hash=content_hash(
                {
                    "base": snapshot.hash,
                    "lora_layers": LORA_LAYERS,
                    "lora": LORA_CONFIG,
                }
            ),
            params=params,
        ),
        MLXBackend(reference_model, model_hash=snapshot.hash, params=params),
        ChatTokenizer(tokenizer),
    )
