"""Scalar GRPO reference and group advantages; no model or framework dependency."""

import math
from statistics import fmean

from .contracts import Datum, LossConfig, Objective, TrainingError, finite


def group_advantages(
    rewards: tuple[float, ...], *, epsilon: float = 1e-6
) -> tuple[float, ...]:
    """Use population SD within one prompt group; ties give zero advantages."""
    if len(rewards) < 2 or finite(epsilon, "Advantage epsilon") <= 0:
        raise TrainingError("GRPO needs at least two rewards and positive epsilon")
    values = tuple(finite(reward, "Reward") for reward in rewards)
    mean = fmean(values)
    std = math.sqrt(fmean((reward - mean) ** 2 for reward in values))
    return tuple((reward - mean) / (std + epsilon) for reward in values)


def reference_loss(
    logprobs: tuple[tuple[float, ...], ...], data: tuple[Datum, ...], config: LossConfig
) -> float:
    """Independent scalar oracle: per-sequence token mean, then equal sequence mean."""
    config.validate(data)
    losses = []
    for current, row in zip(logprobs, data, strict=True):
        if len(current) != row.completion_tokens:
            raise TrainingError("Scored logprobs differ from completion length")
        tokens = []
        for index, logp in enumerate(current):
            finite(logp, "Current logprob")
            if config.objective is Objective.CROSS_ENTROPY:
                tokens.append(-logp)
                continue
            ratio = math.exp(logp - row.old_logprobs[index])
            clipped = min(1 + config.clip_epsilon, max(1 - config.clip_epsilon, ratio))
            policy = min(ratio * row.advantage, clipped * row.advantage)
            delta = row.reference_logprobs[index] - logp if config.kl_beta else 0.0
            kl = math.exp(delta) - delta - 1
            tokens.append(-policy + config.kl_beta * kl)
        losses.append(fmean(tokens))
    return finite(fmean(losses), "Reference loss")
