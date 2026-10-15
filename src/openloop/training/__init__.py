"""Local Tinker-shaped contracts. Backends are optional and loaded explicitly."""

from .contracts import (
    AdamParams,
    Checkpoint,
    Datum,
    ForwardBackwardResult,
    LossConfig,
    Objective,
    Sample,
    SamplingParams,
    TrainingClient,
    TrainingError,
)
from .loss import group_advantages

__all__ = [
    "AdamParams",
    "Checkpoint",
    "Datum",
    "ForwardBackwardResult",
    "LossConfig",
    "Objective",
    "Sample",
    "SamplingParams",
    "TrainingClient",
    "TrainingError",
    "group_advantages",
]
