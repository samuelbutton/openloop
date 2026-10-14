"""Typed research operators. Importing starts no models, processes, or training."""

from .feedback import bind_candidate, record_idea_failure
from .implementer import ImplementedCandidate, Implementer
from .models import (
    HypothesisCard,
    ImplementationProvider,
    ImplementationRequest,
    ModelReply,
    Operator,
    Patch,
    PredictedEffect,
    ProposalDraft,
    ProposalProvider,
    ProposalRequest,
)
from .outcomes import CalibrationSummary, calibration_summary, record_outcome
from .proposer import Proposer
from .spend import (
    CallBudget,
    Committed,
    Pricing,
    Reservation,
    SpendGate,
    SpendLimitError,
)

__all__ = [
    "CalibrationSummary",
    "CallBudget",
    "Committed",
    "HypothesisCard",
    "ImplementationProvider",
    "ImplementationRequest",
    "ImplementedCandidate",
    "Implementer",
    "ModelReply",
    "Operator",
    "Patch",
    "PredictedEffect",
    "Pricing",
    "ProposalDraft",
    "ProposalProvider",
    "ProposalRequest",
    "Proposer",
    "Reservation",
    "SpendGate",
    "SpendLimitError",
    "bind_candidate",
    "calibration_summary",
    "record_idea_failure",
    "record_outcome",
]
