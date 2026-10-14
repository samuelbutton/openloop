"""Campaign transition kinds and the sink a durable journal implements.

No I/O here. A sink is coordinator-side storage; the decider's decision
functions never see it.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol as TypingProtocol

from .models import Protocol
from .statistics import Calibration


class TransitionKind(StrEnum):
    """Every campaign state change a journal records, in recorded order."""

    CREATED = "created"
    DECISION = "decision"
    EXTERNAL_SCREEN = "external_screen"
    CONFIRMATION_SET = "confirmation_set"
    SELECTION_CLOSED = "selection_closed"
    FINAL_STARTED = "final_started"
    FINAL = "final"


class TransitionSink(TypingProtocol):
    """Durable, ordered record of campaign transitions."""

    def open(
        self, protocol: Protocol, screen_noise: Calibration, confirm_noise: Calibration
    ) -> None:
        """Register the protocol and start an empty journal, or refuse."""
        ...

    def append(
        self,
        protocol_hash: str,
        kind: TransitionKind,
        payload: Mapping[str, object],
    ) -> None:
        """Record one transition durably before the campaign applies it."""
        ...
