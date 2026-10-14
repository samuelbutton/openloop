"""Pure plugin report records. Trusted plugins own their admission policy.

These records are derived reports, never evaluator scores or independent samples.
Ledger owns transactions; plugins cannot write SQLite or artifact storage.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .identity import content_hash
from .validation import JSONValue, check_digest, check_text, freeze_object


@dataclass(frozen=True)
class Derivation:
    """One immutable, content-addressed plugin report and its publication time."""

    namespace: str
    subject_hash: str
    payload: Mapping[str, JSONValue]
    timestamp: str

    def __post_init__(self) -> None:
        check_text(self.namespace)
        check_digest(self.subject_hash)
        check_text(self.timestamp)
        object.__setattr__(self, "payload", freeze_object(self.payload))

    @property
    def hash(self) -> str:
        return content_hash((self.namespace, self.subject_hash, self.payload))
