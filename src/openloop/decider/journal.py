"""Durable campaign journal over the ledger derivation API (coordinator-side I/O).

Boundary: a trusted coordinator owns one journal per protocol hash. The ledger
supplies record order and times. Nothing here proves that the coordinator
declared its protocol honestly; the journal fixes the order of ledger records
and lets a new process rebuild a campaign by re-deciding every recorded
transition from `Ledger.get` scores. Recorded statistics are never trusted.

Two namespaces, both keyed by protocol hash:
- `campaign_protocol.v0`: the frozen protocol. Identical content keeps its first
  record, so registering again never moves the registration time later.
- `campaign_journal.v0`: a hash chain. Entry 0 binds the registration record and
  the calibration run references. Each entry names its index and predecessor.
  A gap, edit, reorder, or fork breaks the chain and restore rejects it.
"""

from collections.abc import Mapping
from datetime import datetime

from openloop.ledger import (
    Derivation,
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    NotFoundError,
)
from openloop.ledger.validation import check_digest, plain

from .campaign import Campaign
from .models import Protocol
from .replay import Replay, protocol_from_record
from .statistics import Calibration
from .transitions import TransitionKind

PROTOCOL_NAMESPACE = "campaign_protocol.v0"
JOURNAL_NAMESPACE = "campaign_journal.v0"


def _noise_record(noise: Calibration) -> Mapping[str, object]:
    return {
        "kind": noise.kind,
        "metric_name": noise.samples[0].metric_name,
        "pairs": [[left.reference, right.reference] for left, right in noise.pairs],
    }


class CampaignJournal:
    """Record and rebuild one campaign per protocol hash on a trusted ledger.

    Appends read the whole chain first, so one coordinator at a time must own a
    journal. Two writers can fork it; a fork fails the chain check on restore.
    """

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def register(self, protocol: Protocol) -> Derivation:
        """Record the frozen protocol. Repeating it returns the first record.

        Call this before running any baseline or candidate run to make the
        registration earlier than every calibration run too.
        """
        return self.ledger.record_derivation(
            PROTOCOL_NAMESPACE,
            protocol.hash,
            {"protocol_hash": protocol.hash, "protocol": protocol},
        )

    def _registered(self, protocol_hash: str) -> tuple[Derivation, Protocol]:
        check_digest(protocol_hash)
        for record in self.ledger.derivations(
            PROTOCOL_NAMESPACE, subject_hash=protocol_hash
        ):
            try:
                protocol = protocol_from_record(plain(record.payload["protocol"]))
            except (KeyError, InvalidInputError):
                continue
            if protocol.hash == protocol_hash:
                return record, protocol
        raise NotFoundError(f"Protocol was never registered: {protocol_hash}")

    def registration(self, protocol_hash: str) -> Derivation:
        """Return the earliest valid registration record for a protocol hash."""
        return self._registered(protocol_hash)[0]

    def entries(self, protocol_hash: str) -> tuple[Derivation, ...]:
        """Return the journal in order, after checking that its chain is intact."""
        previous = self.registration(protocol_hash).hash
        records = self.ledger.derivations(JOURNAL_NAMESPACE, subject_hash=protocol_hash)
        for index, record in enumerate(records):
            if (
                record.payload.get("index") != index
                or record.payload.get("previous") != previous
                or record.payload.get("kind") not in set(TransitionKind)
                or (record.payload.get("kind") == TransitionKind.CREATED)
                != (index == 0)
            ):
                raise InvalidInputError("Campaign journal chain is broken or forked")
            previous = record.hash
        return records

    def open(
        self, protocol: Protocol, screen_noise: Calibration, confirm_noise: Calibration
    ) -> None:
        """Register the protocol and write entry 0, or refuse an existing journal.

        Strict order: the protocol must be registered before its calibration
        runs, so call `register` before calibrating. Refusing here fails fast;
        research memory enforces the same rule when it admits a finding.
        """
        registered = datetime.fromisoformat(self.register(protocol).timestamp)
        for noise in (screen_noise, confirm_noise):
            for sample in noise.samples:
                created = self.ledger.get(sample.reference).created_at
                if datetime.fromisoformat(created) <= registered:
                    raise InvalidTransitionError(
                        "Calibration run predates protocol registration; "
                        "register the protocol before calibrating"
                    )
        if self.ledger.derivations(JOURNAL_NAMESPACE, subject_hash=protocol.hash):
            raise InvalidTransitionError(
                "A journal already exists for this protocol; restore it instead"
            )
        self.append(
            protocol.hash,
            TransitionKind.CREATED,
            {
                "screen_noise": _noise_record(screen_noise),
                "confirm_noise": _noise_record(confirm_noise),
            },
        )

    def append(
        self, protocol_hash: str, kind: TransitionKind, payload: Mapping[str, object]
    ) -> None:
        """Record one transition at the end of the chain."""
        entries = self.entries(protocol_hash)
        previous = (
            entries[-1].hash if entries else self.registration(protocol_hash).hash
        )
        self.ledger.record_derivation(
            JOURNAL_NAMESPACE,
            protocol_hash,
            {
                "index": len(entries),
                "previous": previous,
                "kind": kind,
                "body": payload,
            },
        )

    async def restore(self, protocol_hash: str) -> Campaign:
        """Rebuild the campaign by re-running every recorded transition.

        Each decision is recomputed from `Ledger.get` scores and must equal the
        recorded one. The restored campaign keeps appending to this journal. A
        recorded final start marks the held-out permission consumed, and no
        evaluator is called, so restoring never grants a second held-out test.
        """
        _, protocol = self._registered(protocol_hash)
        entries = self.entries(protocol_hash)
        if not entries:
            raise NotFoundError(f"No campaign journal for protocol {protocol_hash}")
        replay = Replay(self.ledger, protocol, entries[0])
        for entry in entries[1:]:
            await replay.apply(entry)
        campaign = await replay.finish()
        campaign.attach_journal(self)
        return campaign
