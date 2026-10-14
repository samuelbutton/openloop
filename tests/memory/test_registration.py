"""A finding needs a protocol registration older than every comparison run."""

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

from openloop.decider import PROTOCOL_NAMESPACE
from openloop.ledger import (
    Derivation,
    InvalidTransitionError,
    Ledger,
    NotFoundError,
    canonical_json,
    content_hash,
)
from openloop.memory import ResearchMemory
from tests.memory.test_memory import closed_campaign


def facts(finding: Derivation) -> dict[str, Any]:
    return json.loads(canonical_json(finding.payload))


def test_registered_protocol_is_published_with_its_order(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    finding = asyncio.run(ResearchMemory(ledger).write(campaign))
    registration = facts(finding)["registration"]
    [record] = ledger.derivations(PROTOCOL_NAMESPACE)
    assert registration["record_hash"] == record.hash
    assert registration["registered_at"] == record.timestamp
    # Strict order: registration precedes every run, calibration included.
    assert record.timestamp < registration["first_run_at"]


def test_registration_after_calibration_is_refused_at_campaign_creation(
    ledger: Ledger,
) -> None:
    with pytest.raises(InvalidTransitionError, match="register the protocol before"):
        closed_campaign(ledger, registration="creation")
    assert ResearchMemory(ledger).query() == ()


def test_registration_after_calibration_is_rejected_by_memory(
    ledger: Ledger,
) -> None:
    # The old rule allowed this: registered before every comparison run, but
    # after calibration. The strict rule rejects it at publication.
    campaign = closed_campaign(ledger, registration="after_calibration")
    with pytest.raises(InvalidTransitionError, match="including calibration"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


def test_unregistered_protocol_is_rejected(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger, registration="never")
    with pytest.raises(NotFoundError, match="never registered"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


def test_late_registration_is_rejected(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger, registration="late")
    with pytest.raises(InvalidTransitionError, match="not earlier"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


def test_a_record_that_only_claims_the_protocol_hash_is_not_a_registration(
    ledger: Ledger,
) -> None:
    campaign = closed_campaign(ledger, registration="never")
    other = replace(campaign.protocol, alpha=0.01)
    ledger.record_derivation(
        PROTOCOL_NAMESPACE,
        campaign.protocol.hash,
        {"protocol_hash": campaign.protocol.hash, "protocol": other},
    )
    assert content_hash(other) != content_hash(campaign.protocol)
    with pytest.raises(NotFoundError, match="never registered"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()
