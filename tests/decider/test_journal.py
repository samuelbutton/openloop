"""The campaign journal rebuilds a campaign by re-deciding from ledger scores."""

import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from openloop.decider import (
    JOURNAL_NAMESPACE,
    PROTOCOL_NAMESPACE,
    Campaign,
    CampaignJournal,
    ConfirmationScope,
    DecisionStage,
    Policy,
    TransitionKind,
    Verdict,
)
from openloop.decider.journal import _noise_record
from openloop.ledger import (
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    NotFoundError,
    canonical_json,
)
from openloop.studies.t0_family import T0Study
from tests.memory.test_memory import closed_campaign
from tests.operators.conftest import make_world


def state(campaign: Campaign) -> str:
    return canonical_json(
        [
            campaign.decisions,
            campaign.final,
            campaign.confirmation_set,
            campaign.confirmation_alpha,
            dict(campaign.screen_sources),
            campaign.closed,
            campaign.selected,
            campaign.final_consumed,
        ]
    )


def only_protocol(ledger: Ledger) -> str:
    [record] = ledger.derivations(PROTOCOL_NAMESPACE)
    return record.subject_hash


@pytest.mark.parametrize(
    ("policy", "scope", "top_k"),
    [
        (Policy.NOISE_AWARE, ConfirmationScope.FAMILY, None),
        (Policy.AUTOSCIENTISTS, ConfirmationScope.FAMILY, None),
        (Policy.NOISE_AWARE, ConfirmationScope.SCREENED, 1),
        (Policy.ALLOCATOR, ConfirmationScope.SCREENED, None),
    ],
)
def test_restore_reproduces_the_campaign_and_keeps_the_final_consumed(
    ledger: Ledger, policy: Policy, scope: ConfirmationScope, top_k: int | None
) -> None:
    campaign = closed_campaign(ledger, policy=policy, scope=scope, top_k=top_k)
    runs = ledger.query_all(include_reused=True)
    restored = asyncio.run(CampaignJournal(ledger).restore(campaign.protocol.hash))
    assert state(restored) == state(campaign)
    assert restored.final_consumed
    assert restored.final is not None
    assert restored.final.verdict is Verdict.VERIFIED

    async def never(candidate: str, plan: object):
        raise AssertionError("A restored campaign must not evaluate held-out again")

    with pytest.raises(InvalidTransitionError, match="unused permission"):
        asyncio.run(restored.finalize(never))  # type: ignore[arg-type]
    assert ledger.query_all(include_reused=True) == runs


def test_restore_after_a_crash_inside_the_final_keeps_permission_consumed(
    ledger: Ledger,
) -> None:
    def crash() -> None:
        raise RuntimeError("coordinator died")

    with pytest.raises(RuntimeError):
        closed_campaign(ledger, during_final=crash)
    restored = asyncio.run(CampaignJournal(ledger).restore(only_protocol(ledger)))
    assert restored.closed
    assert restored.final_consumed
    assert restored.final is None

    async def never(candidate: str, plan: object):
        raise AssertionError("A restored campaign must not evaluate held-out again")

    with pytest.raises(InvalidTransitionError, match="unused permission"):
        asyncio.run(restored.finalize(never))  # type: ignore[arg-type]


def test_a_restored_campaign_continues_and_keeps_journaling(ledger: Ledger) -> None:
    study = T0Study(99, gain=1.0, worsening=3.0, family_size=3)
    protocol = study.protocol
    journal = CampaignJournal(ledger)
    journal.register(protocol)
    world = make_world(ledger, study)
    campaign = Campaign(
        protocol, world.screen_noise, world.confirm_noise, journal=journal
    )
    for member in protocol.candidates:
        campaign.evaluate(
            DecisionStage.SCREEN, member, world.pairs(member, protocol.screen)
        )
    restored = asyncio.run(journal.restore(protocol.hash))
    assert state(restored) == state(campaign)
    assert study.improvement is not None
    for decision in restored.decisions:
        if decision.verdict is Verdict.PROMOTE:
            member = decision.candidate_hash
            restored.evaluate(
                DecisionStage.CONFIRM, member, world.pairs(member, protocol.confirm)
            )
    restored.close_selection(study.improvement)
    again = asyncio.run(journal.restore(protocol.hash))
    assert state(again) == state(restored)
    assert again.closed
    assert not again.final_consumed


def test_a_second_campaign_cannot_reuse_a_journaled_protocol(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    with pytest.raises(InvalidTransitionError, match="already exists"):
        Campaign(
            campaign.protocol,
            campaign.screen_noise,
            campaign.confirm_noise,
            journal=CampaignJournal(ledger),
        )


def test_unregistered_protocol_cannot_be_restored(ledger: Ledger) -> None:
    with pytest.raises(NotFoundError):
        asyncio.run(CampaignJournal(ledger).restore("0" * 64))


def journal_from(ledger: Ledger, source: Campaign, tamper: str | None) -> str:
    """Write a chained journal by hand, as a dishonest coordinator could."""
    journal = CampaignJournal(ledger)
    # A hand-written journal skips open()'s ordering check, as a dishonest
    # coordinator could; restore must still re-decide every entry.
    journal.register(source.protocol)
    journal.append(
        source.protocol.hash,
        TransitionKind.CREATED,
        {
            "screen_noise": _noise_record(source.screen_noise),
            "confirm_noise": _noise_record(source.confirm_noise),
        },
    )
    final = source.final
    assert final is not None
    for decision in source.decisions:
        if tamper == "screen" and decision.stage is DecisionStage.SCREEN:
            decision = replace(decision, median_gain=99.0)
            tamper = None
        if tamper == "confirm" and decision.stage is DecisionStage.CONFIRM:
            decision = replace(decision, p_value=0.0)
            tamper = None
        journal.append(
            source.protocol.hash, TransitionKind.DECISION, {"decision": decision}
        )
    journal.append(
        source.protocol.hash,
        TransitionKind.SELECTION_CLOSED,
        {"selected": source.selected},
    )
    journal.append(
        source.protocol.hash,
        TransitionKind.FINAL_STARTED,
        {"candidate_hash": source.selected},
    )
    if tamper == "final":
        final = replace(final, median_gain=99.0)
    journal.append(
        source.protocol.hash,
        TransitionKind.FINAL,
        {"decision": final},
    )
    return source.protocol.hash


def test_hand_written_journal_restores_only_when_honest(ledger: Ledger) -> None:
    source = closed_campaign(ledger, registration="never")
    protocol_hash = journal_from(ledger, source, None)
    restored = asyncio.run(CampaignJournal(ledger).restore(protocol_hash))
    assert state(restored) == state(source)


@pytest.mark.parametrize("tamper", ["screen", "confirm", "final"])
def test_tampered_decision_is_rejected(ledger: Ledger, tamper: str) -> None:
    source = closed_campaign(ledger, registration="never")
    protocol_hash = journal_from(ledger, source, tamper)
    with pytest.raises(InvalidInputError, match="differs from ledger evidence"):
        asyncio.run(CampaignJournal(ledger).restore(protocol_hash))


def edit_journal(path: Path, statement: str) -> None:
    with sqlite3.connect(path) as db:
        for (name,) in db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND tbl_name='derivation'"
        ).fetchall():
            db.execute(f"DROP TRIGGER {name}")
        db.execute(statement, (JOURNAL_NAMESPACE,))


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE derivation SET payload = json_set(payload, '$.body.decision.reason',"
        " 'edited') WHERE namespace = ? AND json_extract(payload, '$.index') = 1",
        "DELETE FROM derivation WHERE namespace = ? "
        "AND json_extract(payload, '$.index') = 1",
    ],
)
def test_edited_or_missing_journal_entries_break_the_chain(
    tmp_path: Path, statement: str
) -> None:
    path = tmp_path / "chain.sqlite3"
    with Ledger(path) as ledger:
        protocol_hash = closed_campaign(ledger).protocol.hash
    edit_journal(path, statement)
    with Ledger(path) as ledger, pytest.raises(InvalidInputError, match="chain"):
        asyncio.run(CampaignJournal(ledger).restore(protocol_hash))


def test_a_forked_journal_is_rejected(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    journal = CampaignJournal(ledger)
    entries = journal.entries(campaign.protocol.hash)
    ledger.record_derivation(
        JOURNAL_NAMESPACE,
        campaign.protocol.hash,
        {
            "index": 1,
            "previous": entries[0].hash,
            "kind": TransitionKind.SELECTION_CLOSED,
            "body": {"selected": None},
        },
    )
    with pytest.raises(InvalidInputError, match="chain"):
        asyncio.run(journal.restore(campaign.protocol.hash))
