"""Memory admission uses real temporary-ledger T0 evidence, never model claims."""

import asyncio
import json
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from openloop.decider import (
    Calibration,
    Campaign,
    CampaignJournal,
    ConfirmationScope,
    DecisionStage,
    Pair,
    Policy,
    Sample,
    StagePlan,
    Verdict,
)
from openloop.ledger import (
    ActivityKind,
    InvalidInputError,
    InvalidTransitionError,
    Ledger,
    NotFoundError,
    ReadOnlyError,
    Result,
    canonical_json,
    content_hash,
)
from openloop.memory import ResearchMemory
from openloop.studies.t0_family import T0Study
from tests.operators.conftest import World, make_world


def closed_campaign(
    ledger: Ledger,
    *,
    policy: Policy = Policy.NOISE_AWARE,
    scope: ConfirmationScope = ConfirmationScope.FAMILY,
    top_k: int | None = None,
    persist_final: bool = True,
    registration: str = "early",
    bind: Callable[[World], object] | None = None,
    during_final: Callable[[], None] | None = None,
) -> Campaign:
    """Run a whole campaign on T0 ledger runs.

    `registration` is "early" (journaled campaign, protocol registered before any
    run), "creation" (journaled campaign registered when it is created, after
    calibration), "after_calibration" (unjournaled, registered after calibration but
    before any comparison), "late" (unjournaled, registered just before the final
    test), or
    "never". `during_final` runs inside the final evaluator.
    """
    study = T0Study(99, gain=1.0, worsening=3.0, family_size=3)
    protocol = replace(
        study.protocol, policy=policy, confirmation_scope=scope, confirm_top_k=top_k
    )
    journal = CampaignJournal(ledger)
    journaled = registration in ("early", "creation")
    if registration == "early":
        journal.register(protocol)
    world = make_world(ledger, study)
    if registration == "after_calibration":
        journal.register(protocol)
    if bind is not None:
        bind(world)
    campaign = Campaign(
        protocol,
        world.screen_noise,
        world.confirm_noise,
        journal=journal if journaled else None,
    )
    for member in protocol.candidates:
        pairs = world.pairs(member, protocol.screen)
        if policy is Policy.ALLOCATOR:
            campaign.record_external_screen(
                member,
                promoted=member == study.improvement,
                evidence=[pair.candidate.reference for pair in pairs],
                seeds=protocol.screen.seeds,
                source="random-fixture",
            )
        else:
            campaign.evaluate(DecisionStage.SCREEN, member, pairs)
    for decision in campaign.decisions:
        if decision.verdict is Verdict.PROMOTE:
            if (
                campaign.confirmation_set is not None
                and decision.candidate_hash not in campaign.confirmation_set
            ):
                continue
            campaign.evaluate(
                DecisionStage.CONFIRM,
                decision.candidate_hash,
                world.pairs(decision.candidate_hash, protocol.confirm),
            )
    assert study.improvement is not None
    campaign.close_selection(study.improvement)
    if registration == "late":
        journal.register(protocol)

    async def final(candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        if during_final is not None:
            during_final()
        pairs = await study.pairs(candidate, plan)
        if not persist_final:
            return pairs

        def persist(sample: Sample) -> Sample:
            run = ledger.submit(sample.inputs)
            ledger.start_stage(run.id, "execution")
            ledger.start_stage(run.id, "evaluation")
            result = ledger.complete(
                run.id, Result({sample.metric_name: sample.metric}, {})
            )
            return Sample.from_run(result, sample.metric_name)

        return tuple(
            Pair(persist(pair.baseline), persist(pair.candidate)) for pair in pairs
        )

    assert asyncio.run(campaign.finalize(final)).verdict is Verdict.VERIFIED
    return campaign


@pytest.mark.parametrize(
    ("policy", "scope", "top_k"),
    [
        (Policy.NOISE_AWARE, ConfirmationScope.FAMILY, None),
        (Policy.AUTOSCIENTISTS, ConfirmationScope.FAMILY, None),
        (Policy.NOISE_AWARE, ConfirmationScope.SCREENED, 1),
        (Policy.ALLOCATOR, ConfirmationScope.SCREENED, None),
    ],
)
def test_verified_finding_is_scoped_immutable_and_idempotent(
    ledger: Ledger, policy: Policy, scope: ConfirmationScope, top_k: int | None
) -> None:
    campaign = closed_campaign(ledger, policy=policy, scope=scope, top_k=top_k)
    memory = ResearchMemory(ledger)
    runs = ledger.query_all(include_reused=True)
    finding = asyncio.run(memory.write(campaign))
    assert asyncio.run(memory.write(campaign)) == finding
    assert memory.get(finding.hash) == finding
    assert memory.query(candidate_hash=campaign.selected) == (finding,)
    assert memory.query(protocol_hash=campaign.protocol.hash, simulated=True) == (
        finding,
    )
    assert memory.query(simulated=False) == ()
    assert memory.query(loop_hash="0" * 64) == ()
    assert memory.query(protocol_hash="0" * 64) == ()
    assert ledger.query_all(include_reused=True) == runs
    assert finding.payload["decision_hash"] == content_hash(campaign.final)
    assert finding.payload["held_out_pairs"] == len(campaign.protocol.held_out.seeds)
    evidence = finding.payload["evidence"]
    assert isinstance(evidence, tuple)
    assert finding.payload["evidence_attempts"] == len(evidence)
    with pytest.raises(TypeError):
        finding.payload["simulated"] = False  # type: ignore[index]
    with pytest.raises(NotFoundError):
        memory.get("0" * 64)


def test_query_survives_reopen_and_cli_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    with Ledger(path) as ledger:
        campaign = closed_campaign(ledger)
        finding = asyncio.run(ResearchMemory(ledger).write(campaign))
    before = path.read_bytes()
    with Ledger(path, read_only=True) as ledger:
        memory = ResearchMemory(ledger)
        assert memory.get(finding.hash) == finding
        with pytest.raises(ReadOnlyError):
            asyncio.run(memory.write(campaign))
    output = subprocess.check_output(
        [sys.executable, "-m", "openloop.memory", str(path)], text=True
    )
    assert json.loads(output)[0]["hash"] == finding.hash
    assert path.read_bytes() == before


def test_verified_label_without_runs_cannot_enter_memory(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger, persist_final=False)
    card = ledger.record_activity(ActivityKind.HYPOTHESIS, "a" * 64, {})
    ledger.record_activity(
        ActivityKind.HYPOTHESIS_OUTCOME,
        card.subject_hash,
        {"outcome": "verified", "decision_hash": content_hash(campaign.final)},
        parent_sequence=card.sequence,
    )
    before = ledger.activities()
    with pytest.raises(NotFoundError):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()
    assert ledger.activities() == before


@pytest.mark.parametrize("change", ["gain", "p_value", "verdict", "evidence", "alias"])
def test_forged_or_invalid_final_is_rejected(ledger: Ledger, change: str) -> None:
    campaign = closed_campaign(ledger)
    final = campaign.final
    assert final is not None
    match change:
        case "gain":
            fake = replace(final, median_gain=100.0)
        case "p_value":
            fake = replace(final, p_value=0.0)
        case "verdict":
            fake = replace(final, verdict=Verdict.DISCARD)
        case "evidence":
            fake = replace(final, evidence=final.evidence[2:])
        case "alias":
            alias = ledger.submit(ledger.get(final.evidence[0]).inputs)
            assert alias.reused_from is not None
            fake = replace(final, evidence=(alias.id, *final.evidence[1:]))
        case _:
            raise AssertionError(change)
    campaign._final = fake  # Simulate an untrusted, fabricated coordinator record.
    before = canonical_json(ledger.query_all(include_reused=True))
    with pytest.raises((InvalidInputError, InvalidTransitionError)):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()
    assert canonical_json(ledger.query_all(include_reused=True)) == before


def test_confirmation_promotion_is_not_verified_memory(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    campaign._final = None
    with pytest.raises(InvalidTransitionError, match="verified final"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


def test_fabricated_confirmation_is_rejected(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    assert campaign.selected is not None
    key = campaign.selected, DecisionStage.CONFIRM
    decision = campaign._decisions[key]
    campaign._decisions[key] = replace(decision, test_alpha=campaign.protocol.alpha)
    with pytest.raises(InvalidInputError, match="Confirmation"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


@pytest.mark.parametrize("failed", [False, True])
def test_unfinished_or_failed_attempt_is_not_evidence(
    ledger: Ledger, failed: bool
) -> None:
    campaign = closed_campaign(ledger)
    final = campaign.final
    assert final is not None
    inputs = replace(ledger.get(final.evidence[0]).inputs, seed=999)
    run = ledger.submit(inputs)
    if failed:
        ledger.fail(run.id, "fixture crash")
    campaign._final = replace(final, evidence=(run.id, *final.evidence[1:]))
    with pytest.raises(InvalidInputError, match="successful"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


def test_calibration_scores_must_match_ledger(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    pairs = campaign.screen_noise.pairs
    left, right = pairs[0]
    fake = replace(left, metric=replace(left.metric, value=100))
    campaign._screen_noise = Calibration(((fake, right), *pairs[1:]))
    with pytest.raises(InvalidInputError, match="Calibration differs"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


def test_missing_family_member_prevents_publication(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    missing = next(key for key in campaign._decisions if key[0] != campaign.selected)
    del campaign._decisions[missing]
    with pytest.raises(InvalidTransitionError, match="Resolve the frozen family"):
        asyncio.run(ResearchMemory(ledger).write(campaign))
    assert ResearchMemory(ledger).query() == ()


def test_cli_explains_how_to_read_a_ledger_that_predates_memory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.sqlite3"
    with Ledger(path):
        pass
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE derivation")
        db.execute("PRAGMA user_version = 3")
        db.execute("PRAGMA journal_mode = DELETE")
    before = path.read_bytes()
    process = subprocess.run(
        [sys.executable, "-m", "openloop.memory", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "schema 3" in process.stderr
    assert "migrate=True" in process.stderr
    assert "Traceback" not in process.stderr
    assert path.read_bytes() == before
