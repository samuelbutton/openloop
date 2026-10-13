"""Successive halving screens a T0 family; the decider confirms the survivors.

Every score is simulated. The allocator, confirmation, and held-out stages use
disjoint seeds, so the survivors' confirmation tests are fresh and independent.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from openloop.allocators import (
    AllocationSnapshot,
    AllocationStatus,
    Rung,
    ScreenOutcome,
    SuccessiveHalving,
)
from openloop.decider import Campaign, ConfirmationScope, DecisionStage, Policy, Verdict
from openloop.ledger import Ledger
from openloop.loops import ExperimentEnv, LoopAction
from openloop.studies.t0_family import Arm, T0Study

RUNGS = (Rung("1x", (100, 101)), Rung("4x", (102, 103)))
ARM = Arm("allocator", Policy.ALLOCATOR, confirmation_scope=ConfirmationScope.SCREENED)


@dataclass(frozen=True)
class Trial:
    study: T0Study
    campaign: Campaign
    outcome: ScreenOutcome
    allocator_seeds: set[int]
    verified: bool
    selected: str | None


async def allocate(study: T0Study, ledger: Ledger) -> ScreenOutcome:
    """Run the halving bracket to its end on the frozen family."""
    loop = study.loop
    jobs = tuple(
        loop.build_job(
            loop.spec.inputs(config, LoopAction(seed, rung.fidelity), study.environment)
        )
        for config in study.configs.values()
        for rung in RUNGS
        for seed in rung.seeds
    )
    policy = SuccessiveHalving(tuple(study.configs), RUNGS, eta=2)
    envs = {
        candidate: ExperimentEnv(
            loop,
            ledger,
            runner=loop.run,
            config=config,
            environment_hash=study.environment,
            budget=10**6,
        )
        for candidate, config in study.configs.items()
    }
    while True:
        state = AllocationSnapshot(
            loop.spec,
            study.environment,
            jobs,
            ledger.query(limit=10**6),
            10**6,
            slots=100,
        )
        allocation = policy.select(state)
        if not allocation.jobs:
            assert allocation.status is AllocationStatus.DONE
            return policy.outcome(state)
        for item in allocation.jobs:
            inputs = item.inputs
            await envs[inputs.candidate_hash].step(
                LoopAction(inputs.seed, inputs.fidelity)
            )


async def run_trial(trial: int, ledger: Ledger, *, gain: float | None) -> Trial:
    study = T0Study(
        trial,
        gain=gain,
        worsening=None if gain is None else 1.0,
        arm=ARM,
    )
    protocol = study.protocol
    campaign = Campaign(
        protocol,
        await study.calibration(DecisionStage.SCREEN),
        await study.calibration(DecisionStage.CONFIRM),
    )
    # The family is the pool declared before allocation; the protocol froze it.
    outcome = await allocate(study, ledger)
    assert outcome.candidates == protocol.candidates
    for candidate in protocol.candidates:
        campaign.record_external_screen(
            candidate,
            promoted=candidate in outcome.survivors,
            evidence=outcome.evidence[candidate],
            seeds=outcome.seeds,
            source=outcome.policy_hash,
        )
    confirmed: dict[str, float] = {}
    for candidate in outcome.survivors:
        decision = campaign.evaluate(
            DecisionStage.CONFIRM,
            candidate,
            await study.pairs(candidate, protocol.confirm),
        )
        if decision.verdict is Verdict.PROMOTE and decision.median_gain is not None:
            confirmed[candidate] = decision.median_gain
    selected = min(confirmed, key=lambda item: (-confirmed[item], item), default=None)
    campaign.close_selection(selected)
    verified = False
    if selected is not None:
        verified = (await campaign.finalize(study.pairs)).verdict is Verdict.VERIFIED
    seeds = {run.inputs.seed for run in ledger.query(limit=10**6)}
    return Trial(study, campaign, outcome, seeds, verified, selected)


def test_planted_gain_is_screened_in_then_confirmed_and_verified(
    tmp_path: Path,
) -> None:
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        trial = asyncio.run(run_trial(7, ledger, gain=0.5))
        study, outcome = trial.study, trial.outcome
        protocol = study.protocol
        assert study.improvement in outcome.survivors
        assert len(outcome.survivors) == 4
        assert trial.selected == study.improvement
        assert trial.verified
        assert trial.campaign.final is not None
        assert trial.campaign.final.verdict is Verdict.VERIFIED
        screens = {
            decision.candidate_hash: decision.verdict
            for decision in trial.campaign.decisions
            if decision.stage is DecisionStage.SCREEN
        }
        assert set(screens) == set(protocol.candidates)
        assert {c for c, v in screens.items() if v is Verdict.PROMOTE} == set(
            outcome.survivors
        )
        confirmed = {
            decision.candidate_hash: decision
            for decision in trial.campaign.decisions
            if decision.stage is DecisionStage.CONFIRM
        }
        assert set(confirmed) == set(outcome.survivors)
        assert {d.test_alpha for d in confirmed.values()} == {protocol.alpha / 4}
        assert confirmed[study.improvement or ""].verdict is Verdict.PROMOTE
        assert trial.campaign.confirmation_set == tuple(sorted(outcome.survivors))
        # Screen evidence is the allocator's ledger runs, not decider samples.
        run_ids = {run.id for run in ledger.query(limit=10**6)}
        assert all(set(ref) <= run_ids for ref in outcome.evidence.values())
        assert trial.allocator_seeds == set(outcome.seeds) == {100, 101, 102, 103}
        assert trial.allocator_seeds.isdisjoint(protocol.confirm.seeds)
        assert trial.allocator_seeds.isdisjoint(protocol.held_out.seeds)
        assert trial.allocator_seeds.isdisjoint(protocol.confirm_noise_seeds)


def test_an_all_null_family_usually_yields_no_verified_winner(tmp_path: Path) -> None:
    verified = 0
    for index in range(6):
        with Ledger(tmp_path / f"ledger-{index}.sqlite3") as ledger:
            trial = asyncio.run(run_trial(50 + index, ledger, gain=None))
            assert len(trial.outcome.survivors) == 4
            verified += trial.verified
    assert verified == 0
