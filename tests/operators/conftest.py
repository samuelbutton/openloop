"""CPU fixtures and a container-contract stand-in. No isolation claim."""

import asyncio
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import pytest

from openloop.decider import (
    Calibration,
    Campaign,
    DecisionStage,
    Pair,
    Sample,
    StagePlan,
    Verdict,
)
from openloop.executors import (
    Artifact,
    ArtifactRole,
    CommitSource,
    ContainerExecutor,
    ProcessJob,
    Snapshot,
    State,
)
from openloop.ledger import JSONValue, Ledger, Run
from openloop.loops import T0, ExperimentEnv, LoopAction, PlantedTruth, RunContext
from openloop.operators import (
    HypothesisCard,
    ImplementationRequest,
    ImplementedCandidate,
    Implementer,
    ModelReply,
    Patch,
    ProposalDraft,
    ProposalRequest,
    Proposer,
    bind_candidate,
)
from openloop.operators.smoke_worker import smoke
from openloop.studies.t0_family import T0Study


@pytest.fixture
def loop() -> T0:
    return T0(PlantedTruth(noise_std=0))


def measured(
    loop: T0, ledger: Ledger, *, index: int = 0, seed: int = 0, phase: str = "screen"
) -> Run:
    env = ExperimentEnv(
        loop,
        ledger,
        runner=loop.run,
        config={"coordinates": [index, 0]},
        environment_hash="a" * 64,
        budget=5,
    )
    step = asyncio.run(env.step(LoopAction(seed, phase=phase)))
    return ledger.get(step.run_id)


@pytest.fixture
def baseline(loop: T0, ledger: Ledger) -> Run:
    return measured(loop, ledger)


async def proposal(request: ProposalRequest) -> ModelReply[ProposalDraft]:
    return ModelReply(
        ProposalDraft(
            "Move closer", "Planted fixture", "Change candidate.txt", -0.1, 0.7
        ),
        "scripted",
        "fixture",
        "proposal-1",
        simulated=True,
    )


@pytest.fixture
def proposer(loop: T0, ledger: Ledger) -> Proposer:
    return Proposer(loop.spec, ledger, proposal, simulated_only=True)


class SandboxStandIn(ContainerExecutor):
    """Execute only trusted test fixtures on the host; no isolation claim."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir()
        self._source = root / "fixture-source"
        self._source.mkdir()
        (self._source / "candidate.txt").write_text("before")
        (self._source / "protected.txt").write_text("protected")
        self.receipts: dict[str, Snapshot] = {}
        self.jobs: list[ProcessJob] = []
        self.cancelled: list[str] = []

    @property
    def source(self) -> CommitSource:
        return CommitSource(self._source, "a" * 40)

    @property
    def network(self) -> bool:
        return False

    @property
    def image(self) -> str:
        return "sha256:" + "a" * 64

    async def submit(self, job: ProcessJob, *, key: str) -> str:
        self.jobs.append(job)
        job_id = f"fixture-{len(self.jobs)}"
        directory = self.root / job_id
        directory.mkdir()
        outputs = directory / "outputs"
        outputs.mkdir()
        state, error = State.SUCCEEDED, None
        try:
            smoke(
                self._source,
                directory / "worktree",
                outputs,
                json.loads(job.files["manifest.json"]),
            )
        except RuntimeError as failure:
            state, error = State.FAILED, str(failure)
        artifacts = tuple(
            Artifact(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
                ArtifactRole.OUTPUT,
            )
            for path in outputs.rglob("*")
            if path.is_file()
        )
        self.receipts[job_id] = Snapshot(
            job_id, job.hash, state, datetime.now(UTC), artifacts=artifacts, error=error
        )
        return job_id

    async def collect(self, job_id: str) -> Snapshot:
        return self.receipts[job_id]

    async def cancel(self, job_id: str) -> Snapshot:
        self.cancelled.append(job_id)
        return self.receipts[job_id]


@pytest.fixture
def sandbox(tmp_path: Path) -> SandboxStandIn:
    return SandboxStandIn(tmp_path / "sandbox")


# --- End-to-end world: a real Campaign over ledger-backed T0 runs ---------------


@dataclass
class World:
    """Ledger runs for one T0Study. Baseline samples are shared, as in a study."""

    study: T0Study
    ledger: Ledger
    samples: dict[str, Sample] = field(default_factory=dict)
    screen_noise: Calibration = field(init=False)
    confirm_noise: Calibration = field(init=False)

    def sample(self, plan: StagePlan, seed: int, config: Mapping[str, JSONValue]):
        inputs = replace(plan.template, seed=seed, config=config)
        cached = self.samples.get(inputs.hash)
        if cached is not None:
            return cached
        loop = self.study.loop
        run = self.ledger.submit(inputs)
        job = loop.build_job(inputs)
        output = asyncio.run(loop.run(job, RunContext(run.id, run.attempt_id)))
        self.ledger.start_stage(run.id, "execution")
        self.ledger.start_stage(run.id, "evaluation")
        self.ledger.complete(run.id, loop.evaluate(job, output))
        sample = Sample.from_run(self.ledger.get(run.id), plan.metric_name)
        self.samples[inputs.hash] = sample
        return sample

    def pairs(self, candidate: str, plan: StagePlan) -> tuple[Pair, ...]:
        return tuple(
            Pair(
                self.sample(plan, seed, self.study.baseline),
                self.sample(plan, seed, self.study.configs[candidate]),
            )
            for seed in plan.seeds
        )

    def calibration(self, plan: StagePlan, seeds: tuple[int, ...]) -> Calibration:
        samples = [self.sample(plan, seed, self.study.baseline) for seed in seeds]
        return Calibration(tuple(zip(samples[::2], samples[1::2], strict=True)))


def make_world(ledger: Ledger, study: T0Study) -> World:
    world = World(study, ledger)
    protocol = study.protocol
    world.screen_noise = world.calibration(protocol.screen, protocol.screen_noise_seeds)
    world.confirm_noise = world.calibration(
        protocol.confirm, protocol.confirm_noise_seeds
    )
    return world


@lru_cache(maxsize=1)
def discarded_trial() -> tuple[int, str]:
    """First trial where a null candidate passes screen and fails confirmation.

    The search uses cached study samples only; nothing here touches a ledger.
    """
    for trial in range(200):
        study = T0Study(trial, gain=1.0, worsening=3.0)
        screen = asyncio.run(study.calibration(DecisionStage.SCREEN))
        confirm = asyncio.run(study.calibration(DecisionStage.CONFIRM))
        for candidate in study.configs:
            if candidate in (study.improvement, study.worsening):
                continue
            campaign = Campaign(study.protocol, screen, confirm)
            pairs = asyncio.run(study.pairs(candidate, study.protocol.screen))
            if (
                campaign.evaluate(DecisionStage.SCREEN, candidate, pairs).verdict
                is Verdict.PROMOTE
            ):
                pairs = asyncio.run(study.pairs(candidate, study.protocol.confirm))
                decision = campaign.evaluate(DecisionStage.CONFIRM, candidate, pairs)
                if decision.verdict is Verdict.DISCARD:
                    return trial, candidate
    raise AssertionError("No discarded null candidate in 200 trials")


@dataclass
class Bound:
    card: HypothesisCard
    candidate: ImplementedCandidate


def bind_card(
    world: World,
    sandbox: SandboxStandIn,
    candidate: str,
    *,
    probability: float = 0.7,
    interval: tuple[float, float] | None = None,
    title: str = "Move closer",
) -> Bound:
    """Propose, implement, and bind one family member after its screen runs."""
    ledger, study = world.ledger, world.study
    screen = world.pairs(candidate, study.protocol.screen)

    async def proposing(request: ProposalRequest) -> ModelReply[ProposalDraft]:
        return ModelReply(
            ProposalDraft(
                title,
                "Planted fixture",
                "Edit candidate.txt",
                -0.1,
                probability,
                interval,
            ),
            "scripted",
            "fixture",
            f"proposal-{title}",
            simulated=True,
        )

    async def editing(request: ImplementationRequest) -> ModelReply[Patch]:
        return ModelReply(
            Patch({"candidate.txt": "after"}),
            "scripted",
            "fixture",
            f"edit-{title}",
            simulated=True,
        )

    baseline = world.screen_noise.pairs[0][0].reference
    card = asyncio.run(
        Proposer(study.loop.spec, ledger, proposing, simulated_only=True).propose(
            "draft", baseline=baseline
        )
    )
    implementer = Implementer(
        sandbox,
        ledger,
        editing,
        base_files={"candidate.txt": "before"},
        smoke=(sys.executable, "-c", "pass"),
        simulated_only=True,
    )
    implemented = asyncio.run(implementer.implement(card))
    assert implemented is not None
    bind_candidate(ledger, implemented, screen[0].candidate.inputs)
    return Bound(card, implemented)
