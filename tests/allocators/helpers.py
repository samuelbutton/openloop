"""Small simulated measurements through public loop and ledger APIs."""

from openloop.allocators import AllocationSnapshot
from openloop.ledger import Ledger, Metric, Result, Run
from openloop.loops import T0, Job, LoopAction, LoopSpec

ENVIRONMENT = "a" * 64


def job(loop: T0, index: int, *, seed: int = 0, fidelity: str = "1x") -> Job:
    config = loop.normalize_config({"coordinates": [index, 0]})
    return loop.build_job(
        loop.spec.inputs(config, LoopAction(seed, fidelity), ENVIRONMENT)
    )


def complete(ledger: Ledger, spec: LoopSpec, offered: Job, value: float) -> Run:
    run = ledger.submit(offered.inputs)
    ledger.start_stage(run.id, "execution")
    ledger.start_stage(run.id, "evaluation")
    return ledger.complete(
        run.id,
        Result({spec.metric_name: Metric(value, spec.metric_unit, spec.direction)}),
    )


def snapshot(
    loop: T0,
    ledger: Ledger,
    jobs: tuple[Job, ...],
    *,
    budget: int = 100,
    slots: int = 1,
) -> AllocationSnapshot:
    return AllocationSnapshot(
        loop.spec, ENVIRONMENT, jobs, ledger.query(), budget, slots=slots
    )
