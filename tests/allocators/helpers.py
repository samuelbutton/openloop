"""Small simulated measurements through public loop and ledger APIs."""

from dataclasses import replace

from openloop.allocators import AllocationSnapshot
from openloop.ledger import Ledger, Metric, Result, Run
from openloop.loops import T0, Fidelity, Job, LoopAction, LoopSpec

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
        loop.spec, ENVIRONMENT, jobs, ledger.query_all(), budget, slots=slots
    )


def fail(ledger: Ledger, offered: Job) -> Run:
    """Record a terminal worker failure for one input."""
    run = ledger.submit(offered.inputs)
    ledger.start_stage(run.id, "execution")
    return ledger.fail(run.id, "worker crashed")


def geometric_loop(*amounts: int) -> T0:
    """A T0 whose declared fidelities are the given work amounts, named by amount."""
    loop = T0()
    loop._spec = replace(  # pyright: ignore[reportPrivateUsage]
        loop.spec, fidelities=tuple(Fidelity(str(amount), amount) for amount in amounts)
    )
    return loop
