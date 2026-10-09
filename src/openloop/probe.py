"""Run one explicit replication of a completed experiment and record the drift."""

from collections.abc import Callable
from contextlib import suppress

from openloop.ledger import (
    ExperimentInputs,
    InvalidTransitionError,
    Ledger,
    ProbeResult,
    Purpose,
    Result,
    Stage,
    Status,
    validate_tolerances,
)


def run_probe(
    ledger: Ledger,
    baseline_id: str,
    runner: Callable[[ExperimentInputs], Result],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ProbeResult:
    """Repeat a baseline with its exact inputs and persist the comparison.

    The runner must perform the experiment and return a trusted result.
    Any failure after admission, including interruption, fails the replication
    and propagates.
    """
    validate_tolerances(atol, rtol)
    baseline = ledger.get(baseline_id)
    if baseline.status is not Status.SUCCEEDED:
        raise InvalidTransitionError("A probe requires a completed baseline")
    repeat = ledger.submit(
        baseline.inputs,
        hypothesis="Determinism probe",
        parents=(baseline_id,),
        purpose=Purpose.REPLICATION,
    )
    try:
        ledger.start_stage(repeat.id, Stage.EXECUTION)
        result = runner(baseline.inputs)
        ledger.start_stage(repeat.id, Stage.EVALUATION)
        return ledger.record_probe(baseline_id, repeat.id, result, atol=atol, rtol=rtol)
    except BaseException as error:
        with suppress(InvalidTransitionError):
            ledger.fail(repeat.id, f"{type(error).__name__}: {error}")
        raise
