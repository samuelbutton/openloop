"""Pure agent reports from recorded runs; no claims of causation or verification.

The toolbelt exposes selection evidence only. It never releases held-out scores.
"""

from collections.abc import Mapping, Sequence
from dataclasses import fields

from openloop.decider import Sample
from openloop.ledger import Direction, InvalidInputError, Metric, Run, canonical_json
from openloop.ledger.validation import finite_number, plain
from openloop.loops.models import Phase

type Report = dict[str, object]

MAX_PAGE_SIZE = 200


def page(items: Sequence[object], *, limit: int, offset: int) -> Report:
    """Return a bounded page with an explicit total and continuation offset."""
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
        raise InvalidInputError(f"Limit must be between 1 and {MAX_PAGE_SIZE}")
    if type(offset) is not int or offset < 0:
        raise InvalidInputError("Offset must be a nonnegative integer")
    end = offset + limit
    return {
        "items": list(items[offset:end]),
        "total": len(items),
        "next_offset": end if end < len(items) else None,
    }


def is_selection(run: Run) -> bool:
    """Recognize only the loop contract's two selection phases."""
    return run.inputs.phase in (Phase.SCREEN, Phase.CONFIRM)


def require_selection(run: Run) -> None:
    """Reject held-out and unknown phases before exposing any run details."""
    if not is_selection(run):
        raise InvalidInputError("Toolbelt access is limited to selection runs")


def validation_metrics(run: Run) -> Mapping[str, Metric]:
    """Return validation scores, with no observations or final-split scores."""
    if run.result is None:
        return {}
    return {
        name: metric
        for name, metric in run.result.metrics.items()
        if metric.split == "validation"
    }


def run_reference(run: Run) -> Report:
    """Identify a submission and its execution without counting cache reuse."""
    return {
        "run_id": run.id,
        "attempt_id": run.attempt_id,
        "input_hash": run.inputs.hash,
        "candidate_hash": run.inputs.candidate_hash,
        "status": run.status,
        "purpose": run.purpose,
        "reused_from": run.reused_from,
        "execution": plain(run.inputs.execution),
    }


def metric_row(run: Run, name: str, metric: Metric) -> Report:
    """Expose one original metric with its comparison scope and provenance."""
    return run_reference(run) | {
        "metric_name": name,
        "metric": plain(metric),
        "context_hash": Sample.from_run(run, name).context_hash,
        "loop_hash": run.inputs.loop_hash,
        "data_hash": run.inputs.data_hash,
        "evaluator_hash": run.inputs.evaluator_hash,
        "environment_hash": run.inputs.environment_hash,
        "fidelity": run.inputs.fidelity,
        "phase": run.inputs.phase,
        "seed": run.inputs.seed,
        "budget_unit": run.inputs.budget_unit,
        "budget_amount": run.inputs.budget_amount,
    }


def compare_runs(baseline: Run, candidate: Run) -> Report:
    """Report input differences and metric deltas; a delta is not a decision.

    Comparison requires two attempts with matched context and seed. Different
    candidates are allowed; different data, evaluators, environments, or work
    budgets are not.
    """
    for run in (baseline, candidate):
        require_selection(run)
    differences = {
        field.name: {
            "baseline": plain(getattr(baseline.inputs, field.name)),
            "candidate": plain(getattr(candidate.inputs, field.name)),
        }
        for field in fields(baseline.inputs)
        if canonical_json(getattr(baseline.inputs, field.name))
        != canonical_json(getattr(candidate.inputs, field.name))
    }
    old, new = validation_metrics(baseline), validation_metrics(candidate)
    metrics: list[Report] = []
    for name in sorted(old.keys() | new.keys()):
        left, right = old.get(name), new.get(name)
        comparable, delta, gain = False, None, None
        if left is not None and right is not None:
            left_sample = Sample(
                baseline.id,
                baseline.attempt_id,
                baseline.inputs,
                left,
                metric_name=name,
            )
            right_sample = Sample(
                candidate.id,
                candidate.attempt_id,
                candidate.inputs,
                right,
                metric_name=name,
            )
            comparable = (
                left_sample.context_hash == right_sample.context_hash
                and baseline.inputs.seed == candidate.inputs.seed
                and baseline.attempt_id != candidate.attempt_id
            )
            if comparable:
                delta = finite_number(right.value - left.value, "Metric delta")
                gain = -delta if right.direction is Direction.MINIMIZE else delta
        metrics.append(
            {
                "name": name,
                "baseline": plain(left),
                "candidate": plain(right),
                "comparable": comparable,
                "delta": delta,
                "gain": gain,
            }
        )
    return {
        "baseline": run_reference(baseline),
        "candidate": run_reference(candidate),
        "input_changes": differences,
        "metrics": metrics,
        "independent_attempts": baseline.attempt_id != candidate.attempt_id,
        "is_verification": False,
    }
