"""Pure selection views from ledger records; no evaluation or artifact reads.

Noise bands describe seed variation, not confidence or verified improvement.
Cache aliases stay visible in lineage and never contribute curve samples.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from statistics import fmean, stdev

from openloop.ledger import EventKind, InvalidInputError, Run, Status, content_hash
from openloop.ledger.validation import check_positive, finite_number, plain
from openloop.toolbelt.reports import (
    Report,
    is_selection,
    run_reference,
    validation_metrics,
)


def _scope(run: Run, name: str) -> tuple[str, Report]:
    assert run.result is not None
    metric = run.result.metrics[name]
    template = replace(
        run.inputs,
        source_hash="0" * 64,
        dependencies_hash="0" * 64,
        config={},
        seed=0,
        fidelity="curve",
        budget_amount=1,
    )
    descriptor = {
        "inputs": plain(template),
        "metric": name,
        "unit": metric.unit,
        "direction": metric.direction,
        "split": metric.split,
        "sample_count": metric.sample_count,
    }
    return content_hash(descriptor), descriptor


def _statistic(operation: Callable[[list[float]], float], values: list[float]) -> float:
    try:
        return finite_number(operation(values), "Curve statistic")
    except OverflowError as error:
        raise InvalidInputError("Curve statistics overflow") from error


def curves(runs: tuple[Run, ...]) -> list[Report]:
    """Group exact comparison scopes, then candidate and fidelity, then work budget.

    Each fidelity is its own series, so no line joins endpoints across fidelities.

    Average repeat attempts within each seed first. Give each seed equal weight.
    Show the sample SD of those seed means only when at least two seeds exist.
    """
    groups: dict[tuple[str, str, str, int], list[tuple[Run, float]]] = defaultdict(list)
    scopes: dict[str, Report] = {}
    seen: set[str] = set()
    for run in runs:
        if (
            not is_selection(run)
            or run.reused_from
            or run.status is not Status.SUCCEEDED
        ):
            continue
        if run.attempt_id in seen:
            continue
        seen.add(run.attempt_id)
        for name, metric in validation_metrics(run).items():
            scope_hash, descriptor = _scope(run, name)
            scopes[scope_hash] = descriptor
            key = (
                scope_hash,
                run.inputs.candidate_hash,
                run.inputs.fidelity,
                run.inputs.budget_amount,
            )
            groups[key].append((run, metric.value))
    series: dict[tuple[str, str, str], list[Report]] = defaultdict(list)
    for (scope, candidate, fidelity, budget), measurements in sorted(groups.items()):
        by_seed: dict[int, list[float]] = defaultdict(list)
        for run, value in measurements:
            by_seed[run.inputs.seed].append(value)
        means = [_statistic(fmean, values) for values in by_seed.values()]
        mean = _statistic(fmean, means)
        sd = _statistic(stdev, means) if len(means) > 1 else None
        series[scope, candidate, fidelity].append(
            {
                "budget": budget,
                "fidelity": fidelity,
                "mean": mean,
                "sd": sd,
                "low": finite_number(mean - sd, "Band lower bound")
                if sd is not None
                else None,
                "high": finite_number(mean + sd, "Band upper bound")
                if sd is not None
                else None,
                "seed_count": len(means),
                "attempt_count": len(measurements),
                "run_ids": [run.id for run, _ in measurements],
            }
        )
    return [
        {
            "scope_hash": scope,
            "scope": scopes[scope],
            "candidate_hash": candidate,
            "fidelity": fidelity,
            "points": points,
        }
        for (scope, candidate, fidelity), points in sorted(series.items())
    ]


def snapshot(runs: tuple[Run, ...], *, graph_limit: int = 500) -> Report:
    """Show recent selection submissions and full-history selection curves.

    Missing parents are named by count only; hidden run details never appear.
    Parent links include retry ancestry. Cache links have a separate edge kind.
    """
    check_positive(graph_limit)
    selected = tuple(run for run in runs if is_selection(run))
    visible = selected[-graph_limit:]
    ids = {run.id for run in visible}
    nodes, edges = [], []
    missing: set[str] = set()
    for run in visible:
        stage_times = {
            event.stage.value: event.payload.get("duration_seconds")
            for event in run.events
            if event.kind is EventKind.STAGE_FINISHED and event.stage is not None
        }
        nodes.append(
            run_reference(run)
            | {
                "hypothesis": run.hypothesis,
                "created_at": run.created_at,
                "seed": run.inputs.seed,
                "fidelity": run.inputs.fidelity,
                "phase": run.inputs.phase,
                "budget": run.inputs.budget_amount,
                "budget_unit": run.inputs.budget_unit,
                "metrics": plain(validation_metrics(run)),
                "stage_seconds": stage_times,
                "inputs": plain(run.inputs),
            }
        )
        links = [(parent, "parent") for parent in run.parents]
        if run.reused_from is not None:
            links.append((run.reused_from, "cache"))
        for parent, kind in links:
            if parent in ids:
                edges.append({"source": parent, "target": run.id, "kind": kind})
            else:
                missing.add(parent)
    originals = tuple(run for run in selected if run.reused_from is None)
    return {
        "nodes": nodes,
        "edges": edges,
        "series": curves(selected),
        "summary": {
            "runs": len(selected),
            "attempts": len({run.attempt_id for run in originals}),
            "cache_links": len(selected) - len(originals),
            "active": sum(
                run.status in (Status.QUEUED, Status.RUNNING) for run in originals
            ),
        },
        "graph_limit": graph_limit,
        "graph_truncated": len(visible) < len(selected),
        "omitted_parents": len(missing),
    }
