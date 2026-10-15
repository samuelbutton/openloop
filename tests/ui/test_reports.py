"""Lineage and measured seed bands from synthetic, temporary-ledger scores."""

from dataclasses import replace
from math import sqrt

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.ledger import (
    ExperimentInputs,
    InvalidInputError,
    Metric,
    Purpose,
    Reproducibility,
    Result,
    Run,
    Status,
)
from openloop.ui.reports import curves, snapshot
from tests.toolbelt.conftest import mapping, records


def test_bands_equal_weight_seeds_and_exclude_cache(ledger, inputs, finish):
    first = finish(ledger.submit(inputs), Result({"loss": Metric(1, "BPB")}))
    alias = ledger.submit(inputs)
    repeat = finish(
        ledger.submit(inputs, purpose=Purpose.REPLICATION),
        Result({"loss": Metric(3, "BPB")}),
    )
    second = finish(
        ledger.submit(replace(inputs, seed=43)), Result({"loss": Metric(4, "BPB")})
    )
    series = curves((first, alias, repeat, second, first))
    assert len(series) == 1
    point = records(series[0]["points"])[0]
    assert point["mean"] == 3
    assert point["sd"] == pytest.approx(sqrt(2))
    assert point["seed_count"] == 2
    assert point["attempt_count"] == 3
    assert point["run_ids"] == [first.id, repeat.id, second.id]


def test_one_seed_has_no_band_and_different_budgets_are_points(ledger, inputs, finish):
    first = finish(ledger.submit(inputs), Result({"loss": Metric(1, "BPB")}))
    second = finish(
        ledger.submit(replace(inputs, budget_amount=2048)),
        Result({"loss": Metric(2, "BPB")}),
    )
    series = curves((second, first))
    assert len(series) == 1
    points = records(series[0]["points"])
    assert [point["budget"] for point in points] == [1024, 2048]
    for point in points:
        assert point["sd"] is None
        assert point["low"] is None
        assert point["high"] is None


def test_fidelities_are_separate_series_in_one_scope(ledger, inputs, finish):
    result = Result({"loss": Metric(1, "BPB")})
    first = finish(ledger.submit(inputs), result)
    second = finish(ledger.submit(replace(inputs, fidelity="full")), result)
    series = curves((first, second))
    assert len({item["scope_hash"] for item in series}) == 1
    assert {item["fidelity"] for item in series} == {inputs.fidelity, "full"}
    assert len(series) == 2
    for item in series:
        points = records(item["points"])
        assert {point["fidelity"] for point in points} == {item["fidelity"]}


@pytest.mark.parametrize(
    "change",
    [
        {"data_hash": "a" * 64},
        {"environment_hash": "a" * 64},
        {"evaluator_hash": "a" * 64},
        {"loop_hash": "a" * 64},
        {"execution": {"simulated": True}},
        {"phase": "confirm"},
    ],
)
def test_different_contexts_never_share_a_scope(ledger, inputs, finish, change):
    result = Result({"loss": Metric(1, "BPB")})
    first = finish(ledger.submit(inputs), result)
    second = finish(ledger.submit(replace(inputs, **change)), result)
    series = curves((first, second))
    assert len({item["scope_hash"] for item in series}) == 2


def test_dag_keeps_failed_active_retry_and_cache_lineage(ledger, inputs, finish):
    failed = ledger.fail(ledger.submit(inputs).id, "fixture failure")
    retry = ledger.submit(inputs, purpose=Purpose.RETRY, retry_of=failed.id)
    done = finish(retry, Result({"loss": Metric(1, "BPB")}))
    alias = ledger.submit(inputs)
    active = ledger.submit(replace(inputs, seed=44), parents=(done.id,))
    hidden = finish(
        ledger.submit(replace(inputs, phase="held_out")),
        Result({"loss": Metric(12345, "BPB", split="held_out")}),
    )
    report = snapshot(ledger.query_all(include_reused=True))
    assert report["edges"] == [
        {"source": failed.id, "target": retry.id, "kind": "parent"},
        {"source": done.id, "target": alias.id, "kind": "cache"},
        {"source": done.id, "target": active.id, "kind": "parent"},
    ]
    assert hidden.id not in str(report)
    assert "12345" not in str(report)
    assert mapping(report["summary"]) == {
        "runs": 4,
        "attempts": 3,
        "cache_links": 1,
        "active": 1,
    }
    assert records(report["series"])
    narrow = snapshot(ledger.query_all(include_reused=True), graph_limit=1)
    assert narrow["graph_truncated"] is True
    assert narrow["omitted_parents"] == 1
    assert narrow["series"] == report["series"]


def test_mislabeled_hidden_metric_and_unfinished_run_do_not_plot(
    ledger, inputs, finish
):
    run = finish(
        ledger.submit(inputs), Result({"loss": Metric(9, "BPB", split="held_out")})
    )
    queued = ledger.submit(replace(inputs, seed=55))
    assert curves((run, queued)) == []
    assert records(snapshot((run, queued))["nodes"])[0]["metrics"] == {}


@given(st.lists(st.floats(-1000, 1000, allow_nan=False), min_size=2, max_size=10))
def test_noise_band_is_order_invariant(values):
    inputs = ExperimentInputs(
        source_hash="a" * 64,
        dependencies_hash="b" * 64,
        data_hash="c" * 64,
        environment_hash="d" * 64,
        evaluator_hash="e" * 64,
        loop_hash="f" * 64,
        seed=0,
        fidelity="screen",
        budget_unit="tokens",
        budget_amount=10,
        reproducibility=Reproducibility.NOISY,
    )
    runs = tuple(
        Run(
            str(seed),
            str(seed),
            replace(inputs, seed=seed),
            "",
            (),
            Purpose.RUN,
            None,
            "fixture",
            None,
            Status.SUCCEEDED,
            (),
            (),
            Result({"loss": Metric(value, "BPB")}),
        )
        for seed, value in enumerate(values)
    )
    left = records(curves(runs)[0]["points"])[0]
    right = records(curves(tuple(reversed(runs)))[0]["points"])[0]
    assert left["mean"] == right["mean"]
    assert left["sd"] == right["sd"]


def test_invalid_graph_limit_is_rejected():
    with pytest.raises(InvalidInputError):
        snapshot((), graph_limit=0)
