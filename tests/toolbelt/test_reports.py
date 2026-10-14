"""Input and metric comparisons are descriptive facts, never verified findings."""

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.ledger import Direction, InvalidInputError, Metric, Result
from openloop.toolbelt.reports import compare_runs, page

from .conftest import mapping, records


def test_diff_reports_input_changes_and_oriented_gain(ledger, inputs, finish) -> None:
    baseline = finish(ledger.submit(inputs), Result({"loss": Metric(2, "BPB")}))
    changed = replace(inputs, config={"depth": 8})
    candidate = finish(ledger.submit(changed), Result({"loss": Metric(1.5, "BPB")}))
    report = compare_runs(baseline, candidate)
    assert report["input_changes"] == {
        "config": {
            "baseline": {"depth": 4, "layers": [1, 2]},
            "candidate": {"depth": 8},
        }
    }
    assert records(report["metrics"])[0]["delta"] == -0.5
    assert records(report["metrics"])[0]["gain"] == 0.5
    assert report["is_verification"] is False
    alias = ledger.submit(changed)
    reused = compare_runs(candidate, alias)
    assert reused["independent_attempts"] is False
    assert mapping(reused["candidate"])["reused_from"] == candidate.id


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 2},
        {"data_hash": "a" * 64},
        {"evaluator_hash": "a" * 64},
        {"environment_hash": "a" * 64},
        {"budget_amount": 2048},
        {"fidelity": "full"},
        {"phase": "confirm"},
        {"execution": {"simulated": True}},
    ],
)
def test_context_mismatch_has_no_delta(ledger, inputs, finish, change) -> None:
    result = Result({"loss": Metric(2, "BPB")})
    baseline = finish(ledger.submit(inputs), result)
    candidate = finish(ledger.submit(replace(inputs, **change)), result)
    row = records(compare_runs(baseline, candidate)["metrics"])[0]
    assert row["comparable"] is False
    assert row["delta"] is None
    assert row["gain"] is None


@pytest.mark.parametrize(
    "metric",
    [
        Metric(1, "seconds"),
        Metric(1, "BPB", direction=Direction.MAXIMIZE),
        Metric(1, "BPB", sample_count=2),
    ],
)
def test_metric_metadata_must_match(ledger, inputs, finish, metric) -> None:
    baseline = finish(ledger.submit(inputs), Result({"loss": Metric(2, "BPB")}))
    candidate = finish(
        ledger.submit(replace(inputs, config={})), Result({"loss": metric})
    )
    assert (
        records(compare_runs(baseline, candidate)["metrics"])[0]["comparable"] is False
    )


def test_missing_scores_and_failed_runs_have_no_invented_delta(ledger, inputs, finish):
    baseline = finish(ledger.submit(inputs), Result({"loss": Metric(2, "BPB")}))
    candidate = ledger.submit(replace(inputs, config={}))
    failed = ledger.fail(candidate.id, "fixture failure")
    row = records(compare_runs(baseline, failed)["metrics"])[0]
    assert row["candidate"] is None
    assert row["delta"] is None


def test_maximized_metric_gain_and_finite_delta(ledger, inputs, finish):
    old = finish(
        ledger.submit(inputs),
        Result({"score": Metric(1, "points", Direction.MAXIMIZE)}),
    )
    new = finish(
        ledger.submit(replace(inputs, config={})),
        Result({"score": Metric(2, "points", Direction.MAXIMIZE)}),
    )
    assert records(compare_runs(old, new)["metrics"])[0]["gain"] == 1
    huge_old = replace(old, result=Result({"score": Metric(-1e308, "points")}))
    huge_new = replace(new, result=Result({"score": Metric(1e308, "points")}))
    with pytest.raises(InvalidInputError, match="finite"):
        compare_runs(huge_old, huge_new)


def test_held_out_run_or_metric_never_leaks(ledger, inputs, finish) -> None:
    visible = finish(ledger.submit(inputs), Result({"loss": Metric(2, "BPB")}))
    hidden = finish(
        ledger.submit(replace(inputs, phase="held_out")),
        Result({"secret_score": Metric(999, "BPB", split="held_out")}),
    )
    with pytest.raises(InvalidInputError, match="selection"):
        compare_runs(visible, hidden)
    disguised = replace(visible, result=hidden.result)
    assert compare_runs(disguised, disguised)["metrics"] == []


@given(st.lists(st.integers(), max_size=500), st.integers(min_value=1, max_value=200))
def test_pagination_never_loses_or_duplicates_rows(values, limit: int) -> None:
    restored = []
    offset = 0
    while True:
        report = page(values, limit=limit, offset=offset)
        batch = report["items"]
        assert isinstance(batch, list)
        restored.extend(batch)
        assert report["total"] == len(values)
        next_offset = report["next_offset"]
        if next_offset is None:
            break
        assert isinstance(next_offset, int)
        assert next_offset > offset
        offset = next_offset
    assert restored == values


@pytest.mark.parametrize(("limit", "offset"), [(0, 0), (201, 0), (True, 0), (1, -1)])
def test_page_rejects_invalid_bounds(limit, offset) -> None:
    with pytest.raises(InvalidInputError):
        page([], limit=limit, offset=offset)


def test_same_attempt_pair_has_no_delta(ledger, inputs, finish) -> None:
    run = finish(ledger.submit(inputs), Result({"loss": Metric(2, "BPB")}))
    alias = ledger.submit(inputs)
    assert alias.attempt_id == run.attempt_id
    for other in (run, ledger.get(alias.id)):
        report = compare_runs(run, other)
        row = records(report["metrics"])[0]
        assert row["comparable"] is False
        assert row["delta"] is None
        assert row["gain"] is None
        assert report["independent_attempts"] is False
