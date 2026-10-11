"""Synthetic ledger evidence only; these tests never import MLX or train."""

from collections.abc import Mapping
from dataclasses import replace
from statistics import stdev

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.ledger import (
    InvalidInputError,
    Ledger,
    Metric,
    Purpose,
    Reproducibility,
    Result,
    Run,
    Status,
)
from openloop.ledger.validation import finite_number
from openloop.loops import Fidelity, LoopSpec
from openloop.studies.s1 import Cohort, StudyPlan, analyze, build_plan, spearman


@pytest.fixture
def plan() -> StudyPlan:
    spec = LoopSpec(
        name="synthetic_t1_test",
        source_hash="1" * 64,
        dependencies_hash="2" * 64,
        data_hash="3" * 64,
        evaluator_hash="4" * 64,
        adapter_hash="5" * 64,
        budget_unit="tokens",
        reproducibility=Reproducibility.NOISY,
        fidelities=(Fidelity("8m", 8_388_608), Fidelity("21m", 20_971_520)),
        mutable_keys=("rate",),
        metric_name="val_bpb",
        metric_unit="BPB",
        execution={"simulated": True, "device": "cpu"},
    )
    return build_plan(
        spec,
        {
            "baseline": {"rate": 0},
            **{f"candidate-{i}": {"rate": i} for i in range(1, 6)},
        },
        "6" * 64,
    )


@pytest.fixture
def runs(plan: StudyPlan, ledger: Ledger) -> dict[str, Run]:
    records: dict[str, Run] = {}
    for i, trial in enumerate(plan.trials):
        run = ledger.submit(trial.inputs, purpose=Purpose.REPLICATION)
        ledger.start_stage(run.id, "execution")
        ledger.start_stage(run.id, "evaluation")
        rate = trial.inputs.config["rate"]
        assert isinstance(rate, int)
        score = 1.5 + rate * 0.1 + (i % 3) * 0.001
        records[trial.slot] = ledger.complete(
            run.id, Result({"val_bpb": Metric(score, "BPB")})
        )
    return records


def test_exact_budget_and_separate_noise_cohorts(plan: StudyPlan) -> None:
    assert len(plan.trials) == 70
    assert plan.training_tokens == 1_027_604_480
    assert plan.wall_limit_seconds == 43_200
    assert [trial.inputs.fidelity for trial in plan.trials[:2]] == ["8m", "21m"]
    for fidelity in ("8m", "21m"):
        for cohort in (Cohort.FIXED_SEED, Cohort.DISTINCT_SEED):
            trials = [
                trial
                for trial in plan.trials
                if trial.inputs.fidelity == fidelity and trial.cohort is cohort
            ]
            assert len(trials) == 10
            assert len({trial.inputs.hash for trial in trials}) == (
                1 if cohort is Cohort.FIXED_SEED else 10
            )
    assert len({trial.slot for trial in plan.trials}) == 70


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((1, 2, 3), (4, 5, 6), 1),
        ((1, 2, 3), (6, 5, 4), -1),
        ((1, 1, 2), (3, 3, 4), 1),
        ((1, 1, 1), (1, 2, 3), None),
        ((1, 2, 3), (1, 3, 2), 0.5),
    ],
)
def test_spearman_ties_and_undefined(left, right, expected) -> None:
    result = spearman(left, right)
    assert result is None if expected is None else result == pytest.approx(expected)


@given(st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=15))
def test_rank_invariance_to_monotone_transform(values: list[int]) -> None:
    left = tuple(float(value) for value in values)
    right = tuple(3 * value + 1 for value in left)
    result = spearman(left, right)
    assert result is None if len(set(values)) == 1 else result == pytest.approx(1)


def test_complete_report_uses_candidate_means_and_sample_std(
    plan: StudyPlan, runs: dict[str, Run]
) -> None:
    report = analyze(plan, runs)
    assert report["completed_runs"] == 70
    noise = report["noise"]
    rank = report["rank_correlation"]
    assert isinstance(noise, Mapping)
    high = noise["21m"]
    assert isinstance(high, Mapping)
    fixed = high["fixed_seed"]
    assert isinstance(fixed, Mapping)
    values = fixed["scores"]
    assert isinstance(values, tuple)
    assert fixed["sample_count"] == 10
    assert fixed["sample_std_bpb"] == pytest.approx(
        stdev(tuple(finite_number(value, "Score") for value in values))
    )
    assert isinstance(rank, Mapping)
    assert rank["candidate_count"] == 6
    assert rank["rho"] == pytest.approx(1)
    assert rank["seeds"] == (1000, 1001, 1002)
    assert rank["matched_seed_bootstrap_95_percentile_interval"] == (1, 1)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "alias",
        "failed",
        "duplicate_attempt",
        "wrong_input",
        "held_out",
        "not_replication",
    ],
)
def test_invalid_evidence_is_rejected(
    plan: StudyPlan, runs: dict[str, Run], invalid: str
) -> None:
    altered = dict(runs)
    slot = plan.trials[0].slot
    run = runs[slot]
    match invalid:
        case "missing":
            del altered[slot]
        case "alias":
            altered[slot] = replace(run, reused_from="other")
        case "failed":
            altered[slot] = replace(run, status=Status.FAILED, result=None)
        case "duplicate_attempt":
            altered[slot] = replace(
                run, attempt_id=runs[plan.trials[1].slot].attempt_id
            )
        case "wrong_input":
            altered[slot] = replace(run, inputs=replace(run.inputs, seed=999))
        case "held_out":
            altered[slot] = replace(
                run, result=Result({"val_bpb": Metric(1.5, "BPB", split="held_out")})
            )
        case "not_replication":
            altered[slot] = replace(run, purpose=Purpose.RUN)
    before = tuple(runs.values())
    with pytest.raises(InvalidInputError):
        analyze(plan, altered)
    assert tuple(runs.values()) == before


def test_plan_rejects_changed_context_or_missing_cohort(plan: StudyPlan) -> None:
    first = plan.trials[0]
    changed = replace(first, inputs=replace(first.inputs, data_hash="9" * 64))
    with pytest.raises(InvalidInputError, match="frozen"):
        replace(plan, trials=(changed, *plan.trials[1:]))
    with pytest.raises(InvalidInputError, match="ten"):
        replace(plan, trials=plan.trials[1:])


def test_rank_seeds_cannot_overlap_fixed_seed_or_skip_baseline(plan: StudyPlan) -> None:
    overlapping = tuple(
        replace(trial, inputs=replace(trial.inputs, seed=1000))
        if trial.cohort is Cohort.FIXED_SEED
        else trial
        for trial in plan.trials
    )
    with pytest.raises(InvalidInputError, match="separate"):
        replace(plan, trials=overlapping)
    with pytest.raises(InvalidInputError, match="Rank seeds"):
        replace(plan, rank_seeds=(9, 10, 11))
