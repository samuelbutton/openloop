"""Synthetic observations only; these tests never import MLX or train."""

import pytest

from openloop.decider import Verdict
from openloop.ledger import InvalidInputError
from openloop.studies.claims_audit import (
    ClaimedOutcome,
    Comparison,
    Finding,
    MemoryGuard,
    MemoryReading,
    Observation,
    analyze_audit,
    inject_seed,
    interleaved_order,
    mean_interval,
    memory_guard_policy,
    paired_t_p,
    parse_summary,
    substitute_once,
    t_quantile,
    t_upper_tail,
)

SEEDS = (2000, 2001, 2002, 2003, 2004)


def runs(arm: str, scores: tuple[float, ...]) -> list[Observation]:
    return [Observation(arm, s, v, 700) for s, v in zip(SEEDS, scores, strict=True)]


def test_t_distribution_matches_tables() -> None:
    assert t_quantile(0.025, 4) == pytest.approx(2.776445, abs=1e-5)
    assert t_quantile(0.025, 2) == pytest.approx(4.302653, abs=1e-5)
    assert t_upper_tail(2.776445, 4) == pytest.approx(0.025, abs=1e-6)
    assert t_upper_tail(0.0, 4) == pytest.approx(0.5)


def test_interval_and_p_value() -> None:
    gains = [0.1, 0.2, 0.15, 0.12, 0.18]
    low, high = mean_interval(gains, 0.95)
    assert low < sum(gains) / 5 < high
    assert low > 0
    assert paired_t_p(gains) < 0.01
    assert paired_t_p([0.1, -0.1, 0.1, -0.1]) == pytest.approx(1.0)
    assert paired_t_p([0.0, 0.0, 0.0]) == 1.0
    assert paired_t_p([0.1, 0.1, 0.1]) == 0.0


def test_inject_seed_requires_exactly_one_match() -> None:
    source = "a = 1\nmx.random.seed(42)\nb = 2\n"
    assert inject_seed(source, 2003) == "a = 1\nmx.random.seed(2003)\nb = 2\n"
    for bad in ("x = 1\n", source + source, source + "np.random.seed(1)\n"):
        with pytest.raises(InvalidInputError):
            inject_seed(bad, 1)
    with pytest.raises(InvalidInputError):
        inject_seed(source, -1)


def test_substitute_once_fails_loudly() -> None:
    assert substitute_once("a=1\nb=2\n", "b=2", "b=3") == "a=1\nb=3\n"
    for old in ("c=3", "=", ""):
        with pytest.raises(InvalidInputError):
            substitute_once("a=1\nb=2\n", old, "x")


def test_order_is_deterministic_complete_and_interleaved() -> None:
    arms = ["a", "b", "c"]
    first = interleaved_order(arms, SEEDS, 7)
    assert first == interleaved_order(arms[::-1], SEEDS[::-1], 7)
    assert sorted(first) == sorted((a, s) for a in arms for s in SEEDS)
    assert first != interleaved_order(arms, SEEDS, 8)
    with pytest.raises(InvalidInputError):
        interleaved_order(["a", "a"], SEEDS, 1)


def test_parse_summary_reads_only_the_final_block() -> None:
    log = "val_bpb: 9.9\n---\nval_bpb:          1.4\nnum_steps:        800\nchip: M4\n"
    parsed = parse_summary(log)
    assert parsed["val_bpb"] == 1.4
    assert parsed["num_steps"] == 800
    for bad in (
        "no summary",
        "x\n---\nval_bpb: nan\nnum_steps: 5\n",
        "x\n---\nval_bpb: 1\n",
    ):
        with pytest.raises(InvalidInputError):
            parse_summary(bad)


def comparison(claimed: ClaimedOutcome, child: float, parent: float) -> Comparison:
    return Comparison("c vs p", "c", "p", claimed, child, parent)


def test_replicated_win() -> None:
    observations = runs("p", (1.50, 1.52, 1.49, 1.51, 1.50)) + runs(
        "c", (1.40, 1.41, 1.39, 1.42, 1.40)
    )
    report = analyze_audit(
        [comparison(ClaimedOutcome.WIN, 1.44, 1.50)], observations, alpha=0.05
    )
    result = report.comparisons[0]
    assert result.finding is Finding.REPLICATED
    assert result.sign_p == pytest.approx(1 / 32)
    assert result.sign_p_bonferroni == pytest.approx(1 / 32)
    assert result.mean_gain == pytest.approx(0.1, abs=0.01)
    assert result.published_gain_gate is Verdict.PROMOTE


def test_win_does_not_replicate_when_gain_is_noise() -> None:
    observations = runs("p", (1.50, 1.52, 1.49, 1.51, 1.50)) + runs(
        "c", (1.49, 1.53, 1.50, 1.50, 1.51)
    )
    result = analyze_audit(
        [comparison(ClaimedOutcome.WIN, 1.429, 1.430)], observations
    ).comparisons[0]
    assert result.finding is Finding.NOT_REPLICATED
    assert result.published_gain_gate is Verdict.INCONCLUSIVE


def test_win_reversed_when_child_is_clearly_worse() -> None:
    observations = runs("p", (1.40, 1.41, 1.39, 1.42, 1.40)) + runs(
        "c", (1.50, 1.52, 1.49, 1.51, 1.50)
    )
    result = analyze_audit(
        [comparison(ClaimedOutcome.WIN, 1.40, 1.50)], observations
    ).comparisons[0]
    assert result.finding is Finding.REVERSED


@pytest.mark.parametrize(
    ("child", "expected"),
    [
        ((1.50, 1.53, 1.49, 1.52, 1.50), Finding.INDISTINGUISHABLE),
        ((1.60, 1.62, 1.59, 1.61, 1.60), Finding.ACTUALLY_WORSE),
        ((1.40, 1.41, 1.39, 1.42, 1.40), Finding.ACTUALLY_BETTER),
    ],
)
def test_not_win_findings(child: tuple[float, ...], expected: Finding) -> None:
    observations = runs("p", (1.50, 1.52, 1.49, 1.51, 1.50)) + runs("c", child)
    result = analyze_audit(
        [comparison(ClaimedOutcome.NOT_WIN, 1.43, 1.42)], observations
    ).comparisons[0]
    assert result.finding is expected


def test_unmatched_seeds_are_reported_and_few_pairs_are_insufficient() -> None:
    observations = runs("p", (1.5,) * 5)[:2] + runs("c", (1.4,) * 5)
    report = analyze_audit([comparison(ClaimedOutcome.WIN, 1.4, 1.5)], observations)
    result = report.comparisons[0]
    assert result.finding is Finding.INSUFFICIENT_DATA
    assert result.seeds == SEEDS[:2]
    assert result.t_p is None
    assert report.unmatched["c vs p"] == SEEDS[2:]


def test_bonferroni_scales_with_comparison_count() -> None:
    observations = runs("p", (1.5, 1.51, 1.52, 1.5, 1.49)) + runs(
        "c", (1.39, 1.42, 1.40, 1.41, 1.38)
    )
    one = comparison(ClaimedOutcome.WIN, 1.4, 1.5)
    other = Comparison("x vs y", "x", "y", ClaimedOutcome.NOT_WIN, 1.0, 1.0)
    report = analyze_audit([one, other], observations)
    first = report.comparisons[0]
    assert report.comparison_count == 2
    assert first.sign_p_bonferroni == pytest.approx(2 / 32)
    assert first.t_p_bonferroni == pytest.approx(min(1.0, 2 * (first.t_p or 0)))
    assert first.adjusted_interval is not None
    assert first.interval is not None
    assert first.adjusted_interval[0] < first.interval[0]
    assert report.comparisons[1].finding is Finding.INSUFFICIENT_DATA


def test_duplicate_runs_are_rejected() -> None:
    with pytest.raises(InvalidInputError):
        analyze_audit(
            [comparison(ClaimedOutcome.WIN, 1.4, 1.5)],
            runs("p", (1.5,) * 5) + runs("p", (1.5,) * 5),
        )


def test_comparison_rejects_self_comparison() -> None:
    with pytest.raises(InvalidInputError):
        Comparison("x", "a", "a", ClaimedOutcome.WIN, 1.0, 1.0)


def _reading(level: int, free: int, at: str = "t") -> MemoryReading:
    return MemoryReading(at, level, free)


def test_guard_trips_on_critical_pressure() -> None:
    guard = MemoryGuard()
    assert guard.observe(_reading(1, 80)) is None
    trip = guard.observe(_reading(4, 80, "t2"))
    assert trip is not None
    assert "critical" in trip.reason
    assert trip.readings[-1].observed_at == "t2"


def test_guard_trips_on_two_consecutive_low_free_samples() -> None:
    guard = MemoryGuard()
    assert guard.observe(_reading(2, 14)) is None
    trip = guard.observe(_reading(2, 10))
    assert trip is not None
    assert [r.free_percent for r in trip.readings] == [14, 10]


def test_guard_ignores_warning_and_isolated_low_samples() -> None:
    guard = MemoryGuard()
    for level, free in [(2, 60), (2, 15), (2, 14), (2, 40), (2, 14), (2, 90)]:
        assert guard.observe(_reading(level, free)) is None


def test_guard_policy_is_recorded_and_readings_validated() -> None:
    assert memory_guard_policy() == {
        "sample_seconds": 5.0,
        "critical_pressure_level": 4,
        "free_percent_floor": 15,
        "free_percent_consecutive_samples": 2,
    }
    with pytest.raises(InvalidInputError):
        _reading(3, 50)
    with pytest.raises(InvalidInputError):
        _reading(1, 101)
