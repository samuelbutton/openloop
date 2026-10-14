"""Synthetic held-out transform, parsing and analysis tests; no MLX, no training."""

import pytest

from openloop.ledger import InvalidInputError
from openloop.studies.claims_audit import Observation
from openloop.studies.heldout import (
    EVAL_ANCHOR,
    HELDOUT_EVAL,
    HeldoutFinding,
    HeldoutObservation,
    analyze_heldout,
    combined_observations,
    heldout_run_observations,
    inject_heldout_eval,
    parse_heldout_bpb,
)

SEEDS = (3000, 3001, 3002, 3003, 3004)
PROGRAM = f"import os\nx = 1\n{EVAL_ANCHOR}print(val_bpb)\n"


def test_transform_appends_once_directly_after_the_published_evaluation() -> None:
    result = inject_heldout_eval(PROGRAM)
    assert result == PROGRAM.replace(EVAL_ANCHOR, EVAL_ANCHOR + HELDOUT_EVAL)
    assert result.index(EVAL_ANCHOR) < result.index("heldout_bpb = ")
    assert result.count("heldout_bpb = ") == 1
    assert result.endswith("print(val_bpb)\n")


@pytest.mark.parametrize(
    "bad", ["x = 1\n", PROGRAM + EVAL_ANCHOR, "val_bpb = evaluate_bpb(model)\n"]
)
def test_transform_fails_on_missing_or_repeated_anchor(bad: str) -> None:
    with pytest.raises(InvalidInputError):
        inject_heldout_eval(bad)


def test_transform_refuses_to_run_twice() -> None:
    with pytest.raises(InvalidInputError):
        inject_heldout_eval(inject_heldout_eval(PROGRAM))


def test_transformed_code_is_valid_python_and_reads_the_environment_variable() -> None:
    compile(inject_heldout_eval(PROGRAM), "train_mlx.py", "exec")
    assert 'os.environ["OPENLOOP_HELDOUT_DIR"]' in HELDOUT_EVAL


def test_parse_heldout_bpb_needs_exactly_one_positive_line() -> None:
    log = "step 1\nheldout_bpb:      1.402531\n---\nval_bpb:          1.399000\n"
    assert parse_heldout_bpb(log) == pytest.approx(1.402531)
    for bad in (
        "no score\n",
        log + "\nheldout_bpb: 1.4\n",
        "heldout_bpb: 0.000000\n",
        "heldout_bpb: nan\n",
        "x heldout_bpb: 1.4\n",
    ):
        with pytest.raises(InvalidInputError):
            parse_heldout_bpb(bad)


def pairs(
    child: tuple[float, ...],
    parent: tuple[float, ...],
    child_val: tuple[float, ...] | None = None,
    parent_val: tuple[float, ...] | None = None,
) -> list[HeldoutObservation]:
    child_val = child_val or child
    parent_val = parent_val or parent
    rows: list[HeldoutObservation] = []
    for arm, held, val in (("c", child, child_val), ("p", parent, parent_val)):
        rows += [
            HeldoutObservation(arm, s, v, h, 800)
            for s, h, v in zip(SEEDS, held, val, strict=True)
        ]
    return rows


def test_replicated_when_interval_excludes_zero_upward() -> None:
    parent = (1.40, 1.41, 1.42, 1.405, 1.415)
    child = tuple(x - 0.02 - 0.001 * i for i, x in enumerate(parent))
    report = analyze_heldout(pairs(child, parent), child="c", parent="p")
    assert report.finding is HeldoutFinding.REPLICATED
    assert report.heldout.interval is not None
    assert report.heldout.interval[0] > 0
    assert report.heldout.wins == 5
    assert report.heldout.sign_p == pytest.approx(1 / 32)
    assert report.heldout.mean_gain == pytest.approx(0.022)
    assert report.agreement.same_sign_seeds == 5


def test_reversed_and_inconclusive() -> None:
    parent = (1.40, 1.41, 1.42, 1.405, 1.415)
    worse = tuple(x + 0.02 + 0.001 * i for i, x in enumerate(parent))
    assert (
        analyze_heldout(pairs(worse, parent), child="c", parent="p").finding
        is HeldoutFinding.REVERSED
    )
    mixed = tuple(
        x + d for x, d in zip(parent, (0.03, -0.02, 0.01, -0.01, 0.0), strict=True)
    )
    report = analyze_heldout(pairs(mixed, parent), child="c", parent="p")
    assert report.finding is HeldoutFinding.INCONCLUSIVE


def test_finding_uses_heldout_scores_not_validation() -> None:
    parent = (1.40, 1.41, 1.42, 1.405, 1.415)
    better = tuple(x - 0.03 for x in parent)
    worse = tuple(x + 0.03 for x in parent)
    # Validation says the child is worse; held-out says it is better.
    report = analyze_heldout(
        pairs(better, parent, worse, parent), child="c", parent="p"
    )
    assert report.finding is HeldoutFinding.REPLICATED
    assert report.validation.mean_gain is not None
    assert report.validation.mean_gain < 0
    assert report.agreement.same_sign_seeds == 0


def test_agreement_shift_and_correlation() -> None:
    parent_h = (1.40, 1.41, 1.42, 1.405, 1.415)
    child_h = (1.39, 1.40, 1.41, 1.40, 1.40)
    parent_v = tuple(x - 0.01 for x in parent_h)
    child_v = tuple(x - 0.01 for x in child_h)
    report = analyze_heldout(
        pairs(child_h, parent_h, child_v, parent_v), child="c", parent="p"
    )
    assert report.agreement.mean_shift_child == pytest.approx(0.01)
    assert report.agreement.mean_shift_parent == pytest.approx(0.01)
    assert report.agreement.gain_correlation == pytest.approx(1.0)


def test_insufficient_data_and_unmatched_seeds() -> None:
    rows = [
        HeldoutObservation("c", 3000, 1.4, 1.4, 800),
        HeldoutObservation("p", 3000, 1.41, 1.41, 800),
        HeldoutObservation("c", 3001, 1.4, 1.4, 800),
    ]
    report = analyze_heldout(rows, child="c", parent="p")
    assert report.finding is HeldoutFinding.INSUFFICIENT_DATA
    assert report.heldout.interval is None
    assert report.unmatched == (3001,)


def test_analysis_rejects_duplicates_unknown_arms_and_same_arm() -> None:
    row = HeldoutObservation("c", 3000, 1.4, 1.4, 800)
    with pytest.raises(InvalidInputError):
        analyze_heldout([row, row], child="c", parent="p")
    with pytest.raises(InvalidInputError):
        analyze_heldout(
            [HeldoutObservation("z", 3000, 1.4, 1.4, 1)], child="c", parent="p"
        )
    with pytest.raises(InvalidInputError):
        analyze_heldout([], child="c", parent="c")


def receipt(
    kind: str, arm: str, seed: int, val: float | None, held: float | None = None
) -> dict[str, object]:
    summary: dict[str, object] | None = None
    if val is not None:
        summary = {"val_bpb": val, "num_steps": 800}
        if held is not None:
            summary["heldout_bpb"] = held
    return {"kind": kind, "arm": arm, "seed": seed, "summary": summary}


def test_heldout_receipts_skip_failures_and_other_kinds() -> None:
    records = [
        receipt("heldout", "c", 3000, 1.4, 1.41),
        receipt("heldout", "p", 3000, None),
        receipt("seed_completion", "x", 2004, 1.39),
    ]
    rows = heldout_run_observations(records)
    assert rows == [HeldoutObservation("c", 3000, 1.4, 1.41, 800)]


def test_combined_observations_add_the_completion_run() -> None:
    original = [
        {"arm": "a", "seed": 2000, "summary": {"val_bpb": 1.4, "num_steps": 800}},
        {"arm": "a", "seed": 2004, "summary": None, "failure": "env"},
    ]
    completion = [receipt("seed_completion", "a", 2004, 1.39)]
    rows = combined_observations(original, completion)
    assert rows == [Observation("a", 2000, 1.4, 800), Observation("a", 2004, 1.39, 800)]
    with pytest.raises(InvalidInputError):
        combined_observations(original, [receipt("heldout", "a", 3000, 1.4, 1.4)])
