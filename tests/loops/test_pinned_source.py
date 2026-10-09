"""Golden hashes of the adapted real upstream source. Compile only; nothing executes."""

import hashlib
from pathlib import Path

import pytest

from openloop.loops import ContractError, TrainingSource
from openloop.loops.t1_report import (
    REPORT_SEED,
    REPORT_TOKENIZER_DIR,
    build_report,
    generate_programs,
)

CLONE = Path(__file__).parents[2] / "research" / "autoresearch-mlx"

# Computed once from the pinned upstream source with the baseline configuration,
# seed 42, and tokenizer directory "/frozen/tokenizer". A change here means the
# adapter changed the code that frozen evidence refers to; find the cause first.
GOLDEN_PROGRAM_SHA256 = {
    "8m": "e09fb9e46b8abaf2c69079ecd3b796f3d1eaecfc81dd1b5e205c5fd3cd5bdb4a",
    "21m": "066b9b56802a2f1df14784b90e79082837bfa83a6933335b129fd08a8bd4e9c1",
}


@pytest.fixture(scope="module")
def source() -> TrainingSource:
    try:
        return TrainingSource.load(CLONE)
    except (FileNotFoundError, ContractError) as error:
        pytest.skip(f"Pinned upstream clone unavailable at {CLONE}: {error}")


def test_fixed_report_parameters() -> None:
    assert (REPORT_SEED, REPORT_TOKENIZER_DIR) == (42, "/frozen/tokenizer")


def test_real_source_programs_match_golden_hashes(source: TrainingSource) -> None:
    programs = generate_programs(source)
    assert set(programs) == set(GOLDEN_PROGRAM_SHA256)
    for name, program in programs.items():
        compile(program, f"candidate/{name}/train.py", "exec")
        digest = hashlib.sha256(program.encode("utf-8")).hexdigest()
        assert digest == GOLDEN_PROGRAM_SHA256[name]


def test_real_source_programs_have_token_control_and_no_evaluator(
    source: TrainingSource,
) -> None:
    for name, program in generate_programs(source).items():
        tokens = {"8m": 8_388_608, "21m": 20_971_520}[name]
        assert "mx.random.seed(42)" in program
        assert f"step * TOTAL_BATCH_SIZE >= {tokens}" in program
        assert f"progress = min(step * TOTAL_BATCH_SIZE / {tokens}, 1.0)" in program
        assert "STARTUP_EXCLUDE_STEPS = 0" in program
        assert "from prepare import" not in program
        assert "evaluate_bpb(" not in program
        assert "total_training_time >= TIME_BUDGET" not in program
        assert "/frozen/tokenizer" in program


def test_report_matches_golden_hashes_without_importing_runtime(
    source: TrainingSource,
) -> None:
    report = build_report(CLONE, report_date="2026-01-01")
    assert report["native_training_executed"] is False
    assert report["fidelities"] == [
        {
            "name": name,
            "training_tokens": tokens,
            "expected_updates": tokens // 65536,
            "program_sha256": GOLDEN_PROGRAM_SHA256[name],
            "compiled": True,
        }
        for name, tokens in (("8m", 8_388_608), ("21m", 20_971_520))
    ]
