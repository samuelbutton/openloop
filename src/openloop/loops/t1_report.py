"""Compile-only T1 report. It generates the training programs and never runs them.

Nothing here imports MLX or starts training. Run it with the interpreter whose
installed versions the report should certify. Boundary: reads pinned source files
and, when given, the frozen corpus manifest and tokenizer files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from openloop.ledger import JSONValue

from .corpus import Corpus
from .files import adapter_digest
from .t1 import (
    ADAPTER_FILES,
    FIDELITIES,
    UPSTREAM_HASHES,
    UPSTREAM_REVISION,
    TrainingSource,
    tokenizer_digest,
)
from .t1_worker import verify_runtime
from .training import TrainingSetting, baseline_config, training_program

REPORT_SEED = 42
REPORT_TOKENIZER_DIR = "/frozen/tokenizer"


def program_digest(program: str) -> str:
    return hashlib.sha256(program.encode("utf-8")).hexdigest()


def generate_programs(source: TrainingSource) -> dict[str, str]:
    """Return each fidelity's program for the baseline, report seed, and report path."""
    config = baseline_config(source.train)
    return {
        level.name: training_program(
            source.train,
            config,
            seed=REPORT_SEED,
            token_budget=level.amount,
            tokenizer_dir=REPORT_TOKENIZER_DIR,
        )
        for level in FIDELITIES
    }


def build_report(
    repository: Path,
    *,
    report_date: str,
    corpus_root: Path | None = None,
    tokenizer_dir: Path | None = None,
    check_runtime: bool = False,
) -> dict[str, JSONValue]:
    """Compile both programs without executing them and describe the result."""
    source = TrainingSource.load(repository)
    batch = baseline_config(source.train)[TrainingSetting.TOTAL_BATCH_SIZE]
    if type(batch) is not int:
        raise TypeError("Baseline batch size must be an integer")
    fidelities: list[JSONValue] = []
    for name, program in generate_programs(source).items():
        compile(program, f"candidate/{name}/train.py", "exec")
        amount = next(level.amount for level in FIDELITIES if level.name == name)
        fidelities.append(
            {
                "name": name,
                "training_tokens": amount,
                "expected_updates": amount // batch,
                "program_sha256": program_digest(program),
                "compiled": True,
            }
        )
    report: dict[str, JSONValue] = {
        "date": report_date,
        "check": "compile only; installed versions checked"
        if check_runtime
        else "compile only",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_sha256": dict(UPSTREAM_HASHES),
        "python": ".".join(map(str, sys.version_info[:3])),
        "seed": REPORT_SEED,
        "tokenizer_dir": REPORT_TOKENIZER_DIR,
        "adapter_hash": adapter_digest(*ADAPTER_FILES),
        "fidelities": fidelities,
    }
    if check_runtime:
        report["dependencies"] = dict(verify_runtime(source.lock))
    if corpus_root is not None:
        report["data_hash"] = Corpus.load(corpus_root).snapshot_hash
    if tokenizer_dir is not None:
        report["tokenizer_hash"] = tokenizer_digest(tokenizer_dir)
    report["native_training_executed"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.upstream,
        report_date=args.date,
        corpus_root=args.corpus,
        tokenizer_dir=args.tokenizer_dir,
        check_runtime=args.verify_runtime,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
