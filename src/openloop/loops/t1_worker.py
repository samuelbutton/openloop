"""Trusted MLX worker. Frozen code controls data packing and final BPB evaluation.

This worker accepts only generated constant changes. It is not a code sandbox.
MLX and upstream dependencies are imported only during an explicit training call.
"""

from __future__ import annotations

import json
import sys
import time
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from openloop.ledger import ExperimentInputs, canonical_json
from openloop.ledger.identity import freeze_object

from .models import ContractError, Job, JobOutput

if TYPE_CHECKING:
    from .t1 import T1


def verify_runtime(lock: str) -> dict[str, str]:
    """Require the Python 3.12 branch of the frozen upstream dependency lock."""
    if sys.version_info[:2] != (3, 12):
        raise ContractError("T1 requires Python 3.12")
    packages = tomllib.loads(lock).get("package", [])
    observed: dict[str, str] = {}
    for package in packages:
        if "registry" not in package.get("source", {}):
            continue
        markers = package.get("resolution-markers", [])
        if markers == ["python_full_version < '3.11'"]:
            continue
        if markers and markers != ["python_full_version >= '3.11'"]:
            raise ContractError("Unsupported upstream dependency resolution marker")
        name, expected = package["name"], package["version"]
        try:
            actual = version(name)
        except PackageNotFoundError as error:
            raise ContractError(f"Missing locked dependency: {name}") from error
        if actual != expected:
            raise ContractError(
                f"Locked dependency mismatch: {name}: {actual} != {expected}"
            )
        observed[name] = actual
    return observed


def run_training(loop: T1, job: Job) -> JobOutput:
    """Train the exact declared tokens and evaluate on the frozen validation split."""
    from .t1 import (
        EVALUATION_BATCH,
        EVALUATION_TOKENS,
        SEQUENCE_LENGTH,
        tokenizer_digest,
    )

    if loop.build_job(job.inputs) != job or job.program is None:
        raise ContractError("Worker requires the exact generated training program")
    start = time.monotonic()
    verify_runtime(loop.source.lock)
    loop.corpus.verify()
    if tokenizer_digest(loop.tokenizer_dir) != loop.spec.tokenizer_hash:
        raise ContractError("Frozen tokenizer changed")
    prepare: dict[str, object] = {"__name__": "frozen_prepare"}
    exec(compile(loop.source.prepare, "frozen/prepare.py", "exec"), prepare)
    if (
        prepare.get("MAX_SEQ_LEN") != SEQUENCE_LENGTH
        or prepare.get("EVAL_TOKENS") != EVALUATION_TOKENS
    ):
        raise ContractError("Frozen evaluator uses an incompatible evaluation budget")
    prepare["TOKENIZER_DIR"] = str(loop.tokenizer_dir)
    prepare["_document_batches"] = loop.corpus.batches
    evaluator = prepare["evaluate_bpb"]
    if not callable(evaluator):
        raise ContractError("Frozen evaluator must be callable")
    namespace = {"__name__": __name__} | {
        name: prepare[name]
        for name in (
            "MAX_SEQ_LEN",
            "TIME_BUDGET",
            "Tokenizer",
            "make_dataloader",
        )
    }
    exec(compile(job.program, "candidate/train.py", "exec"), namespace)
    training_seconds = namespace["total_training_time"]
    if not isinstance(training_seconds, (int, float)):
        raise ContractError("Training did not report its observed step durations")
    tokens = namespace["total_tokens"]
    steps = namespace["step"]
    if (
        type(tokens) is not int
        or tokens != job.inputs.budget_amount
        or type(steps) is not int
    ):
        raise ContractError("Training stopped outside its token budget")
    score: object = evaluator(
        namespace["model"], namespace["tokenizer"], EVALUATION_BATCH
    )
    if not isinstance(score, (int, float)):
        raise ContractError("Frozen evaluator did not return numeric BPB")
    return freeze_object(
        {
            "work_units": tokens,
            "steps": steps,
            "val_bpb": float(score),
            "evaluation_tokens": EVALUATION_TOKENS,
            "training_seconds": training_seconds,
            "total_seconds": time.monotonic() - start,
        }
    )


def main() -> None:
    """Read a trusted job on stdin; emit a typed JSON receipt after frozen scoring."""
    from .corpus import Corpus
    from .t1 import T1, TrainingSource

    payload = json.loads(sys.stdin.read())
    loop = T1(
        TrainingSource(**payload["source"]),
        Corpus.load(Path(payload["corpus_root"])),
        Path(payload["tokenizer_dir"]),
    )
    job = Job(ExperimentInputs(**payload["job"]["inputs"]), payload["job"]["program"])
    output = run_training(loop, job)
    print("OPENLOOP_RESULT " + canonical_json(output), flush=True)


if __name__ == "__main__":
    main()
