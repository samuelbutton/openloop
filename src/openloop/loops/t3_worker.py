"""Trusted mixture worker. Only train streams mix; validation stays on T1's corpus.

Each source uses an unchanged upstream packer with its own frozen train reader.
Weights allocate whole, equal-sized token microbatches, not document counts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import FunctionType
from typing import Protocol

from openloop.fineweb import POLICY, load_manifest, split_for_digest
from openloop.ledger import ExperimentInputs, canonical_json
from openloop.ledger.validation import freeze_object

from .corpus import Corpus, DataSplit
from .mixture import batch_schedule
from .models import ContractError, Job, JobOutput
from .t1 import SEQUENCE_LENGTH, T1, TrainingSource
from .t1_worker import run_training
from .t3 import T3
from .training import TrainingSetting

type PackedBatch = tuple[object, object, int]


class PackedLoader(Protocol):
    def __call__(
        self,
        tokenizer: object,
        batch_size: int,
        seq_len: int,
        split: str,
        buffer_size: int = 1000,
    ) -> Iterator[PackedBatch]: ...


def checked_batches(
    corpus: Corpus, split: str, tokenizer_batch_size: int = 128
) -> Iterator[tuple[list[str], int]]:
    """Enforce the common content-split policy without opening held-out membership."""
    for texts, epoch in corpus.batches(split, tokenizer_batch_size):
        if any(
            split_for_digest(hashlib.sha256(text.encode("utf-8")).digest()) != split
            for text in texts
        ):
            raise ContractError("Mixture document violates the frozen split policy")
        yield texts, epoch


class MixtureLoader:
    """One finite training stream and its measured source-token accounting."""

    def __init__(self, loop: T3, job: Job) -> None:
        self._loop = loop
        self._job = job
        self._allocation = loop.token_allocation(job.inputs)
        self._observed = dict.fromkeys(loop.sources, 0)
        self._started = False

    def bind(self, prepare: dict[str, object]) -> PackedLoader:
        """Bind each packer to private globals; never replace validation's packer."""
        packer = prepare.get("make_dataloader")
        if not isinstance(packer, FunctionType):
            raise ContractError("Frozen token packer must be a Python function")
        validation = self._loop.training.corpus

        def validation_batches(
            split: str, tokenizer_batch_size: int = 128
        ) -> Iterator[tuple[list[str], int]]:
            if split != DataSplit.VAL:
                raise ContractError(
                    "Evaluator may read only the fixed validation split"
                )
            return checked_batches(validation, split, tokenizer_batch_size)

        prepare["_document_batches"] = validation_batches

        def mixed_loader(
            tokenizer: object,
            batch_size: int,
            seq_len: int,
            split: str,
            buffer_size: int = 1000,
        ) -> Iterator[PackedBatch]:
            if self._started or split != DataSplit.TRAIN:
                raise ContractError("Mixture training permits one train loader only")
            self._started = True
            expected_batch = self._loop.training_config[
                TrainingSetting.DEVICE_BATCH_SIZE
            ]
            if batch_size != expected_batch or seq_len != SEQUENCE_LENGTH:
                raise ContractError("Mixture packer dimensions differ from the recipe")
            quantum = batch_size * seq_len
            counts = {
                name: tokens // quantum for name, tokens in self._allocation.items()
            }
            streams: dict[str, Iterator[PackedBatch]] = {}

            def source_reader(
                corpus: Corpus,
            ) -> Callable[[str, int], Iterator[tuple[list[str], int]]]:
                def batches(
                    split: str, tokenizer_batch_size: int = 128
                ) -> Iterator[tuple[list[str], int]]:
                    if split != DataSplit.TRAIN:
                        raise ContractError("Mixture sources expose training data only")
                    return checked_batches(corpus, split, tokenizer_batch_size)

                return batches

            for name, count in counts.items():
                if count:
                    bound = FunctionType(
                        packer.__code__,
                        {
                            **prepare,
                            "_document_batches": source_reader(
                                self._loop.sources[name]
                            ),
                        },
                        packer.__name__,
                        packer.__defaults__,
                        packer.__closure__,
                    )
                    streams[name] = bound(
                        tokenizer, batch_size, seq_len, DataSplit.TRAIN, buffer_size
                    )
            last: PackedBatch | None = None
            for name in batch_schedule(counts, seed=self._job.inputs.seed):
                last = next(streams[name])
                self._observed[name] += quantum
                yield last
            # Upstream prefetches after its last trained microbatch. Reuse that
            # batch once, so untrained prefetch cannot change the mixture quota.
            if last is not None:
                yield last
            raise ContractError("Training requested tokens beyond its mixture budget")

        return mixed_loader

    def receipt(self) -> Mapping[str, int]:
        """Require that the complete finite token allocation was delivered."""
        if self._observed != self._allocation:
            raise ContractError("Training did not consume the declared mixture tokens")
        return self._observed.copy()


def run_mixture_training(loop: T3, job: Job) -> JobOutput:
    """Verify all frozen inputs, train the finite mix, and score on T1 validation."""
    if loop.build_job(job.inputs) != job:
        raise ContractError("Worker requires the exact generated mixture program")
    for corpus in (loop.training.corpus, *loop.sources.values()):
        try:
            manifest = load_manifest(corpus.root)
        except ValueError as error:
            raise ContractError("Mixture manifest is invalid") from error
        if manifest["policy"] != POLICY:
            raise ContractError(
                "All mixture corpora must use the frozen content-split policy"
            )
        corpus.verify_for_reading()
    loader = MixtureLoader(loop, job)
    output = run_training(
        loop.training, loop.training_job(job.inputs), training_loader=loader.bind
    )
    return freeze_object({**output, "mixture_tokens": loader.receipt()})


def main() -> None:
    """Rebuild the pinned mixture from trusted stdin and emit its scoring receipt."""
    payload = json.loads(sys.stdin.read())
    training = T1(
        TrainingSource(**payload["source"]),
        Corpus.load(Path(payload["corpus_root"])),
        Path(payload["tokenizer_dir"]),
        Path(sys.executable),
    )
    loop = T3(
        training,
        {name: Corpus.load(Path(root)) for name, root in payload["sources"].items()},
        training_config=payload["training_config"],
    )
    job = Job(ExperimentInputs(**payload["job"]["inputs"]), payload["job"]["program"])
    print(
        "OPENLOOP_RESULT " + canonical_json(run_mixture_training(loop, job)), flush=True
    )


if __name__ == "__main__":
    main()
