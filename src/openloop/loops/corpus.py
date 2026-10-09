"""Read-only FineWeb membership adapter. Only train and validation can be iterated."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow.parquet as pq

from openloop.fineweb import load_manifest, verify_file
from openloop.ledger.identity import coerce_enum

from .models import ContractError


class DataSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"


@dataclass(frozen=True)
class Corpus:
    """A frozen manifest path. Reload and verify its identity before each read."""

    root: Path
    snapshot_hash: str

    @classmethod
    def load(cls, root: Path) -> Corpus:
        manifest = load_manifest(root)
        return cls(root.resolve(), manifest["snapshot_sha256"])

    def verify(self) -> None:
        manifest = load_manifest(self.root)
        if manifest["snapshot_sha256"] != self.snapshot_hash:
            raise ContractError("FineWeb snapshot changed")
        for shard in manifest["shards"]:
            verify_file(self.root / shard["local_path"], shard)
        for split in DataSplit:
            expected = manifest["splits"][split]
            verify_file(self.root / expected["path"], expected)

    def texts(self, split: DataSplit | str) -> Iterator[str]:
        """Yield exact selected texts in membership order, with digest validation."""
        split = coerce_enum(DataSplit, split, "data split")
        manifest = load_manifest(self.root)
        if manifest["snapshot_sha256"] != self.snapshot_hash:
            raise ContractError("FineWeb snapshot changed")
        membership = self.root / manifest["splits"][split]["path"]
        references = (
            row
            for batch in pq.ParquetFile(membership).iter_batches(batch_size=4096)
            for row in batch.to_pylist()
        )
        reference = next(references, None)
        previous = (-1, -1)
        for shard_index, shard in enumerate(manifest["shards"]):
            row_index = 0
            for batch in pq.ParquetFile(self.root / shard["local_path"]).iter_batches(
                batch_size=4096,
                columns=["text"],
            ):
                for text in batch.column(0).to_pylist():
                    if reference is not None and (
                        reference["shard_index"],
                        reference["row_index"],
                    ) == (shard_index, row_index):
                        if (shard_index, row_index) <= previous:
                            raise ContractError("Membership references must increase")
                        if (
                            not isinstance(text, str)
                            or hashlib.sha256(text.encode("utf-8")).digest()
                            != reference["text_sha256"]
                        ):
                            raise ContractError(
                                "Selected document differs from its digest"
                            )
                        yield text
                        previous = (shard_index, row_index)
                        reference = next(references, None)
                    row_index += 1
        if reference is not None:
            raise ContractError(
                "Membership references missing or out-of-order source rows"
            )

    def batches(
        self,
        split: DataSplit | str,
        tokenizer_batch_size: int = 128,
    ) -> Iterator[tuple[list[str], int]]:
        """Repeat frozen documents for upstream packing, without padding."""
        split = coerce_enum(DataSplit, split, "data split")
        if type(tokenizer_batch_size) is not int or tokenizer_batch_size < 1:
            raise ContractError("Tokenizer batch size must be positive")
        epoch = 1
        while True:
            batch: list[str] = []
            count = 0
            for text in self.texts(split):
                count += 1
                batch.append(text)
                if len(batch) == tokenizer_batch_size:
                    yield batch, epoch
                    batch = []
            if batch:
                yield batch, epoch
            if not count:
                raise ContractError("The selected split contains no documents")
            epoch += 1
