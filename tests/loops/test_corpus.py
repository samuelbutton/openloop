"""Read frozen memberships exactly; never expose held-out documents to the loop."""

import hashlib
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openloop.fineweb import split_for_digest
from openloop.ledger import InvalidInputError
from openloop.loops import ContractError
from openloop.loops.corpus import Corpus


def test_membership_matches_frozen_content_policy(corpus: Corpus) -> None:
    corpus.verify()
    train, val = list(corpus.texts("train")), list(corpus.texts("val"))
    assert train
    assert val
    assert not set(train) & set(val)
    for split, documents in (("train", train), ("val", val)):
        assert all(
            split_for_digest(hashlib.sha256(text.encode()).digest()) == split
            for text in documents
        )


def test_held_out_cannot_be_requested(corpus: Corpus) -> None:
    with pytest.raises(InvalidInputError, match="data split"):
        list(corpus.texts("held_out"))
    with pytest.raises(InvalidInputError, match="data split"):
        next(corpus.batches("held_out"))


def test_frozen_batches_repeat_in_order(corpus: Corpus) -> None:
    documents = list(corpus.texts("val"))
    batches = list(islice(corpus.batches("val", len(documents)), 2))
    assert batches == [(documents, 1), (documents, 2)]


def test_corrupt_source_file_is_not_silently_used(corpus: Corpus) -> None:
    with (corpus.root / "raw" / "tiny.parquet").open("ab") as stream:
        stream.write(b"corrupt fixture")
    with pytest.raises(ContractError, match="Size or SHA-256 mismatch"):
        corpus.verify()


def test_per_job_check_skips_raw_hashing_but_reading_catches_corruption(
    corpus: Corpus,
) -> None:
    shard = corpus.root / "raw" / "tiny.parquet"
    corpus.verify_for_reading()
    size = shard.stat().st_size
    # Same length and size, different text: the cheap check passes; reading catches it.
    pq.write_table(
        pa.table({"text": [f"Document {index}" for index in range(500)]}), shard
    )
    assert shard.stat().st_size == size
    corpus.verify_for_reading()
    with pytest.raises(ContractError, match="Size or SHA-256 mismatch"):
        corpus.verify()
    with pytest.raises(ContractError, match="differs from its digest"):
        list(corpus.texts("train"))


def test_per_job_check_rejects_resized_shard(
    corpus: Corpus,
) -> None:
    with (corpus.root / "raw" / "tiny.parquet").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ContractError, match="size changed"):
        corpus.verify_for_reading()


def test_per_job_check_rejects_changed_membership(corpus: Corpus) -> None:
    with (corpus.root / "splits" / "train.parquet").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ContractError, match="Size or SHA-256 mismatch"):
        corpus.verify_for_reading()
