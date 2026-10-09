"""Read frozen memberships exactly; never expose held-out documents to the loop."""

import hashlib
from itertools import islice

import pytest

from openloop.fineweb import split_for_digest
from openloop.ledger import InvalidInputError
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
    with pytest.raises(ValueError, match="Size or SHA-256 mismatch"):
        corpus.verify()
