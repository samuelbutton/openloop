"""Exercise frozen memberships, exact duplicate exclusion, and corruption checks."""

import hashlib
import json
import shutil
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openloop.fineweb import (
    SPLITS,
    download_shard,
    file_hash,
    freeze,
    split_for_digest,
    verify_snapshot,
)


@pytest.mark.parametrize(
    ("bucket", "expected"),
    [
        (0, "train"),
        (8999, "train"),
        (9000, "val"),
        (9499, "val"),
        (9500, "held_out"),
        (9999, "held_out"),
    ],
)
def test_split_boundaries(bucket, expected):
    assert split_for_digest(bucket.to_bytes(8, "big") + bytes(24)) == expected


def test_download_publishes_only_verified_bytes(tmp_path, monkeypatch):
    payload = b"verified shard bytes"
    shard = {
        "path": "sample/test.parquet",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    responses = iter([b"corrupt download", payload])
    monkeypatch.setattr(
        "openloop.fineweb.urlopen", lambda *args, **kwargs: BytesIO(next(responses))
    )
    monkeypatch.setattr("openloop.fineweb.time.sleep", lambda *args: None)
    path = tmp_path / "test.parquet"
    download_shard("test", "fixed", shard, path)
    assert path.read_bytes() == payload
    assert not path.with_suffix(".part").exists()


@pytest.fixture
def corpus(tmp_path):
    groups = {name: [] for name in SPLITS}
    for index in range(10000):
        text = f"Educational document {index}: hello, 世界!"
        split = split_for_digest(hashlib.sha256(text.encode()).digest())
        if len(groups[split]) < 2:
            groups[split].append(text)
        if all(len(group) == 2 for group in groups.values()):
            break
    texts = [text for group in groups.values() for text in group]
    rows = [texts[:3] + [texts[0]], texts[3:] + [texts[0]]]
    root = tmp_path / "first"
    (root / "raw").mkdir(parents=True)
    shards = []
    for index, documents in enumerate(rows):
        path = root / "raw" / f"{index}.parquet"
        pq.write_table(pa.table({"text": documents}), path)
        shards.append(
            {
                "path": f"sample/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    source = {
        "dataset": "test",
        "revision": "fixed",
        "subset": "fixture",
        "license": "test",
        "shards": shards,
    }
    return root, source, rows


def test_frozen_splits_are_disjoint_and_references_match_source(corpus):
    root, source, rows = corpus
    manifest = freeze(source, root)
    observed = set()
    for name, split in manifest["splits"].items():
        records = pq.read_table(root / split["path"]).to_pylist()
        assert len(records) == split["documents"] == 2
        for record in records:
            digest = record["text_sha256"]
            assert digest not in observed
            observed.add(digest)
            text = rows[record["shard_index"]][record["row_index"]]
            assert hashlib.sha256(text.encode()).digest() == digest
            assert split_for_digest(digest) == name
    assert len(observed) == 6
    assert manifest["source_rows"] == [4, 4]
    assert manifest["exact_duplicates_removed"] == 2
    assert verify_snapshot(root) == manifest


def test_freeze_is_reproducible_and_does_not_regenerate(corpus, monkeypatch):
    root, source, _ = corpus
    first = freeze(source, root)
    second_root = root.parent / "second"
    shutil.copytree(root / "raw", second_root / "raw")
    assert freeze(source, second_root) == first
    monkeypatch.setattr(
        "openloop.fineweb.write_splits", lambda *args: pytest.fail("regenerated")
    )
    assert freeze(source, root) == first
    changed_source = {**source, "revision": "changed"}
    with pytest.raises(ValueError, match="Source specification differs"):
        freeze(changed_source, root)


@pytest.mark.parametrize("target", ["raw/0.parquet", "splits/train.parquet"])
def test_snapshot_rejects_corrupt_data(corpus, target):
    root, source, _ = corpus
    freeze(source, root)
    path = root / target
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_snapshot(root)


def test_snapshot_rejects_edited_manifest(corpus):
    root, source, _ = corpus
    freeze(source, root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["train"]["documents"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        verify_snapshot(root)


def test_fresh_checkout_restores_artifacts_without_changing_manifest(
    corpus, monkeypatch
):
    root, source, _ = corpus
    manifest = freeze(source, root)
    restored = root.parent / "restored"
    restored.mkdir()
    shutil.copy(root / "manifest.json", restored / "manifest.json")
    original_manifest_bytes = (restored / "manifest.json").read_bytes()

    def fetch(dataset, revision, shard, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "raw" / destination.name, destination)

    monkeypatch.setattr("openloop.fineweb.download_shard", fetch)
    assert freeze(source, restored) == manifest
    assert verify_snapshot(restored) == manifest
    assert (restored / "manifest.json").read_bytes() == original_manifest_bytes
