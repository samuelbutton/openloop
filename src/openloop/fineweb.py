"""Download, freeze, and verify content-disjoint FineWeb-Edu splits."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import struct
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "val", "held_out")
BUCKETS = 10000
POLICY = {
    "version": 1,
    "text_hash": "SHA-256 of exact UTF-8 text; no normalization",
    "assignment": "first 8 digest bytes as big-endian uint64, modulo 10000",
    "buckets": {"train": [0, 9000], "val": [9000, 9500], "held_out": [9500, BUCKETS]},
    "duplicates": "keep first occurrence in source-shard order, then row order",
    "membership_hash": (
        "SHA-256 of ordered records: >HQ shard_index,row_index + text digest"
    ),
}
SCHEMA = pa.schema(
    [
        ("shard_index", pa.uint16()),
        ("row_index", pa.uint64()),
        ("text_sha256", pa.binary(32)),
    ]
)


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def split_for_digest(digest: bytes) -> str:
    bucket = int.from_bytes(digest[:8], "big") % BUCKETS
    for name, (start, stop) in POLICY["buckets"].items():
        if start <= bucket < stop:
            return name
    raise ValueError(f"Bucket outside policy: {bucket}")


def verify_file(path: Path, expected: dict) -> None:
    if (
        path.stat().st_size != expected["size_bytes"]
        or file_hash(path) != expected["sha256"]
    ):
        raise ValueError(f"Size or SHA-256 mismatch: {path}")


def download_shard(dataset: str, revision: str, shard: dict, path: Path) -> None:
    """Publish a download only after checking the pinned upstream hash."""
    if path.exists():
        verify_file(path, shard)
        print(f"Verified cached {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{shard['path']}"
    )
    partial = path.with_suffix(".part")
    for attempt in range(3):
        try:
            print(f"Downloading {path.name} (attempt {attempt + 1})", flush=True)
            with urlopen(url, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            verify_file(partial, shard)
            partial.replace(path)
            return
        except (OSError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def write_splits(paths: list[Path], directory: Path) -> dict:
    """Stream unique document references into immutable split files."""
    writers = {
        name: pq.ParquetWriter(directory / f"{name}.parquet", SCHEMA) for name in SPLITS
    }
    digests = {name: hashlib.sha256() for name in SPLITS}
    counts: dict[str, int] = dict.fromkeys(SPLITS, 0)
    duplicates = 0
    source_rows = []
    try:
        with sqlite3.connect(directory / "seen.sqlite") as seen:
            seen.execute("CREATE TABLE seen (hash BLOB PRIMARY KEY) WITHOUT ROWID")
            for shard_index, path in enumerate(paths):
                row_index = 0
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=4096, columns=["text"]
                ):
                    rows = {name: [] for name in SPLITS}
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str) or not text:
                            raise ValueError(
                                f"Invalid text: {path.name}, row {row_index}"
                            )
                        digest = hashlib.sha256(text.encode("utf-8")).digest()
                        inserted = seen.execute(
                            "INSERT OR IGNORE INTO seen VALUES (?)", (digest,)
                        )
                        if inserted.rowcount:
                            split = split_for_digest(digest)
                            rows[split].append((shard_index, row_index, digest))
                            digests[split].update(
                                struct.pack(">HQ", shard_index, row_index) + digest
                            )
                            counts[split] += 1
                        else:
                            duplicates += 1
                        row_index += 1
                    for name, records in rows.items():
                        if records:
                            columns = list(zip(*records, strict=True))
                            table = pa.Table.from_arrays(
                                [
                                    pa.array(values, type=field.type)
                                    for values, field in zip(
                                        columns, SCHEMA, strict=True
                                    )
                                ],
                                schema=SCHEMA,
                            )
                            writers[name].write_table(table)
                    seen.commit()
                source_rows.append(row_index)
                print(f"Indexed {path.name}: {row_index:,} source rows", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    if not all(counts.values()):
        raise ValueError("Each frozen split must contain documents")
    return {
        "source_rows": source_rows,
        "exact_duplicates_removed": duplicates,
        "splits": {
            name: {
                "path": f"splits/{name}.parquet",
                "documents": counts[name],
                "membership_sha256": digests[name].hexdigest(),
                "sha256": file_hash(directory / f"{name}.parquet"),
                "size_bytes": (directory / f"{name}.parquet").stat().st_size,
            }
            for name in SPLITS
        },
    }


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    snapshot_hash = manifest.pop("snapshot_sha256")
    if json_hash(manifest) != snapshot_hash:
        raise ValueError("Snapshot manifest hash mismatch")
    manifest["snapshot_sha256"] = snapshot_hash
    return manifest


def verify_snapshot(root: Path) -> dict:
    manifest = load_manifest(root)
    for shard in manifest["shards"]:
        verify_file(root / shard["local_path"], shard)
    for split in manifest["splits"].values():
        verify_file(root / split["path"], split)
    return manifest


def freeze(source: dict, root: Path) -> dict:
    """Create once, or restore missing artifacts against an existing frozen manifest."""
    root.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(root) if (root / "manifest.json").exists() else None
    if existing:
        if existing["source_sha256"] != json_hash(source):
            raise ValueError("Source specification differs from frozen snapshot")
        for split in existing["splits"].values():
            if (root / split["path"]).exists():
                verify_file(root / split["path"], split)
    paths = []
    shards = []
    for shard in source["shards"]:
        local_path = f"raw/{Path(shard['path']).name}"
        path = root / local_path
        download_shard(source["dataset"], source["revision"], shard, path)
        paths.append(path)
        shards.append({**shard, "local_path": local_path})
    if existing and all(
        (root / split["path"]).exists() for split in existing["splits"].values()
    ):
        return existing
    with tempfile.TemporaryDirectory(prefix="freeze-", dir=root) as staging:
        directory = Path(staging)
        membership = write_splits(paths, directory)
        if existing and any(
            existing[key] != value for key, value in membership.items()
        ):
            raise ValueError("Restored memberships differ from frozen snapshot")
        manifest = {
            "format_version": 1,
            "dataset": source["dataset"],
            "revision": source["revision"],
            "subset": source["subset"],
            "license": source["license"],
            "source_sha256": json_hash(source),
            "policy": POLICY,
            "pyarrow_version": pa.__version__,
            "shards": shards,
            **membership,
        }
        manifest["snapshot_sha256"] = json_hash(manifest)
        manifest = existing or manifest
        (root / "splits").mkdir(exist_ok=True)
        for name in SPLITS:
            destination = root / "splits" / f"{name}.parquet"
            if existing is None or not destination.exists():
                (directory / f"{name}.parquet").replace(destination)
        if existing is None:
            pending = directory / "manifest.json"
            pending.write_text(json.dumps(manifest, indent=2) + "\n")
            pending.replace(root / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/fineweb-edu"))
    parser.add_argument(
        "--verify", action="store_true", help="verify without network access"
    )
    args = parser.parse_args()
    if args.verify:
        manifest = verify_snapshot(args.root)
    else:
        source = json.loads((args.root / "source.json").read_text())
        manifest = freeze(source, args.root)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
