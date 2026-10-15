"""Explicit public, pinned model download; no credentials, inference, or training."""

import argparse
import json
from importlib import import_module
from pathlib import Path

from openloop.ledger import canonical_json

from .assets import MODEL_FILES, MODEL_ID, MODEL_REVISION, ModelSnapshot
from .contracts import TrainingError


def load_snapshot(path: Path) -> ModelSnapshot:
    """Load a host-owned snapshot record and verify its path-independent identity."""
    record = json.loads(path.read_text())
    snapshot = ModelSnapshot(
        path.parent / record["root"],
        record["files"],
        record["model_id"],
        record["revision"],
    )
    if snapshot.hash != record["snapshot_hash"]:
        raise TrainingError("Model snapshot record hash differs")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.manifest.exists():
        raise TrainingError("Model snapshot manifest already exists")
    hub = import_module("huggingface_hub")
    for name in MODEL_FILES:
        hub.hf_hub_download(
            MODEL_ID,
            name,
            revision=MODEL_REVISION,
            token=False,
            local_dir=args.output,
            cache_dir="/tmp/openloop-hf-model-cache",
        )
    snapshot = ModelSnapshot.capture(args.output)
    import os

    record = {
        "root": os.path.relpath(snapshot.root, args.manifest.parent.resolve()),
        "model_id": snapshot.model_id,
        "revision": snapshot.revision,
        "files": snapshot.files,
        "snapshot_hash": snapshot.hash,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x") as stream:
        stream.write(canonical_json(record) + "\n")
    print(
        canonical_json({"model_id": snapshot.model_id, "snapshot_hash": snapshot.hash})
    )


if __name__ == "__main__":
    main()
