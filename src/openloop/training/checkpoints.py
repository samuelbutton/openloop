"""Publish backend checkpoints atomically; verify bytes before loading any state."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from openloop.ledger import canonical_json

from .contracts import AdamParams, Checkpoint, TrainingError


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(
    path: Path,
    *,
    backend: str,
    model_hash: str,
    params: AdamParams,
    step: int,
    writer: Callable[[Path], None],
) -> Checkpoint:
    """Publish a new directory once; never replace an existing checkpoint."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise TrainingError(
            "Another checkpoint writer owns this destination"
        ) from error
    os.close(descriptor)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        if path.exists():
            raise TrainingError("Checkpoint destination already exists")
        writer(temporary)
        files = {item.name: digest(item) for item in sorted(temporary.iterdir())}
        metadata = {
            "version": 1,
            "backend": backend,
            "model_hash": model_hash,
            "params": params,
            "step": step,
            "files": files,
        }
        (temporary / "manifest.json").write_text(canonical_json(metadata) + "\n")
        temporary.rename(path)
        return Checkpoint(path, digest(path / "manifest.json"))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink()


def verify_checkpoint(
    path: Path,
    *,
    backend: str,
    model_hash: str,
    params: AdamParams,
) -> int:
    """Check backend, model, optimizer settings, and every payload hash."""
    metadata = json.loads((path / "manifest.json").read_text())
    if (
        metadata.get("version") != 1
        or metadata.get("backend") != backend
        or metadata.get("model_hash") != model_hash
        or canonical_json(metadata.get("params")) != canonical_json(params)
    ):
        raise TrainingError("Checkpoint backend, model, or optimizer identity differs")
    step = metadata.get("step")
    if type(step) is not int or step < 0:
        raise TrainingError("Checkpoint step must be a nonnegative integer")
    expected_name = "state.pt" if backend == "torch" else "state.safetensors"
    if set(metadata["files"]) != {expected_name}:
        raise TrainingError("Checkpoint payload inventory differs")
    for name, expected in metadata["files"].items():
        if Path(name).name != name or digest(path / name) != expected:
            raise TrainingError("Checkpoint payload hash mismatch")
    return step
