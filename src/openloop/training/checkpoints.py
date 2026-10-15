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
    temporary: Path | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
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
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lock.unlink()


def verify_checkpoint(
    path: Path,
    *,
    backend: str,
    model_hash: str,
    params: AdamParams,
    sha256: str | None = None,
) -> int:
    """Check manifest digest, backend, model, optimizer settings, and payload hashes."""
    manifest = path / "manifest.json"
    try:
        observed = digest(manifest)
        metadata = json.loads(manifest.read_text())
    except (OSError, ValueError) as error:
        raise TrainingError("Checkpoint manifest is missing or malformed") from error
    if sha256 is not None and observed != sha256:
        raise TrainingError("Checkpoint manifest differs from the recorded digest")
    files = metadata.get("files") if isinstance(metadata, dict) else None
    if not isinstance(files, dict):
        raise TrainingError("Checkpoint manifest is missing or malformed")
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
    if set(files) != {expected_name}:
        raise TrainingError("Checkpoint payload inventory differs")
    for name, expected in files.items():
        payload = path / name
        if Path(name).name != name or not payload.is_file():
            raise TrainingError("Checkpoint payload is missing")
        if digest(payload) != expected:
            raise TrainingError("Checkpoint payload hash mismatch")
    return step
