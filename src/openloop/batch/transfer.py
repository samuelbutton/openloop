"""File adapter for atomic, immutable return bundles on a mounted destination.

The destination must be a trusted durable filesystem. No credential or network
client runs here. Host-side SSH can fetch the bundles from a remote machine.
"""

import hashlib
import os
import tempfile
from pathlib import Path

from openloop.executors import ExecutorError
from openloop.ledger import Ledger, Status, canonical_json


class BatchError(ExecutorError):
    """Batch state, evidence, or transfer cannot be safely accepted."""


def publish(path: Path, data: bytes) -> None:
    """Publish bytes once. An identical replay is allowed; a change is rejected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise BatchError(f"Return evidence conflicts: {path}") from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def export_ledger(source: Path, destination: Path) -> bool:
    """Export complete stored rows read-only. Return whether every attempt ended."""
    if source.is_symlink() or not source.is_file():
        raise BatchError("The command did not produce a regular ledger file")
    with Ledger(source, read_only=True) as ledger:
        rows = ledger.export_rows().encode()
        runs = ledger.query_all(include_reused=True)
    publish(destination, rows)
    return bool(runs) and all(
        run.status in (Status.SUCCEEDED, Status.FAILED) for run in runs
    )


def sync_bundle(source: Path, destination: Path) -> None:
    """Copy a finished bundle, then publish a hash manifest as its commit marker.

    Resume is idempotent. A changed destination fails closed. Consumers accept
    only bundles whose manifest exists and whose file hashes match.
    """
    files: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise BatchError("Return bundles cannot contain symlinks")
        if path.is_dir():
            continue
        name = path.relative_to(source).as_posix()
        data = path.read_bytes()
        files[name] = hashlib.sha256(data).hexdigest()
        publish(destination / name, data)
    publish(
        destination / "manifest.json",
        canonical_json({"version": 1, "files": files}).encode(),
    )
