"""Trusted sandbox smoke wrapper. Candidate code executes only inside the sandbox.

The materialized source is read-only. Each attempt copies it to scratch, applies
only permitted text replacements, and publishes those files after a smoke pass.
This is a smoke check, never an authoritative experiment evaluator.
"""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path


def smoke(
    source: Path, worktree: Path, outputs: Path, manifest: Mapping[str, object]
) -> None:
    """Apply a prevalidated coordinator manifest to a fresh scratch worktree."""
    base = manifest["base_files"]
    edits = manifest["files"]
    argv = manifest["smoke"]
    if (
        not isinstance(base, dict)
        or not isinstance(edits, dict)
        or not isinstance(argv, list)
    ):
        raise RuntimeError("Malformed coordinator manifest")
    shutil.copytree(source, worktree)
    before = {
        path.relative_to(worktree).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in worktree.rglob("*")
        if path.is_file()
    }
    for name, text in edits.items():
        if name not in base or not isinstance(text, str):
            raise RuntimeError("Edit outside permitted surface")
        path = worktree / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != base[name].encode()
        ):
            raise RuntimeError("Frozen base file differs from the request")
        path.write_bytes(text.encode())
    result = subprocess.run(
        argv,
        cwd=worktree,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(worktree.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Smoke test exited with status {result.returncode}")
    for name, digest in before.items():
        if name not in edits:
            path = worktree / name
            if (
                not path.is_file()
                or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise RuntimeError("Smoke test changed protected source")
    for name, text in edits.items():
        path = worktree / name
        if path.is_symlink() or path.read_bytes() != text.encode():
            raise RuntimeError("Smoke test changed candidate source")
        target = outputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode())


if __name__ == "__main__":
    smoke(
        Path("/workspace"),
        Path("/tmp/worktree"),
        Path("/outputs"),
        json.loads(Path("/inputs/manifest.json").read_text(encoding="utf-8")),
    )
