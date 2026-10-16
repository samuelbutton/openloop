"""Host-side return verification. SSH transport stays outside the worker container."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from .plan import Batch
from .transfer import BatchError, publish


def receive(source: Path, destination: Path) -> Path:
    """Verify all bundles before publishing. Conflicts never overwrite evidence.

    The source must be a return directory for one batch from a trusted host.
    The receiver checks completeness and corruption, not remote-host honesty.
    It does not insert observations into a local ledger or replay submissions.
    """
    source = source.resolve(strict=True)
    queue = (source / "queue.json").read_bytes()
    batch = Batch.parse(json.loads(queue))
    target = destination / batch.hash
    verified: dict[str, bytes] = {"queue.json": queue}
    for job in batch.jobs:
        for suffix in (job.name, f"{job.name}/interrupted"):
            bundle = source / suffix
            manifest_path = bundle / "manifest.json"
            if not manifest_path.exists():
                continue
            data = manifest_path.read_bytes()
            manifest = json.loads(data)
            if set(manifest) != {"version", "files"} or manifest["version"] != 1:
                raise BatchError("Unsupported return manifest")
            for name, digest in manifest["files"].items():
                path = PurePosixPath(name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or str(path) != name
                    or name == "manifest.json"
                ):
                    raise BatchError("Invalid return path")
                file = bundle / name
                if any(part.is_symlink() for part in (file, *file.parents)):
                    raise BatchError("Return paths cannot contain symlinks")
                body = file.read_bytes()
                if hashlib.sha256(body).hexdigest() != digest:
                    raise BatchError(f"Return hash mismatch: {file}")
                verified[f"{suffix}/{name}"] = body
            # Manifest is published after all its files.
            verified[f"{suffix}/manifest.json"] = data
    drained = source / "drained.json"
    if drained.exists():
        data = drained.read_bytes()
        report = json.loads(data)
        if report["batch_hash"] != batch.hash or report["jobs"] != len(batch.jobs):
            raise BatchError("Drain marker does not match queue")
        if any(f"{job.name}/manifest.json" not in verified for job in batch.jobs):
            raise BatchError("Drained batch has missing return bundles")
        verified["drained.json"] = data
    for name, body in verified.items():
        publish(target / name, body)
    return target


def main() -> None:
    """Accept a staged batch fetched with SSH, without overwriting local rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(receive(args.source, args.destination))


if __name__ == "__main__":
    main()
