"""Read-only source snapshots for loop adapters. No files are published or changed."""

import hashlib
from pathlib import Path

from openloop.ledger import content_hash


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapter_digest(*names: str) -> str:
    directory = Path(__file__).parent
    return content_hash({name: file_digest(directory / name) for name in names})
