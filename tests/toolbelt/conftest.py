"""Synthetic Git objects and temporary ledgers; no project evidence is changed."""

from pathlib import Path

import pytest

from openloop.ledger import Ledger
from openloop.toolbelt import Toolbelt
from tests.executors.conftest import git


def records(value: object) -> list[dict[str, object]]:
    """Narrow JSON report arrays at the test boundary."""
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return value


def mapping(value: object) -> dict[str, object]:
    """Narrow a JSON report object at the test boundary."""
    assert isinstance(value, dict)
    return value


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Toolbelt Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "module.py").write_text(
        "raise RuntimeError('must not execute')\n"
        "class Model:\n"
        "    async def score(self):\n"
        "        return 1\n"
        "def train():\n"
        "    pass\n"
    )
    (root / "README.md").write_text("Synthetic source map fixture\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "synthetic toolbelt source")
    return root


@pytest.fixture
def toolbelt(ledger: Ledger, repository: Path, tmp_path: Path) -> Toolbelt:
    return Toolbelt(tmp_path / "ledger.sqlite3", repository)
