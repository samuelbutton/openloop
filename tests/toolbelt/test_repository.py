"""Repository navigation reads committed blobs, without executing source files."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from openloop.ledger import InvalidInputError
from openloop.toolbelt.repository import (
    MAX_SOURCE_BYTES,
    RepositoryError,
    RepositoryMap,
)
from tests.executors.conftest import git

from .conftest import records


def test_map_identifies_the_commit_and_qualified_symbols(repository: Path) -> None:
    (repository / "module.py").write_text("def dirty(): pass\n")
    (repository / "untracked.py").write_text("def invisible(): pass\n")
    before = git(repository, "status", "--porcelain")
    report = RepositoryMap(repository).query()
    assert report["commit"] == git(repository, "rev-parse", "HEAD")
    assert report["working_tree_included"] is False
    rows = records(report["items"])
    assert [row["path"] for row in rows] == ["README.md", "module.py"]
    assert rows[1]["symbols"] == [
        {"name": "Model", "kind": "class", "line": 2, "end_line": 4},
        {"name": "Model.score", "kind": "function", "line": 3, "end_line": 4},
        {"name": "train", "kind": "function", "line": 5, "end_line": 6},
    ]
    assert git(repository, "status", "--porcelain") == before


def test_symlinks_credentials_and_large_sources_are_excluded(repository: Path) -> None:
    # These are empty synthetic fixtures, never real credentials.
    (repository / ".env").write_text("")
    (repository / "credentials.py").write_text("")
    (repository / "link.py").symlink_to("credentials.py")
    (repository / "large.py").write_bytes(b"#" * (MAX_SOURCE_BYTES + 1))
    (repository / "broken.py").write_text("def invalid(:\n")
    (repository / "nested").mkdir()
    (repository / "nested" / "module.py").write_text("def nested(): pass\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "synthetic excluded paths")
    report = RepositoryMap(repository).query()
    rows = {row["path"]: row for row in records(report["items"])}
    assert not {".env", "credentials.py", "link.py"} & rows.keys()
    assert rows["large.py"]["symbols_omitted"] == "source exceeds byte limit"
    assert rows["broken.py"]["parse_error"] is True
    nested = RepositoryMap(repository).query(prefix="nested", limit=1)
    assert nested["total"] == 1
    assert records(nested["items"])[0]["path"] == "nested/module.py"


@pytest.mark.parametrize("prefix", ["..", "../outside", "/tmp", "a/../../b", "a\\b"])
def test_prefix_cannot_escape_root(repository: Path, prefix: str) -> None:
    with pytest.raises(InvalidInputError, match="within"):
        RepositoryMap(repository).query(prefix=prefix)


def test_pagination_and_missing_commit(repository: Path, tmp_path: Path) -> None:
    mapper = RepositoryMap(repository)
    first, second = mapper.query(limit=1), mapper.query(limit=1, offset=1)
    assert first["next_offset"] == 1
    assert second["next_offset"] is None
    assert records(first["items"]) + records(second["items"]) == records(
        mapper.query()["items"]
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    with pytest.raises(RepositoryError, match="commit"):
        RepositoryMap(empty).query()


def test_missing_blob_never_invokes_a_promisor_helper(repository: Path, tmp_path: Path):
    marker = tmp_path / "fetch-ran"
    helper = tmp_path / "dummy-remote"
    helper.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o700)
    git(repository, "config", "core.repositoryformatversion", "1")
    git(repository, "config", "extensions.partialClone", "origin")
    git(repository, "config", "remote.origin.promisor", "true")
    git(repository, "config", "remote.origin.url", f"ext::{helper}")
    git(repository, "config", "protocol.ext.allow", "always")
    oid = git(repository, "rev-parse", "HEAD:module.py")
    (repository / ".git" / "objects" / oid[:2] / oid[2:]).unlink()
    with pytest.raises(RepositoryError):
        RepositoryMap(repository).query(prefix="module.py")
    assert not marker.exists()
    # Positive control: plain Git invokes this local dummy helper, never a network.
    with pytest.raises(subprocess.CalledProcessError):
        git(repository, "cat-file", "blob", oid)
    assert marker.exists()


def test_non_utf8_paths_are_skipped(repository: Path) -> None:
    oid = git(repository, "hash-object", "-w", "module.py")
    # Index the blob directly: some filesystems refuse non-UTF-8 file names.
    subprocess.run(
        [
            b"git",
            b"-C",
            os.fsencode(repository),
            b"update-index",
            b"--add",
            b"--cacheinfo",
            b"100644," + oid.encode() + b",bad-\xff.py",
        ],
        check=True,
    )
    git(repository, "commit", "-qm", "synthetic non-UTF-8 path")
    report = RepositoryMap(repository).query()
    assert [row["path"] for row in records(report["items"])] == [
        "README.md",
        "module.py",
    ]
    assert report["total"] == 2


def test_git_failure_reports_the_reason(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    with pytest.raises(RepositoryError, match="accessible Git commit: fatal"):
        RepositoryMap(empty).query()
