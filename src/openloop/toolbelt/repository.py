"""Read a configured Git commit as a map; never execute code or open worktree files.

Only regular Git blobs are eligible. Symlinks, submodules and credential paths
are excluded. This is a navigation report, not a sandbox or a secret scanner.
"""

import ast
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import cast

from openloop.ledger import InvalidInputError

from .reports import Report, page

MAX_SOURCE_BYTES = 128 * 1024
MAX_SYMBOLS = 200


class RepositoryError(Exception):
    """The configured repository cannot provide a committed source map."""


def python_symbols(source: bytes) -> Report:
    """Parse definitions and line numbers, without imports or source execution."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, UnicodeError, RecursionError):
        return {"symbols": [], "parse_error": True}
    symbols: list[Report] = []

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            scope = prefix
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = f"{prefix}.{child.name}" if prefix else child.name
                symbols.append(
                    {
                        "name": scope,
                        "kind": "class"
                        if isinstance(child, ast.ClassDef)
                        else "function",
                        "line": child.lineno,
                        "end_line": child.end_lineno,
                    }
                )
            visit(child, scope)

    try:
        visit(tree)
    except RecursionError:
        return {"symbols": [], "parse_error": True}
    return {
        "symbols": symbols[:MAX_SYMBOLS],
        "symbols_total": len(symbols),
        "symbols_truncated": len(symbols) > MAX_SYMBOLS,
    }


def _eligible(path: PurePosixPath) -> bool:
    for part in path.parts:
        name = part.lower()
        if (
            name.startswith((".env", "id_rsa", "id_ed25519"))
            or name in (".git", ".aws", ".ssh", ".gnupg")
            or "credential" in name
            or "secret" in name
            or name.endswith((".pem", ".key", ".p12", ".pfx"))
        ):
            return False
    return True


class RepositoryMap:
    """Map the configured repository's HEAD; callers cannot supply Git commands."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        actual = Path(os.fsdecode(self._git("rev-parse", "--show-toplevel")).strip())
        if actual.resolve() != self.root:
            raise InvalidInputError("Configure the repository's top-level directory")

    def _git(self, *arguments: str, stdin: bytes | None = None) -> bytes:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        try:
            result = subprocess.run(
                [
                    "git",
                    "--no-optional-locks",
                    "--no-lazy-fetch",
                    "--no-replace-objects",
                    "-C",
                    str(self.root),
                    *arguments,
                ],
                input=stdin,
                capture_output=True,
                timeout=10,
                env=environment,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RepositoryError(
                "Cannot read the configured Git repository"
            ) from error
        if result.returncode:
            reason = result.stderr.decode(errors="replace").strip()[:500]
            message = "Repository needs an accessible Git commit"
            raise RepositoryError(f"{message}: {reason}" if reason else message)
        return result.stdout

    def _blobs(self, oids: list[str]) -> list[bytes]:
        """Read blobs in one Git process, in the order requested."""
        if not oids:
            return []
        output = self._git(
            "cat-file", "--batch", stdin="".join(f"{o}\n" for o in oids).encode()
        )
        blobs: list[bytes] = []
        position = 0
        for oid in oids:
            end = output.find(b"\n", position)
            header = output[position:end].split() if end >= 0 else []
            if len(header) != 3 or header[0] != oid.encode() or header[1] != b"blob":
                raise RepositoryError("Repository map needs locally available blobs")
            start = end + 1
            stop = start + int(header[2])
            blobs.append(output[start:stop])
            position = stop + 1
        return blobs

    def query(self, *, prefix: str = "", limit: int = 50, offset: int = 0) -> Report:
        """List committed files and Python definitions in a bounded, stable page."""
        page((), limit=limit, offset=offset)
        directory = PurePosixPath(prefix)
        if directory.is_absolute() or ".." in directory.parts or "\\" in prefix:
            raise InvalidInputError("Prefix must stay within the repository")
        commit = self._git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        entries: list[Report] = []
        for record in self._git("ls-tree", "-r", "-l", "-z", commit).split(b"\0"):
            if not record:
                continue
            header, raw_path = record.split(b"\t", 1)
            mode, kind, oid, size = header.split()
            try:
                path = PurePosixPath(raw_path.decode())
            except UnicodeDecodeError:
                continue
            if (
                mode not in (b"100644", b"100755")
                or kind != b"blob"
                or not path.is_relative_to(directory)
                or not _eligible(path)
            ):
                continue
            if not size.isdigit():
                raise RepositoryError("Repository map needs locally available blobs")
            entries.append(
                {"path": str(path), "blob_hash": oid.decode(), "bytes": int(size)}
            )
        entries.sort(key=lambda item: str(item["path"]))
        result = page(entries, limit=limit, offset=offset)
        items = cast(list[Report], result["items"])
        sources: list[Report] = []
        for item in items:
            if str(item["path"]).endswith(".py"):
                if int(str(item["bytes"])) > MAX_SOURCE_BYTES:
                    item["symbols_omitted"] = "source exceeds byte limit"
                else:
                    sources.append(item)
        blobs = self._blobs([str(item["blob_hash"]) for item in sources])
        for item, blob in zip(sources, blobs, strict=True):
            item.update(python_symbols(blob))
        return result | {
            "commit": commit,
            "working_tree_included": False,
        }
