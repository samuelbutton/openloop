"""Synthetic Git histories and a simulated Docker CLI. No credentials are used."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from openloop.executors import CommitSource


def git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", *arguments],
        cwd=repository,
        text=True,
        env={
            "PATH": os.defpath,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        },
    ).strip()


@pytest.fixture
def source(tmp_path: Path) -> CommitSource:
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.name", "Executor Fixture")
    git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / "candidate.txt").write_text("frozen source")
    (repository / ".gitattributes").write_text("*.txt filter=fixture\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "synthetic executor source")
    return CommitSource(repository, git(repository, "rev-parse", "HEAD"))


@pytest.fixture
def socket_path(tmp_path: Path) -> Path:
    """An explicit socket path; the fake CLI never opens it."""
    return tmp_path / "docker.sock"


@pytest.fixture
def fake_docker(tmp_path: Path) -> Path:
    """A fake Docker CLI. Containers it "runs" are registered in containers.json."""
    executable = tmp_path / "docker-fixture"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
registry = Path(sys.argv[0]).parent / "containers.json"
def containers():
    return json.loads(registry.read_text()) if registry.exists() else {}
if "run" in args:
    mounts = [args[i+1] for i, arg in enumerate(args) if arg == "--mount"]
    paths = [Path(m.split("src=",1)[1].split(",dst=",1)[0]) for m in mounts]
    source, inputs, outputs = paths
    command = args[args.index("HOME=/tmp") + 1:]
    tree = sorted(str(p.relative_to(source)) for p in source.rglob("*"))
    print(json.dumps({"args": args, "source": (source / "candidate.txt").read_text(),
                      "files": sorted(p.name for p in source.iterdir()),
                      "tree": tree,
                      "mounts": mounts,
                      "inputs": {p.name:p.read_text() for p in inputs.iterdir()}}),
          flush=True)
    if "fixture-wait" in args:
        (source.parent / "started").write_text("ready")
        time.sleep(20)
    if command[0] == "fixture-outputs":
        (outputs / "result.txt").write_text("ok")
        (outputs / "nested").mkdir()
        (outputs / "nested" / "metrics.json").write_text("{}")
    if command[0] == "fixture-link":
        (outputs / "link").symlink_to("/etc/hosts")
    if command[0] in ("fixture-big", "fixture-big-wait"):
        (outputs / "big").write_bytes(b"x" * int(command[1]))
        if command[0] == "fixture-big-wait":
            time.sleep(20)
elif "rm" in args:
    config = Path(args[args.index("--config") + 1])
    (config.parent / "removed").write_text(args[-1])
    with (config.parent / "removed-all").open("a") as stream:
        stream.write(args[-1] + "\\n")
    kept = {k: v for k, v in containers().items() if k != args[-1]}
    registry.write_text(json.dumps(kept))
elif "ps" in args:
    label = args[args.index("--filter") + 1].removeprefix("label=")
    for name, value in containers().items():
        if f"openloop.root={value}" == label:
            print(name)
elif "inspect" in args:
    print("null")
"""
    )
    executable.chmod(0o755)
    return executable
