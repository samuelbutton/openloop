"""Real Implementer, ContainerExecutor and smoke worker in Docker. Opt-in, offline.

Set OPENLOOP_TEST_PYTHON_IMAGE to a pinned, locally cached image with `python`
(3.x) on /usr/local/bin or /usr/bin. Providers are scripted: no model is called.
"""

import asyncio
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from openloop.executors import CommitSource, ContainerExecutor
from openloop.executors.container import resolve_docker, resolve_socket
from openloop.ledger import ActivityKind, Ledger
from openloop.operators import (
    ImplementationRequest,
    Implementer,
    ModelReply,
    Patch,
)
from openloop.operators.smoke_checks import t1_smoke_command
from tests.executors.conftest import git

IMAGE = os.environ.get("OPENLOOP_TEST_PYTHON_IMAGE")
pytestmark = pytest.mark.skipif(
    not IMAGE, reason="requires OPENLOOP_TEST_PYTHON_IMAGE, a pinned cached image"
)

CHECK = (
    "python",
    "-c",
    "assert open('candidate.txt').read() == 'after'",
)


@pytest.fixture
def repository(tmp_path: Path) -> CommitSource:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Sandbox Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "candidate.txt").write_text("before")
    (root / "protected.txt").write_text("protected")
    git(root, "add", ".")
    git(root, "commit", "-qm", "synthetic sandbox source")
    return CommitSource(root, git(root, "rev-parse", "HEAD"))


@pytest.fixture
def executor(tmp_path: Path, repository: CommitSource) -> ContainerExecutor:
    assert IMAGE
    docker = os.environ.get("OPENLOOP_TEST_DOCKER")
    socket = os.environ.get("OPENLOOP_TEST_DOCKER_SOCKET")
    return ContainerExecutor(
        tmp_path / "jobs",
        repository,
        image=IMAGE,
        docker=Path(docker) if docker else None,
        socket=Path(socket) if socket else None,
    )


def leftover_containers(executor: ContainerExecutor) -> str:
    socket = os.environ.get("OPENLOOP_TEST_DOCKER_SOCKET")
    docker = os.environ.get("OPENLOOP_TEST_DOCKER")
    found = socket or str(resolve_socket(os.environ.get("DOCKER_HOST"), Path.home()))
    return subprocess.run(
        [
            docker or str(resolve_docker()),
            "--host",
            f"unix://{found}",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            "label=openloop.root",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class Scripted:
    """Replays one patch per attempt (the last repeats); records each request."""

    def __init__(self, *values: dict[str, str]) -> None:
        self.values = values
        self.requests: list[ImplementationRequest] = []

    async def __call__(self, request: ImplementationRequest) -> ModelReply[Patch]:
        self.requests.append(request)
        value = self.values[min(request.attempt, len(self.values)) - 1]
        return ModelReply(
            Patch(value), "scripted", "fixture", str(request.attempt), simulated=True
        )


def run(
    proposer,
    baseline,
    ledger: Ledger,
    executor: ContainerExecutor,
    provider,
    smoke: tuple[str, ...] = CHECK,
):
    card = asyncio.run(proposer.propose("draft", baseline=baseline.id))
    implementer = Implementer(
        executor,
        ledger,
        provider,
        base_files={"candidate.txt": "before"},
        smoke=smoke,
        simulated_only=True,
    )
    try:
        return card, asyncio.run(implementer.implement(card))
    finally:
        asyncio.run(executor.close())


def reasons(ledger: Ledger, card) -> list[str]:
    return [
        str(event.payload["reason"])
        for event in ledger.activities(subject_hash=card.hash)
        if event.kind is ActivityKind.IMPLEMENTATION_FAILURE
    ]


def test_passing_patch_publishes_only_the_edit(
    proposer, baseline, ledger: Ledger, executor: ContainerExecutor
) -> None:
    provider = Scripted({"candidate.txt": "after"})
    card, candidate = run(proposer, baseline, ledger, executor, provider)
    assert candidate is not None
    assert dict(candidate.files) == {"candidate.txt": "after"}
    success = ledger.activities(subject_hash=card.hash)[-1]
    assert success.kind is ActivityKind.IMPLEMENTATION_SUCCESS
    assert success.payload["artifacts"] == {
        "candidate.txt": hashlib.sha256(b"after").hexdigest()
    }
    published = list((executor.root / str(success.payload["executor_job"])).rglob("*"))
    outputs = [p for p in published if "outputs" in p.parts and p.is_file()]
    assert [p.name for p in outputs] == ["candidate.txt"]
    assert (executor.source.repository / "candidate.txt").read_text() == "before"
    assert leftover_containers(executor) == ""


def test_failed_smoke_is_repaired_with_the_reason(
    proposer, baseline, ledger: Ledger, executor: ContainerExecutor
) -> None:
    provider = Scripted({"candidate.txt": "bad"}, {"candidate.txt": "after"})
    card, candidate = run(proposer, baseline, ledger, executor, provider)
    assert candidate is not None
    assert len(provider.requests) == 2
    assert provider.requests[0].previous_failure is None
    retry = provider.requests[1].previous_failure
    assert retry is not None
    assert "exited with status 1" in retry or "AssertionError" in retry
    assert len(reasons(ledger, card)) == 1
    assert not any(
        event.kind is ActivityKind.IDEA_FAILURE
        for event in ledger.activities(subject_hash=card.hash)
    )
    assert leftover_containers(executor) == ""


def test_three_failures_exhaust_the_cap(
    proposer, baseline, ledger: Ledger, executor: ContainerExecutor
) -> None:
    provider = Scripted({"candidate.txt": "bad"})
    card = asyncio.run(proposer.propose("draft", baseline=baseline.id))
    implementer = Implementer(
        executor,
        ledger,
        provider,
        base_files={"candidate.txt": "before"},
        smoke=CHECK,
        simulated_only=True,
    )
    assert asyncio.run(implementer.implement(card)) is None
    assert [r.attempt for r in provider.requests] == [1, 2, 3]
    assert len(reasons(ledger, card)) == 3
    # At the cap, implement() returns None without another provider call.
    assert asyncio.run(implementer.implement(card)) is None
    assert len(provider.requests) == 3
    asyncio.run(executor.close())
    assert leftover_containers(executor) == ""


def python(code: str) -> tuple[str, ...]:
    return ("python", "-c", code)


@pytest.mark.parametrize(
    ("script", "message"),
    [
        (
            "open('/outputs/extra.txt', 'w').write('smuggled')",
            "Smoke outputs do not match the patch",
        ),
        (
            "import os; os.symlink('/etc/hosts', '/outputs/link')",
            "regular files",
        ),
        (
            "open('protected.txt', 'w').write('changed')",
            "Smoke test changed protected source",
        ),
        (
            "import os; os.remove('protected.txt')",
            "Smoke test changed protected source",
        ),
        (
            "open('candidate.txt', 'w').write('rewritten')",
            "Smoke test changed candidate source",
        ),
    ],
)
def test_malicious_smoke_is_rejected(
    proposer,
    baseline,
    ledger: Ledger,
    executor: ContainerExecutor,
    script: str,
    message: str,
) -> None:
    provider = Scripted({"candidate.txt": "after"})
    card, candidate = run(
        proposer, baseline, ledger, executor, provider, python(script)
    )
    assert candidate is None
    failures = reasons(ledger, card)
    assert len(failures) == 3
    assert all(message in reason for reason in failures), failures
    assert (executor.source.repository / "protected.txt").read_text() == "protected"
    assert leftover_containers(executor) == ""


def test_network_is_denied(
    proposer, baseline, ledger: Ledger, executor: ContainerExecutor
) -> None:
    probe = python(
        "import socket\n"
        "s = socket.socket(); s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 53))\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('network reachable')\n"
        "assert open('candidate.txt').read() == 'after'\n"
    )
    provider = Scripted({"candidate.txt": "after"})
    card, candidate = run(proposer, baseline, ledger, executor, provider, probe)
    assert candidate is not None, reasons(ledger, card)


def test_t1_static_smoke_runs_in_the_sandbox(
    proposer, baseline, ledger: Ledger, executor: ContainerExecutor, tmp_path: Path
) -> None:
    from tests.loops.conftest import TOY_TRAIN

    root = executor.source.repository
    (root / "train.py").write_text(TOY_TRAIN)
    git(root, "add", ".")
    git(root, "commit", "-qm", "train")
    source = CommitSource(root, git(root, "rev-parse", "HEAD"))
    assert IMAGE
    sandbox = ContainerExecutor(tmp_path / "t1", source, image=IMAGE)
    good = TOY_TRAIN.replace("WARMDOWN_RATIO = 0.5", "WARMDOWN_RATIO = 0.4")
    bad = TOY_TRAIN.replace("WARMDOWN_RATIO = 0.5\n", "")
    provider = Scripted({"train.py": bad}, {"train.py": good})
    card = asyncio.run(proposer.propose("draft", baseline=baseline.id))
    implementer = Implementer(
        sandbox,
        ledger,
        provider,
        base_files={"train.py": TOY_TRAIN},
        smoke=t1_smoke_command("train.py"),
        simulated_only=True,
    )
    candidate = asyncio.run(implementer.implement(card))
    asyncio.run(sandbox.close())
    assert candidate is not None
    assert "WARMDOWN_RATIO" in (provider.requests[1].previous_failure or "")
    assert leftover_containers(sandbox) == ""


def test_non_ascii_source_round_trips_exactly(
    proposer, baseline, ledger: Ledger, executor: ContainerExecutor
) -> None:
    text = "après ✓\r\nline\n"
    check = python(
        f"assert open('candidate.txt', encoding='utf-8', newline='').read() == {text!r}"
    )
    provider = Scripted({"candidate.txt": text})
    card, candidate = run(proposer, baseline, ledger, executor, provider, check)
    assert candidate is not None, reasons(ledger, card)
    success = ledger.activities(subject_hash=card.hash)[-1]
    assert success.payload["artifacts"] == {
        "candidate.txt": hashlib.sha256(text.encode()).hexdigest()
    }
