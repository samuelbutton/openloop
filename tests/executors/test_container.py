"""Container policy and lifecycle; real Docker tests are opt-in offline."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from openloop.executors import (
    ArtifactRole,
    CommitSource,
    ContainerExecutor,
    ExecutorError,
    InvalidJobError,
    Limits,
    ProcessJob,
    State,
)
from openloop.executors.container import resolve_docker, resolve_socket
from tests.executors.conftest import git

IMAGE = "sha256:" + "a" * 64


def build(
    root: Path, source: CommitSource, docker: Path, socket: Path, **options
) -> ContainerExecutor:
    return ContainerExecutor(
        root, source, image=IMAGE, docker=docker, socket=socket, **options
    )


def label(root: Path) -> str:
    """The documented root label: SHA-256 of the resolved root path."""
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()


async def wait_file(path: Path) -> None:
    async with asyncio.timeout(5):
        while not path.exists():
            await asyncio.sleep(0.01)


def test_frozen_checkout_policy_and_cleanup(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    sentinel = tmp_path / "filter-ran"
    git(source.repository, "config", "filter.fixture.smudge", f"touch {sentinel}")
    (source.repository / "candidate.txt").write_text("uncommitted change")
    (source.repository / "untracked.txt").write_text("not an input")
    before = git(source.repository, "status", "--porcelain")
    root = tmp_path / "jobs"

    async def scenario() -> None:
        executor = build(root, source, fake_docker, socket_path)
        command = ProcessJob(("/bin/true",), files={"data": b"frozen input"})
        identifier = await executor.submit(command, key="one")
        assert await executor.submit(command, key="one") == identifier
        result = await executor.collect(identifier)
        assert result.state == State.SUCCEEDED
        assert result.peak_rss_bytes is None
        output = json.loads(result.artifacts[0].path.read_text())
        args = output["args"]
        assert args[args.index("--network") + 1] == "none"
        assert "--pull=never" in args
        assert "--read-only" in args
        assert "--cap-drop=ALL" in args
        assert "--security-opt=no-new-privileges" in args
        assert args[args.index("--user") + 1] == "65534:65534"
        assert args[args.index("--memory") + 1] == "1073741824"
        assert args[args.index("--pids-limit") + 1] == "64"
        assert args[args.index("--ulimit") + 1] == "fsize=67108864:67108864"
        assert args[args.index("--label") + 1] == f"openloop.root={label(root)}"
        mounts = output["mounts"]
        assert [m.split("dst=")[1].split(",")[0] for m in mounts] == [
            "/workspace",
            "/inputs",
            "/outputs",
        ]
        assert [m.endswith(",readonly") for m in mounts] == [True, True, False]
        assert output["source"] == "frozen source"
        assert "untracked.txt" not in output["files"]
        assert not any(".git" in name.split("/") for name in output["tree"])
        assert output["inputs"] == {"data": "frozen input"}
        assert not (root / identifier / "source").exists()
        assert not (root / identifier / "inputs").exists()
        assert (root / "removed").read_text() == f"openloop-{identifier}"
        await executor.close()

    asyncio.run(scenario())
    assert not sentinel.exists()
    assert git(source.repository, "status", "--porcelain") == before
    # Materializing the source must not register anything in the repository.
    assert not (source.repository / ".git" / "worktrees").exists()
    assert (
        git(source.repository, "worktree", "list", "--porcelain").count("worktree ")
        == 1
    )


def test_explicit_network_policy_enters_identity(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    async def scenario() -> None:
        receipts = []
        for network in (False, True):
            executor = build(
                tmp_path / str(network),
                source,
                fake_docker,
                socket_path,
                network=network,
            )
            identifier = await executor.submit(ProcessJob(("true",)), key="policy")
            result = await executor.collect(identifier)
            receipts.append(result)
            args = json.loads(result.artifacts[0].path.read_text())["args"]
            assert args[args.index("--network") + 1] == (
                "bridge" if network else "none"
            )
            await executor.close()
        assert receipts[0].input_hash != receipts[1].input_hash

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command",
    [
        ProcessJob(("true",), gpu=True),
        ProcessJob(("true",), environment={"PATH": "/bin"}),
        ProcessJob(("true",), cwd=Path("/tmp")),
    ],
)
def test_host_access_options_are_rejected(
    source: CommitSource,
    fake_docker: Path,
    socket_path: Path,
    tmp_path: Path,
    command: ProcessJob,
) -> None:
    async def scenario() -> None:
        executor = build(tmp_path / "jobs", source, fake_docker, socket_path)
        with pytest.raises(InvalidJobError):
            await executor.submit(command, key="unsafe")
        await executor.close()

    asyncio.run(scenario())


def test_mutable_image_tags_are_rejected(source: CommitSource, tmp_path: Path) -> None:
    with pytest.raises(InvalidJobError, match="pinned"):
        ContainerExecutor(tmp_path / "jobs", source, image="alpine:latest")


def test_symlinks_fail_closed_before_checkout(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    (source.repository / "link").symlink_to(tmp_path / "outside")
    git(source.repository, "add", "link")
    git(source.repository, "commit", "-qm", "synthetic symlink source")
    invalid = CommitSource(
        source.repository, git(source.repository, "rev-parse", "HEAD")
    )

    async def scenario() -> None:
        executor = build(tmp_path / "jobs", invalid, fake_docker, socket_path)
        identifier = await executor.submit(ProcessJob(("true",)), key="symlink")
        result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert "regular files" in (result.error or "")
        assert result.started_at is None
        assert result.artifacts == ()
        await executor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancel_remove_container_and_checkout(
    source: CommitSource,
    fake_docker: Path,
    socket_path: Path,
    tmp_path: Path,
    cancel: bool,
) -> None:
    async def scenario() -> None:
        root = tmp_path / "jobs"
        executor = build(root, source, fake_docker, socket_path)
        identifier = await executor.submit(
            ProcessJob(
                ("fixture-wait",), limits=Limits(wall_seconds=5 if cancel else 0.4)
            ),
            key="stop",
        )
        if cancel:
            await wait_file(root / identifier / "started")
            result = await executor.cancel(identifier)
        else:
            result = await executor.collect(identifier)
        assert result.state == (State.CANCELLED if cancel else State.TIMED_OUT)
        assert (root / "removed").read_text() == f"openloop-{identifier}"
        assert not (root / identifier / "source").exists()
        await executor.close()

    asyncio.run(scenario())


def test_missing_docker_never_falls_back_to_host(
    source: CommitSource, socket_path: Path, tmp_path: Path
) -> None:
    async def scenario() -> None:
        executor = build(
            tmp_path / "jobs", source, tmp_path / "missing-docker", socket_path
        )
        identifier = await executor.submit(
            ProcessJob(("/bin/true",)), key="unavailable"
        )
        result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert result.exit_code is None
        assert not (tmp_path / "jobs" / identifier / "source").exists()
        await executor.close()

    asyncio.run(scenario())


def test_images_with_implicit_writable_volumes_are_rejected(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    program = fake_docker.read_text().replace(
        'print("null")', "print('{\"/data\": {}}')"
    )
    fake_docker.write_text(program)

    async def scenario() -> None:
        executor = build(tmp_path / "jobs", source, fake_docker, socket_path)
        identifier = await executor.submit(ProcessJob(("true",)), key="volume")
        result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert "writable volumes" in (result.error or "")
        assert result.artifacts == ()
        await executor.close()

    asyncio.run(scenario())


def test_cleanup_failure_cannot_publish_success(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    program = fake_docker.read_text().replace(
        'elif "rm" in args:',
        'elif "rm" in args:\n    print("cleanup failed", file=sys.stderr); sys.exit(1)',
    )
    fake_docker.write_text(program)

    async def scenario() -> None:
        executor = build(tmp_path / "jobs", source, fake_docker, socket_path)
        identifier = await executor.submit(ProcessJob(("true",)), key="cleanup")
        result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert result.exit_code == 0
        assert "cleanup failed" in (result.error or "")
        assert not (tmp_path / "jobs" / identifier / "source").exists()
        await executor.close()

    asyncio.run(scenario())


def test_outputs_become_hashed_artifacts_next_to_logs(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    async def scenario() -> None:
        root = tmp_path / "jobs"
        executor = build(root, source, fake_docker, socket_path)
        identifier = await executor.submit(ProcessJob(("fixture-outputs",)), key="out")
        result = await executor.collect(identifier)
        assert result.state == State.SUCCEEDED
        logs = [a for a in result.artifacts if a.role == ArtifactRole.LOG]
        outputs = [a for a in result.artifacts if a.role == ArtifactRole.OUTPUT]
        assert [a.path.name for a in logs] == ["stdout", "stderr"][: len(logs)]
        assert {
            a.path.relative_to(root.resolve() / identifier / "outputs") for a in outputs
        } == {
            Path("result.txt"),
            Path("nested/metrics.json"),
        }
        for item in outputs:
            assert item.sha256 == hashlib.sha256(item.path.read_bytes()).hexdigest()
            assert item.size_bytes == item.path.stat().st_size
        assert (await executor.poll(identifier)).artifacts == result.artifacts
        await executor.close()

    asyncio.run(scenario())


def test_outputs_over_limit_after_exit_fail_the_job(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    async def scenario() -> None:
        executor = build(tmp_path / "jobs", source, fake_docker, socket_path)
        identifier = await executor.submit(
            ProcessJob(("fixture-big", "5000"), limits=Limits(outputs_bytes=1000)),
            key="big",
        )
        result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert result.exit_code == 0
        assert "outputs limit" in (result.error or "")
        assert all(a.role == ArtifactRole.LOG for a in result.artifacts)
        await executor.close()

    asyncio.run(scenario())


def test_outputs_over_limit_stop_a_running_container(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    async def scenario() -> None:
        root = tmp_path / "jobs"
        executor = build(root, source, fake_docker, socket_path)
        identifier = await executor.submit(
            ProcessJob(
                ("fixture-big-wait", "5000"),
                limits=Limits(outputs_bytes=1000, wall_seconds=15),
            ),
            key="big-wait",
        )
        async with asyncio.timeout(10):
            result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert "outputs limit" in (result.error or "")
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds < 10
        assert (root / "removed").read_text() == f"openloop-{identifier}"
        await executor.close()

    asyncio.run(scenario())


def test_output_symlinks_are_rejected(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    async def scenario() -> None:
        executor = build(tmp_path / "jobs", source, fake_docker, socket_path)
        identifier = await executor.submit(ProcessJob(("fixture-link",)), key="link")
        result = await executor.collect(identifier)
        assert result.state == State.FAILED
        assert "regular files" in (result.error or "")
        assert all(a.role == ArtifactRole.LOG for a in result.artifacts)
        await executor.close()

    asyncio.run(scenario())


def test_file_size_limit_is_explicit_or_tied_to_scratch_and_outputs(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    async def scenario() -> None:
        executor = build(tmp_path / "jobs", source, fake_docker, socket_path)
        seen = []
        for key, limits in (
            ("tied", Limits(scratch_bytes=1000, outputs_bytes=5000)),
            ("explicit", Limits(file_bytes=777)),
        ):
            identifier = await executor.submit(
                ProcessJob(("true",), limits=limits), key=key
            )
            result = await executor.collect(identifier)
            args = json.loads(result.artifacts[0].path.read_text())["args"]
            seen.append(args[args.index("--ulimit") + 1])
        assert seen == ["fsize=5000:5000", "fsize=777:777"]
        await executor.close()

    asyncio.run(scenario())


def test_socket_resolution_order(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".docker" / "run").mkdir(parents=True)
    desktop = home / ".docker" / "run" / "docker.sock"
    environment = tmp_path / "env.sock"
    default = tmp_path / "default.sock"
    with pytest.raises(ExecutorError, match="No Docker socket"):
        resolve_socket(None, home, default)
    default.touch()
    assert resolve_socket(None, home, default) == default
    desktop.touch()
    assert resolve_socket(None, home, default) == desktop
    assert resolve_socket("tcp://127.0.0.1:2375", home, default) == desktop
    assert resolve_socket(f"unix://{environment}", home, default) == desktop
    environment.touch()
    assert resolve_socket(f"unix://{environment}", home, default) == environment


def test_constructor_resolves_socket_from_coordinator_environment(
    source: CommitSource, fake_docker: Path, tmp_path: Path, monkeypatch
) -> None:
    socket = tmp_path / "from-env.sock"
    socket.touch()
    monkeypatch.setenv("DOCKER_HOST", f"unix://{socket}")

    async def scenario() -> None:
        executor = ContainerExecutor(
            tmp_path / "jobs", source, image=IMAGE, docker=fake_docker
        )
        identifier = await executor.submit(ProcessJob(("true",)), key="env")
        result = await executor.collect(identifier)
        args = json.loads(result.artifacts[0].path.read_text())["args"]
        assert args[args.index("--host") + 1] == f"unix://{socket}"
        await executor.close()

    asyncio.run(scenario())


def test_missing_socket_and_client_fail_at_construction(
    source: CommitSource, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("openloop.executors.container.Path.exists", lambda self: False)
    with pytest.raises(ExecutorError, match="No Docker client"):
        resolve_docker()
    with pytest.raises(ExecutorError, match="No Docker socket"):
        ContainerExecutor(
            tmp_path / "jobs", source, image=IMAGE, docker=Path("/bin/docker")
        )


def test_docker_client_falls_back_to_path(tmp_path: Path, monkeypatch) -> None:
    client = tmp_path / "docker"
    client.write_text("#!/bin/sh\n")
    client.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("openloop.executors.container.Path.exists", lambda self: False)
    assert resolve_docker() == client


def test_recover_removes_only_unowned_containers_of_this_root(
    source: CommitSource, fake_docker: Path, socket_path: Path, tmp_path: Path
) -> None:
    root = tmp_path / "jobs"
    registry = fake_docker.parent / "containers.json"

    async def scenario() -> None:
        executor = build(root, source, fake_docker, socket_path)
        identifier = await executor.submit(
            ProcessJob(("fixture-wait",), limits=Limits(wall_seconds=10)), key="live"
        )
        await wait_file(root / identifier / "started")
        mine = label(root)
        registry.write_text(
            json.dumps(
                {
                    f"openloop-{identifier}": mine,
                    "openloop-stale": mine,
                    "openloop-foreign": label(tmp_path / "other-root"),
                    "unrelated": "none",
                }
            )
        )
        assert await executor.recover() == ("openloop-stale",)
        removed = (root / "removed-all").read_text().split()
        assert removed == ["openloop-stale"]
        assert set(json.loads(registry.read_text())) == {
            f"openloop-{identifier}",
            "openloop-foreign",
            "unrelated",
        }
        await executor.cancel(identifier)
        await executor.close()

    asyncio.run(scenario())


def real_executor(root: Path, source: CommitSource) -> ContainerExecutor:
    docker = os.environ.get("OPENLOOP_TEST_DOCKER")
    socket = os.environ.get("OPENLOOP_TEST_DOCKER_SOCKET")
    return ContainerExecutor(
        root,
        source,
        image=os.environ["OPENLOOP_TEST_IMAGE"],
        docker=Path(docker) if docker else None,
        socket=Path(socket) if socket else None,
    )


@pytest.mark.skipif(
    not os.environ.get("OPENLOOP_TEST_IMAGE"),
    reason="requires a pinned cached Docker image",
)
def test_real_container_denies_network_and_host_writes(
    source: CommitSource, tmp_path: Path
) -> None:
    script = """
set -eu
test "$(id -u)" = 65534
test -z "$(ip -4 route show)"
test -z "$(ip -6 route show default)"
! nc -w 1 198.51.100.1 80 </dev/null
test ! -e /var/run/docker.sock
test ! -d /host
test ! -e /inputs/hidden
test ! -e /workspace/.git
! touch /workspace/changed
! touch /inputs/changed
! touch /root/changed
test "$(cat /inputs/data)" = frozen
test "$(cat /workspace/candidate.txt)" = 'frozen source'
test "$(cat /sys/fs/cgroup/memory.max)" = 1073741824
test "$(cat /sys/fs/cgroup/pids.max)" = 64
grep -q '^NoNewPrivs:.*1' /proc/self/status
grep -q '^CapEff:.*0000000000000000' /proc/self/status
touch /tmp/permitted
echo written > /outputs/result.txt
echo verified
"""

    async def scenario() -> None:
        executor = real_executor(tmp_path / "jobs", source)
        identifier = await executor.submit(
            ProcessJob(("/bin/sh", "-c", script), files={"data": b"frozen"}),
            key="offline",
        )
        result = await executor.collect(identifier)
        assert result.state == State.SUCCEEDED, (
            result.error,
            [(a.path.name, a.path.read_text()) for a in result.artifacts],
        )
        assert result.artifacts[0].path.read_text() == "verified\n"
        outputs = [a for a in result.artifacts if a.role == ArtifactRole.OUTPUT]
        assert [a.path.name for a in outputs] == ["result.txt"]
        assert outputs[0].path.read_text() == "written\n"
        await executor.close()

    asyncio.run(scenario())
    assert not (source.repository / "changed").exists()


@pytest.mark.skipif(
    not os.environ.get("OPENLOOP_TEST_IMAGE"),
    reason="requires a pinned cached Docker image",
)
@pytest.mark.parametrize("cancel", [False, True])
def test_real_container_stops_and_is_removed(
    source: CommitSource, tmp_path: Path, cancel: bool
) -> None:
    async def scenario() -> None:
        root = tmp_path / "jobs"
        executor = real_executor(root, source)
        identifier = await executor.submit(
            ProcessJob(
                ("/bin/sh", "-c", "echo started; sleep 20"),
                limits=Limits(wall_seconds=5 if cancel else 1),
            ),
            key="stop",
        )
        if cancel:
            async with asyncio.timeout(10):
                stdout = root / identifier / "stdout"
                while not stdout.exists() or stdout.read_bytes() != b"started\n":
                    await asyncio.sleep(0.01)
            result = await executor.cancel(identifier)
        else:
            result = await executor.collect(identifier)
        assert result.state == (State.CANCELLED if cancel else State.TIMED_OUT)
        assert not (root / identifier / "source").exists()
        docker = os.environ.get("OPENLOOP_TEST_DOCKER")
        socket = os.environ.get("OPENLOOP_TEST_DOCKER_SOCKET")
        inspector = await asyncio.create_subprocess_exec(
            docker or str(resolve_docker()),
            "--config",
            str(root / "docker-config"),
            "--host",
            "unix://"
            + (
                socket
                or str(resolve_socket(os.environ.get("DOCKER_HOST"), Path.home()))
            ),
            "inspect",
            f"openloop-{identifier}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await inspector.communicate()
        assert inspector.returncode != 0
        assert b"no such object" in stderr.lower()
        await executor.close()

    asyncio.run(scenario())
