"""Committed source inside Docker. Docker is the security boundary.

The repository and pinned image are trusted inputs. Candidate code receives
only committed regular files and explicit byte inputs, never host credentials,
home directories, evaluator files, or Docker sockets. No image is downloaded.
The candidate can write only to /tmp (RAM) and to /outputs, a host directory
that the coordinator limits and hashes but never interprets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from openloop.ledger import content_hash

from .models import (
    Artifact,
    ArtifactRole,
    ExecutorError,
    InvalidJobError,
    ProcessJob,
    Snapshot,
)
from .supervisor import (
    Command,
    Run,
    Supervisor,
    artifact,
    kill_group,
    wait_for_exit,
)

OUTPUT_POLL_SECONDS = 1.0
ROOT_LABEL = "openloop.root"


@dataclass(frozen=True)
class CommitSource:
    """A trusted repository and a full commit object ID, never a moving ref."""

    repository: Path
    commit: str

    def __post_init__(self) -> None:
        if not self.repository.is_absolute() or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", self.commit
        ):
            raise InvalidJobError(
                "Source needs an absolute repository and full commit ID"
            )


def resolve_socket(
    docker_host: str | None, home: Path, default: Path = Path("/var/run/docker.sock")
) -> Path:
    """Find the Docker socket: DOCKER_HOST (unix:// only), Desktop, then default."""
    candidates: list[Path] = []
    if docker_host and docker_host.startswith("unix://"):
        candidates.append(Path(docker_host.removeprefix("unix://")))
    candidates += [home / ".docker" / "run" / "docker.sock", default]
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
    raise ExecutorError(
        "No Docker socket found; pass socket= or start Docker. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_docker() -> Path:
    """Find the Docker client: the usual path, then the coordinator's PATH."""
    default = Path("/usr/local/bin/docker")
    if default.exists():
        return default
    found = shutil.which("docker")
    if found is None:
        raise ExecutorError("No Docker client found; pass docker=")
    return Path(found).absolute()


def scan_outputs(root: Path, limit: int) -> list[tuple[Path, int]]:
    """List regular files under root. Reject links, devices, and oversize totals."""
    files: list[tuple[Path, int]] = []
    total = 0
    pending = [root]
    while pending:
        try:
            entries = list(os.scandir(pending.pop()))
        except FileNotFoundError:
            continue  # The candidate removed a directory during the scan.
        for entry in entries:
            try:
                status = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(status.st_mode):
                pending.append(Path(entry.path))
            elif stat.S_ISREG(status.st_mode):
                total += status.st_size
                files.append((Path(entry.path), status.st_size))
            else:
                raise ExecutorError("Outputs may contain only regular files")
            if total > limit:
                raise ExecutorError("Declared outputs limit exceeded")
    return sorted(files)


def docker_command(
    job: ProcessJob,
    *,
    image: str,
    name: str,
    root_label: str,
    source: Path,
    inputs: Path,
    outputs: Path,
    network: bool,
) -> tuple[str, ...]:
    """Build the fixed container policy. Only bridge networking can be opted into."""
    for path in (source, inputs, outputs):
        if any(char in str(path) for char in (",", "\n", "\0")):
            raise InvalidJobError("Docker mount paths must not contain delimiters")
    limits = job.limits
    # No file may exceed the largest place that the candidate can write.
    file_bytes = limits.file_bytes or max(limits.scratch_bytes, limits.outputs_bytes)
    return (
        "run",
        "--rm",
        "--name",
        name,
        "--label",
        f"{ROOT_LABEL}={root_label}",
        "--pull=never",
        "--init",
        "--interactive",
        "--network",
        "bridge" if network else "none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--no-healthcheck",
        "--log-driver=none",
        "--memory",
        str(limits.memory_bytes),
        "--memory-swap",
        str(limits.memory_bytes),
        "--cpus",
        str(limits.cpus),
        "--pids-limit",
        str(limits.processes),
        "--ulimit",
        f"fsize={file_bytes}:{file_bytes}",
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,size={limits.scratch_bytes},mode=1777",
        "--mount",
        f"type=bind,src={source},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={inputs},dst=/inputs,readonly",
        "--mount",
        f"type=bind,src={outputs},dst=/outputs",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "/usr/bin/env",
        image,
        "-i",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "HOME=/tmp",
        *job.argv,
    )


class _ContainerBackend:
    def __init__(
        self,
        root: Path,
        source: CommitSource,
        *,
        image: str,
        network: bool,
        docker: Path,
        socket: Path,
    ) -> None:
        self.source = source
        self.image = image
        self.network = network
        self._docker = docker
        self._socket = socket
        self.label = hashlib.sha256(str(root).encode()).hexdigest()
        self._config = root / "docker-config"
        self._config.mkdir(mode=0o700, exist_ok=True)
        self._launched: set[str] = set()

    def name(self, job_id: str) -> str:
        return f"openloop-{job_id}"

    def _docker_command(self, *arguments: str) -> tuple[str, ...]:
        return (
            str(self._docker),
            "--config",
            str(self._config),
            "--host",
            f"unix://{self._socket}",
            *arguments,
        )

    async def _tool(
        self, *command: str, cwd: Path | None = None, stdin: bytes | None = None
    ) -> bytes:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env={
                "PATH": os.defpath,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
            },
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            async with asyncio.timeout(15):
                stdout, stderr = await process.communicate(stdin)
        except TimeoutError as error:
            kill_group(process)
            await process.wait()
            raise ExecutorError(
                "Control command timed out; check executor cleanup"
            ) from error
        except BaseException:
            kill_group(process)
            await process.wait()
            raise
        if process.returncode:
            raise ExecutorError(stderr.decode(errors="replace")[-2000:])
        return stdout

    async def _git(self, *arguments: str, stdin: bytes | None = None) -> bytes:
        return await self._tool(
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
            cwd=self.source.repository,
            stdin=stdin,
        )

    async def _remove(self, name: str) -> None:
        try:
            await self._tool(*self._docker_command("rm", "--force", name))
        except ExecutorError as error:
            if "No such container" not in str(error):
                raise

    async def prepare(self, run: Run) -> None:
        inputs = run.directory / "inputs"
        outputs = run.directory / "outputs"
        inputs.mkdir(mode=0o755)
        # The job directory is private; UID 65534 needs this one to be writable.
        outputs.mkdir(mode=0o777)
        outputs.chmod(0o777)
        volumes = await self._tool(
            *self._docker_command(
                "image",
                "inspect",
                "--format",
                "{{json .Config.Volumes}}",
                self.image,
            )
        )
        if json.loads(volumes) not in (None, {}):
            raise InvalidJobError("Sandbox images must not declare writable volumes")
        await self._checkout(run, run.directory / "source")
        for filename, data in run.job.files.items():
            destination = inputs / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(0o644)

    async def _checkout(self, run: Run, checkout: Path) -> None:
        # Reading Git objects avoids smudge filters, hooks, submodules, and LFS
        # downloads in the host. The result is a plain directory: no worktree
        # registration in the repository and no .git entry for the candidate.
        if await self._git("cat-file", "-t", self.source.commit) != b"commit\n":
            raise InvalidJobError("Sandbox source must identify a commit object")
        tree = await self._git("ls-tree", "-rzl", self.source.commit)
        files: list[tuple[str, str, str, int]] = []
        total = 0
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, raw_name = record.split(b"\t", 1)
            mode, kind, object_id, size = metadata.decode().split()
            name = os.fsdecode(raw_name)
            parts = Path(name).parts
            if mode not in ("100644", "100755") or kind != "blob":
                raise InvalidJobError("Sandbox source must contain only regular files")
            if (
                Path(name).is_absolute()
                or ".." in parts
                or ".git" in parts
                or any(part == ".env" or part.startswith(".env.") for part in parts)
            ):
                raise InvalidJobError(
                    "Protected source path is not allowed in a sandbox"
                )
            total += int(size)
            files.append((name, mode, object_id, int(size)))
        total += sum(len(data) for data in run.job.files.values())
        if total > run.job.limits.scratch_bytes:
            raise InvalidJobError(
                "Source and input bytes exceed the declared scratch limit"
            )
        objects = await self._git(
            "cat-file",
            "--batch",
            stdin="".join(f"{oid}\n" for _, _, oid, _ in files).encode(),
        )
        checkout.mkdir(mode=0o755)
        checkout.chmod(0o755)
        offset = 0
        for name, mode, _, size in files:
            header_end = objects.index(b"\n", offset)
            data_start = header_end + 1
            destination = checkout / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(objects[data_start : data_start + size])
            destination.chmod(0o755 if mode == "100755" else 0o644)
            offset = data_start + size + 1

    def command(self, run: Run) -> Command:
        job_id = run.snapshot.job_id
        # Mark before starting: a dying Docker client can leave a container.
        self._launched.add(job_id)
        return Command(
            self._docker_command(
                *docker_command(
                    run.job,
                    image=self.image,
                    name=self.name(job_id),
                    root_label=self.label,
                    source=run.directory / "source",
                    inputs=run.directory / "inputs",
                    outputs=run.directory / "outputs",
                    network=self.network,
                )
            ),
            None,
            {"PATH": os.defpath},
        )

    async def monitor(self, run: Run, process: asyncio.subprocess.Process) -> None:
        # Candidate RAM is bounded by Docker cgroups. CLI RSS is not job usage.
        # Poll the outputs directory, which no kernel limit bounds in total.
        limit = run.job.limits.outputs_bytes
        while process.returncode is None:
            await asyncio.to_thread(scan_outputs, run.directory / "outputs", limit)
            await wait_for_exit(process, OUTPUT_POLL_SECONDS)

    async def finish(self, run: Run) -> tuple[Artifact, ...]:
        def collect() -> tuple[Artifact, ...]:
            outputs = run.directory / "outputs"
            files = scan_outputs(outputs, run.job.limits.outputs_bytes)
            return tuple(artifact(path, ArtifactRole.OUTPUT) for path, _ in files)

        return await asyncio.to_thread(collect)

    async def cleanup(self, run: Run) -> None:
        # Removing the container is required even if the Docker client dies.
        # Never announce cancellation while a candidate container still runs.
        try:
            if run.snapshot.job_id in self._launched:
                await self._remove(self.name(run.snapshot.job_id))
        finally:
            for directory in ("source", "inputs"):
                if (run.directory / directory).exists():
                    shutil.rmtree(run.directory / directory)

    async def recover(self, owned: frozenset[str]) -> tuple[str, ...]:
        listing = await self._tool(
            *self._docker_command(
                "ps",
                "--all",
                "--filter",
                f"label={ROOT_LABEL}={self.label}",
                "--format",
                "{{.Names}}",
            )
        )
        keep = {self.name(job_id) for job_id in owned}
        removed = []
        for name in listing.decode().split():
            if name not in keep:
                await self._remove(name)
                removed.append(name)
        return tuple(removed)


class ContainerExecutor:
    """Run candidate commands in a pinned, local POSIX image with no network.

    network=True is an explicit coordinator policy for this executor. It grants
    bridge networking only. It never grants host networking or host file access.
    Omitted docker and socket arguments are resolved when the executor is built.
    """

    def __init__(
        self,
        root: Path,
        source: CommitSource,
        *,
        image: str,
        network: bool = False,
        docker: Path | None = None,
        socket: Path | None = None,
    ) -> None:
        if not re.fullmatch(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}", image):
            raise InvalidJobError("Sandbox images must be pinned by SHA-256")
        if (
            type(network) is not bool
            or (docker is not None and not docker.is_absolute())
            or (socket is not None and not socket.is_absolute())
        ):
            raise InvalidJobError(
                "Network policy must be boolean; Docker paths must be absolute"
            )
        docker = docker or resolve_docker()
        socket = socket or resolve_socket(os.environ.get("DOCKER_HOST"), Path.home())
        resolved = root.resolve()
        resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._backend = _ContainerBackend(
            resolved,
            source,
            image=image,
            network=network,
            docker=docker,
            socket=socket,
        )
        self._supervisor = Supervisor(resolved, self._backend)
        self.root = self._supervisor.root

    @property
    def source(self) -> CommitSource:
        return self._backend.source

    @property
    def image(self) -> str:
        return self._backend.image

    @property
    def network(self) -> bool:
        return self._backend.network

    async def submit(self, job: ProcessJob, *, key: str) -> str:
        if job.gpu or job.environment or job.cwd is not None:
            raise InvalidJobError(
                "Sandbox jobs cannot request GPU, host cwd, or environment"
            )
        digest = content_hash(
            {
                "job": job.hash,
                "commit": self.source.commit,
                "repository": str(self.source.repository),
                "image": self.image,
                "network": self.network,
            }
        )
        return self._supervisor.admit(job, key=key, input_hash=digest)

    async def poll(self, job_id: str) -> Snapshot:
        return await self._supervisor.poll(job_id)

    async def collect(self, job_id: str) -> Snapshot:
        return await self._supervisor.collect(job_id)

    async def cancel(self, job_id: str) -> Snapshot:
        return await self._supervisor.cancel(job_id)

    async def close(self) -> None:
        await self._supervisor.close()

    async def recover(self) -> tuple[str, ...]:
        """Force-remove this root's labelled containers that this instance does not own.

        Call it after a coordinator crash and before submitting work. It must
        not run while another live executor uses the same root. Containers of
        other roots are never touched.
        """
        return await self._backend.recover(self._supervisor.job_ids)
