"""Owned child processes for reviewed jobs. Host access is trusted, not isolated.

Job ownership and idempotency live in memory. Logs live under the explicit
artifact root. This adapter never writes ledger records or authoritative scores.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

from .models import (
    Artifact,
    ExecutorError,
    InvalidJobError,
    ProcessJob,
    Snapshot,
    require_positive,
)
from .supervisor import Command, Run, Supervisor, kill_group, wait_for_exit


class _HostBackend:
    """A child process on the host, limited by a launcher and sampled for RSS."""

    def __init__(self, sample_seconds: float) -> None:
        self._sample_seconds = sample_seconds

    async def prepare(self, run: Run) -> None:
        return

    def command(self, run: Run) -> Command:
        limits = run.job.limits
        return Command(
            (
                sys.executable,
                str(Path(__file__).with_name("launcher.py")),
                str(limits.memory_bytes),
                "-"
                if limits.cpu_seconds is None
                else str(math.ceil(limits.cpu_seconds)),
                "-" if limits.file_bytes is None else str(limits.file_bytes),
                *run.job.argv,
            ),
            run.job.cwd,
            run.job.environment,
        )

    async def monitor(self, run: Run, process: asyncio.subprocess.Process) -> None:
        """Poll process-group RSS. Each sample starts one /bin/ps process."""
        while process.returncode is None:
            observer = await asyncio.create_subprocess_exec(
                "/bin/ps",
                "-A",
                "-o",
                "pgid=,rss=",
                env={"PATH": os.defpath},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, _ = await observer.communicate()
            finally:
                kill_group(observer)
                await observer.wait()
            if observer.returncode != 0:
                raise ExecutorError("Could not monitor resident memory")
            ram = sum(
                int(rss) * 1024
                for group, rss in (line.split() for line in stdout.splitlines())
                if int(group) == process.pid
            )
            run.peak_rss_bytes = max(run.peak_rss_bytes or 0, ram)
            if ram > run.job.limits.memory_bytes:
                raise ExecutorError("Declared resident memory limit exceeded")
            await wait_for_exit(process, self._sample_seconds)

    async def finish(self, run: Run) -> tuple[Artifact, ...]:
        return ()

    async def cleanup(self, run: Run) -> None:
        return


class LocalExecutor:
    """Submit reviewed POSIX commands with an empty inherited environment.

    A user-wide file lock serializes jobs marked gpu, including jobs submitted
    through another executor or coordinator process on this host.

    The memory limit is sampled every memory_sample_seconds. A job can exceed
    it between samples. On Apple silicon, RSS excludes Metal and GPU
    allocations, so this check does not bound GPU memory. Each sample costs
    about 10 ms of CPU, so short intervals disturb throughput measurements.
    """

    def __init__(self, root: Path, *, memory_sample_seconds: float = 1.0) -> None:
        require_positive(memory_sample_seconds, "memory_sample_seconds")
        self._supervisor = Supervisor(root, _HostBackend(memory_sample_seconds))
        self.root = self._supervisor.root
        self.memory_sample_seconds = memory_sample_seconds

    async def submit(self, job: ProcessJob, *, key: str) -> str:
        if not job.reviewed:
            raise InvalidJobError("Local execution requires reviewed source")
        if job.files:
            raise InvalidJobError("Local jobs use explicit cwd; files are sandbox-only")
        job = replace(job, cwd=job.cwd or Path.cwd())
        return self._supervisor.admit(job, key=key, input_hash=job.hash)

    async def poll(self, job_id: str) -> Snapshot:
        return await self._supervisor.poll(job_id)

    async def collect(self, job_id: str) -> Snapshot:
        return await self._supervisor.collect(job_id)

    async def cancel(self, job_id: str) -> Snapshot:
        return await self._supervisor.cancel(job_id)

    async def close(self) -> None:
        await self._supervisor.close()
