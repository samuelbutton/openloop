# Executors

Implemented on October 9, 2026. Needs: N3, N8, and N10 in [NEEDS.md](../NEEDS.md).
The [source](../src/openloop/executors) defines `submit`, `poll`, `cancel`, and `collect`.
The [tests](../tests/executors) check lifecycle events, limits, isolation, outputs, and cleanup.

## Contract

`ProcessJob` contains the command, byte inputs, limits, and review status.
It freezes sequences and mappings.
Its hash includes every command argument, input byte digest, limit, and local setting.
This executor request hash is separate from the ledger's experiment identity.
Do not put credentials in job inputs, source snapshots, images, or command arguments.

`submit(job, key=...)` returns a job ID without waiting for completion.
The same key and inputs return the same ID.
A changed input with the same key raises `SubmissionConflictError`.
Different keys can run the same inputs again.
The ledger owns experiment caching.
Executor idempotency lasts for the life of one executor instance.
It does not survive a coordinator restart.

`poll(job_id)` returns an immutable `Snapshot`.
`collect(job_id)` waits for a terminal snapshot.
Cancelling a collector does not cancel its job.
`cancel(job_id)` stops owned work and waits for cleanup.
`close()` cancels pending jobs and rejects new submissions.
Always call `close()` in a `finally` block.

States are `queued`, `running`, `succeeded`, `failed`, `cancelled`, `timed_out`, and `expired`.
`expired` means the queue limit passed before the process started.
Each snapshot has an input hash and a UTC submission time.
Start and finish times appear when observed.
`queued_seconds` runs from submission to process start.
`elapsed_seconds` runs from process start to finish.
A job that never started has neither value.
The result includes the observed exit code and SHA-256 references to stdout and stderr.
Each `Artifact` has a `role`: `log` for stdout and stderr, `output` for files that a container wrote to `/outputs`.
Failed setup can leave no exit code or log.
The artifact root belongs to the coordinator.
Each log stream has a separate byte limit.
An overflow keeps a bounded prefix and stops the job.
The executor does not publish scores or write ledger tables.
The coordinator must pass candidate outputs to a trusted evaluator.

Default limits are initial policy choices.
Override them with a validated `Limits` record.
Optional limits are off when `None`.

| Limit | Default | Applies to |
| --- | ---: | --- |
| `queue_seconds`: wait from submission to process start | none | both |
| `wall_seconds`: run time from process start | 60 seconds | both |
| `cpu_seconds`: CPU time summed over all threads | none | local |
| `memory_bytes` | 1 GiB | both |
| `cpus`: scheduling quota | 1 | container |
| `processes` | 64 | container |
| `output_bytes`: per log stream | 1 MiB | both |
| `file_bytes`: largest single file | none locally; the larger of `scratch_bytes` and `outputs_bytes` in a container | both |
| `outputs_bytes`: total of `/outputs` | 64 MiB | container |
| `scratch_bytes`: `/tmp` and input-size allowance | 64 MiB | container |

Time spent waiting for the GPU lane counts against `queue_seconds` only.
The wall clock starts when the process starts.
In a container, source materialization counts as queue time.

## Local process

`LocalExecutor` runs reviewed POSIX commands in a new process group.
It rejects jobs without `reviewed=True`.
It inherits no environment variables.
Supply the required variables in `ProcessJob.environment`.
An omitted working directory is captured from the coordinator at submission.
The effective directory enters the request hash.
The child can access host files and the network.
Review status is a caller assertion, not an automatic code review.
Use this backend only for trusted tools and reviewed training programs.

The child receives a CPU-time limit only when `cpu_seconds` is set.
The child receives a file-size limit only when `file_bytes` is set.
Log limits never restrict the files that a job writes.
Linux also enforces an address-space limit.
macOS rejects that limit.
The coordinator therefore samples aggregate resident RAM (RSS) of the process group on both platforms.
The interval is `LocalExecutor(root, memory_sample_seconds=1.0)`.
It kills the group when a sample exceeds `memory_bytes`.
The check is sampled, so a job can exceed the limit between samples.
On Apple silicon, RSS excludes Metal and GPU allocations.
The check therefore does not bound GPU memory.
Each sample starts one `ps` process, which costs about 10 ms of CPU.
A 50 ms interval used about 21% of one core in a measurement.
Short intervals disturb throughput measurements.
`peak_rss_bytes` records the largest sampled value.
If memory monitoring fails, the executor stops the job.
Local `cpus`, `processes`, `outputs_bytes`, and `scratch_bytes` do nothing; they do not restrict trusted host access.

Jobs marked `gpu=True` share one user-wide file lock in `/tmp`.
The lock spans executor instances and coordinator processes.
Cancellation kills the process group and reaps the direct child.
Reviewed code must not detach children into new sessions.
A coordinator crash does not guarantee child termination.
Crash recovery and durable ownership remain planned.

```python
import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from openloop.executors import LocalExecutor, ProcessJob, State


async def example(root: Path) -> None:
    executor = LocalExecutor(root)
    try:
        job = ProcessJob((sys.executable, "-c", "print('ready')"), reviewed=True)
        job_id = await executor.submit(job, key="prepare-1")
        result = await executor.collect(job_id)
        assert result.state == State.SUCCEEDED
    finally:
        await executor.close()


with TemporaryDirectory() as directory:
    asyncio.run(example(Path(directory)))
```

T1 uses this backend for its trusted worker.
The caller passes a long-lived executor to T1, and T1 submits each job with the attempt identifier as its key.
It declares a one-hour wall limit from process start, 16 GiB of RAM, and 16 MiB per log stream.
It sets no queue, CPU-time, or file-size limit.
These are initial limits, not calibrated training measurements.
They enter T1's execution specification.
Executor code does not enter T1's adapter hash.
T1 records the snapshot times, peak RSS, and log digests as observations and artifacts.
The logs stay in the caller's executor root, and the caller must retain them.
Native GPU training still requires explicit approval.

## Container

`ContainerExecutor` materializes a full commit object ID into a plain directory.
It registers no Git worktree and leaves no `.git` entry in `/workspace`.
The repository is only read.
The repository must be trusted.
Uncommitted changes and untracked files are excluded.
Source files are read from Git objects without checkout filters, hooks, or submodule downloads.
Symlinks, submodules, and `.env` paths are rejected.
Explicit `files` inputs contain bytes and use normalized relative paths.
Source and input byte sizes together must fit within `scratch_bytes`.
Do not include trusted evaluator code or held-out data in this repository or these inputs.

Docker provides the security boundary.
The [Docker policy](https://docs.docker.com/reference/cli/docker/container/run/) sets these controls:

- No network by default: `--network none`.
- A local image pinned by SHA-256, with `--pull=never`.
- Read-only root, source at `/workspace`, and explicit inputs at `/inputs`.
- A bounded writable RAM filesystem at `/tmp`.
- A writable host directory at `/outputs`; see below.
- UID and GID 65534, no capabilities, and no new privileges.
- Memory, swap, CPU, process-count, file-size, wall-time, output, and log limits.
- No host home, evaluator directory, or container control socket mounted in the worker.

The image must be a trusted POSIX image with `/usr/bin/env`.
Images with declared writable volumes are rejected.
The executor overrides its entrypoint and disables health checks.
The candidate starts with only `PATH` and `HOME=/tmp`.
Image environment variables and coordinator environment variables are not inherited.
The controller uses an empty Docker configuration.
It does not load the user's Docker authentication configuration.
Docker Desktop must share the executor's artifact path with its VM.

`network=True` is an explicit policy at executor construction.
It grants Docker bridge networking.
The image, commit, repository path, and network policy enter the request hash.
Sandbox jobs reject host working directories, environment overrides, and GPU requests.
The sandbox supports CPU workloads; it does not expose native Apple Metal.
If Docker is unavailable, the job fails. There is no local fallback.
Container RAM usage is left absent; controller RSS is not candidate RAM usage.

### Outputs

The candidate can write files to `/outputs`.
The directory belongs to the job and is mode 0777, because UID 65534 must write to it.
The job directory itself is private to the coordinator's user.
The executor polls the directory size about once per second.
It kills the container when the total exceeds `outputs_bytes`.
It checks again after the container exits.
Symlinks and other non-regular files fail the job.
Subdirectories are allowed.
Files must be readable by the coordinator; a file that is mode 0600 fails the job when hashed.
After a normal exit, each file becomes an `Artifact` with `role=output`, a SHA-256 digest, and a size.
The snapshot lists logs first, then outputs sorted by path.
Timeouts, cancellation, and limit failures publish no output artifacts.
The executor never interprets outputs.
A trusted evaluator must validate them.
`/workspace`, `/inputs`, and the root filesystem stay read-only.

### Docker discovery

Without `socket=`, the executor uses the first existing path of these:

1. A `unix://` path in the coordinator's `DOCKER_HOST`. Other schemes are ignored.
2. `~/.docker/run/docker.sock`, used by Docker Desktop.
3. `/var/run/docker.sock`.

Construction fails if none exists.
Without `docker=`, the executor uses `/usr/local/bin/docker`, then `docker` from the coordinator's `PATH`.
Construction fails if neither exists.
Explicit paths are not checked at construction.
A bad explicit path fails the job.

### Recovery

Every container has the label `openloop.root=<SHA-256 of the resolved root path>`.
After a coordinator crash, call `await executor.recover()` before submitting work.
It force-removes containers with this root's label that this instance does not own.
It returns their names.
It never touches containers of other roots.
Do not call it while another live executor uses the same root.
It does not remove job directories.

Construct a sandbox with `CommitSource(repository, full_commit_id)` and a cached `sha256:` image ID or `name@sha256:` digest.
Then submit the same `ProcessJob` contract, with `files={"input": b"..."}` if needed.
The command runs with `/workspace` as its working directory.
Use `/tmp` for temporary files and `/outputs` for results.

## Validation

The default tests use CPU commands, temporary Git repositories, and a simulated Docker CLI.
They use no paid services or GPU training.
The GPU lock tests use sleeping CPU processes.
The real-container tests are opt-in with `OPENLOOP_TEST_IMAGE` set to a pinned cached image.
They check denied TCP egress, absent routes, protected mounts, a read-only `/workspace`, a writable collected `/outputs`, UID, capabilities, cgroup limits, and container removal.
No image pull is needed.
`--basetemp` must be a path that Docker shares with its VM, such as one inside the repository.
macOS `/private/tmp` may not be shared.
Run them from the repository root:

```sh
OPENLOOP_TEST_IMAGE=<name@sha256:digest> uv run pytest tests/executors/test_container.py -k real --basetemp=$PWD/.pytest_cache/sandbox
```

Set `OPENLOOP_TEST_DOCKER` and `OPENLOOP_TEST_DOCKER_SOCKET` to override discovery.
Delete `.pytest_cache/sandbox` afterwards.

The executor has no durable queue, cloud backend, mock backend, or container-escape audit.
No native T1 training was run for this change.
