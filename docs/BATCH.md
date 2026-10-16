# Finite container batches

The batch runner serves N3, N5, and N10. It runs a fixed list of reviewed commands.
It does not allocate cloud machines or schedule new experiments.
Each command is a trusted coordinator. Candidate isolation remains the
[container executor's](EXECUTORS.md) responsibility.

## Build

```sh
docker build -t openloop-batch:local .
docker image inspect openloop-batch:local --format '{{.Id}}'
```

Use the returned `sha256:...` image ID for execution.
The [Dockerfile](../Dockerfile) pins Python 3.12.13 and uv 0.11.29 by registry
manifest digest. It pins `procps` and `libproc2-0` at Debian `2:4.0.2-3`.
Python runtime dependencies come from the committed `uv.lock`, with `--locked`.
The default image installs PyArrow 23.0.1 and no development group.
The build context has an [allowlist](../.dockerignore). It excludes credentials,
data, local ledgers, and research artifacts.
Source is copied into the image. Rebuild after source changes.
The runtime uses the interpreter in `/app/.venv` and has no dependency downloads.

`--build-arg TRAINING_TORCH=1` adds the locked optional PyTorch group.
This option does not port T1 or T2 to CUDA. Their native adapters still use MLX
on macOS. The batch image and launcher provide CPU infrastructure. CUDA, NVIDIA
runtime setup, native GPU workloads, and RunPod shutdown remain unverified.

## Queue contract

Save this CPU-only wiring check as `queue.json`:

```json
{
  "version": 1,
  "image": "sha256:IMAGE_ID",
  "jobs": [
    {
      "name": "t0-smoke",
      "argv": ["python", "-m", "openloop.batch.smoke"],
      "wall_seconds": 60,
      "memory_bytes": 536870912
    }
  ]
}
```

Replace `sha256:IMAGE_ID` with the full built image ID before launch.
The launcher and queue must name the same image. The image ID feeds the batch
hash, so a rebuild cannot silently reuse another image's command receipts.
T0 is simulated. This command checks wiring and makes no research claim.
The parser rejects duplicate names, unknown fields, invalid paths, and invalid
limits before execution. Queue content determines the batch hash.
A changed queue gets separate state. Do not edit a queue to retry an interrupted
study without first reconciling its attempts and obtaining the required approval.

Commands receive an argument vector, with no shell expansion.
The environment contains explicit runtime paths, offline Hugging Face settings,
and these two paths:

| Variable | Contract |
| --- | --- |
| `OPENLOOP_LEDGER` | Write the job's SQLite ledger here through the public `Ledger` API. |
| `OPENLOOP_OUTPUT` | Job working directory. Put local reports and checkpoint files here. |

A study entry point must accept these paths. Existing study CLIs need a wrapper
if they choose different output paths. The batch runner does not rewrite them.
Each job has a separate ledger. Cross-job cache lookup and cross-job lineage are
not provided. A command can run a complete study in its own shared ledger.
Freeze and verify source, data, seeds, and scientific budgets in that command.
Wall and memory limits are process controls, not scientific token budgets.
The outer command does not hold the GPU lane. Inner loop executors own that lane.
Commands must join their own workers before exit.

## Drain and return

Create separate state and return directories on durable storage:

```sh
mkdir -p batch-state batch-return
scripts/run-batch.sh sha256:IMAGE_ID queue.json batch-state batch-return
```

Replace `sha256:IMAGE_ID` with the full image ID from the build.
The launcher runs under the host's non-root UID and GID.
For a root SSH session, set `OPENLOOP_WORKER_USER=10001:10001` and first give that
UID write access to both mounted directories. The image also defaults to this UID.
The queue file must be readable by that UID.

The launcher disables network access, drops capabilities, uses a read-only root
filesystem, limits process count, and mounts no credentials or Docker socket.
Its defaults are 8 GiB container RAM and four CPUs. Set
`OPENLOOP_CONTAINER_MEMORY` and `OPENLOOP_CONTAINER_CPUS` before launch to change them.
The existing local executor enforces each command's wall limit, process-group
cancellation, log limit, and sampled RSS limit. See its documented limitations.
A container resource limit can stop the coordinator before it returns evidence.
Durable mounted state is required for recovery.

The worker takes an exclusive batch lock. It drains jobs in queue order.
It exports and syncs a job before admitting the next job.
Command failures remain in receipts and do not prevent later jobs from running.
A missing, empty, invalid, or unfinished ledger makes the command unsuccessful.
A return conflict or transfer error stops admission.
Exit 0 means all commands succeeded, all their attempts ended, and return sync
completed. Exit 1 means at least one command was unsuccessful.
A terminal failed experiment can be valid evidence in a successful command.
`drained.json` is written only after every job's bundle has returned.
The container exits when the queue drains. This does not stop a rented pod.

Returned files are under `batch-return/<batch-hash>/`:

- `queue.json`: the frozen command queue.
- `<job>/ledger-rows.json`: a versioned export of every stored ledger table.
- `<job>/receipt.json`: command outcome and observed executor times and RAM.
- `<job>/executor/`: executor logs.
- `<job>/ledger-error.json`: an export failure, when present.
- `<job>/manifest.json`: SHA-256 for each returned file. Published last.
- `drained.json`: final batch outcome.

Rows retain original IDs, events, timestamps, cache references, probes,
activities, and derivations. Export uses one read transaction through
`Ledger.export_rows()`. It opens job ledgers read-only.
Return exports are archives, not new observations or submissions.
They are not merged into an existing SQLite ledger. Keep the original job ledger
and output volume for native queries and checkpoint recovery.
Checkpoint and data bytes are not included in the return bundle. Their ledger
hashes remain references. Arrange separate artifact transfer before removing a pod.
Return rows can contain held-out results. Give the return directory the same
access policy as the original study ledger.

## Remote sync to the Mac

The worker syncs to a host-mounted return directory. If the worker runs on another
machine, fetch one returned batch through an existing host SSH connection:

```sh
mkdir -p research/batch-return
scripts/sync-batch.sh openloop-pod /network-volume/batch-return/BATCH_HASH research/batch-return
```

`openloop-pod` is an SSH config alias that the operator configures on the Mac.
The host needs `scp`, uv, and this repository's installed environment.
The remote host needs SSH file access. No SSH client or credential is in the image.
The script stages the transfer, then calls `openloop.batch.receive`.
The receiver checks file hashes before publishing. Repeated transfers are
idempotent. Different existing bytes cause an error. No evidence is overwritten.
A partially copied bundle without its manifest is not accepted as complete.
Hash verification detects corruption; it does not attest to remote-host honesty.
The script can fetch partial progress while the batch is running.

## Restart and interruption

Keep the same state directory and queue when restarting.
A completed receipt causes sync replay, without another command execution.
SIGINT or SIGTERM cancels the active process group and exports available evidence.
After SIGKILL, a started job without a receipt blocks further admission.
The restarted worker exports its available ledger into an `interrupted` bundle.
A running ledger attempt remains running. The batch runner does not invent a
failure time, result, or retry. Stop and inspect orphan processes before recovery.
Use the study coordinator to record reconciliation and explicit retry lineage.
The batch runner does not resume an ambiguous command automatically.

[Batch tests](../tests/batch/test_worker.py) check real CPU commands, failures,
exclusive ownership, sync conflicts, repeat syncs, SIGTERM, and SIGKILL.
[Export tests](../tests/ledger/test_export.py) check read-only row preservation.
No cloud training or paid service is needed for these tests.
