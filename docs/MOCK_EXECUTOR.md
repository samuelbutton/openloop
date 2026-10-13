# Mock executor

Implemented on October 12, 2026. Need: [N3](../NEEDS.md#n1n10).
The [source](../src/openloop/executors/mock.py) implements the existing `Executor` contract.
The [record types](../src/openloop/executors/replay.py) preserve outcomes and source references.
The [tests](../tests/executors/test_mock.py) check replay with 1 to 10,000 simulated workers.

## Recordings

`RecordedRun` contains an exact `ProcessJob` hash, a source hash, an execution duration, and a terminal state.
A successful recording can contain its original `Result`.
A failed recording cannot contain a score.
The duration must be finite and nonnegative. Unknown durations are rejected.
The optional `pool_key` declares which recordings are interchangeable draws of one candidate.
The recording hash covers all fields and replay version 2.

`RecordedRun.from_run(run, job_hash=..., duration_metric=..., pool_key=...)` captures a terminal public ledger attempt.
`duration_metric` is required, with no default, because the choices differ.
A string, such as `"process_seconds"`, names a seconds observation with the execution split.
For T1 it times the training process alone.
`Stage.EXECUTION` selects the execution-stage duration instead.
That stage can include orchestration overhead around the process.
Its source hash covers the full original run record.
The method rejects cache aliases, pending attempts, and missing durations.
The job binding is explicit. The method cannot reconstruct a native process command from ledger inputs.

## Drawing recordings

Recordings are pooled by `RecordedRun.pool`: the declared `pool_key`, or else the exact job hash.
A good declared key is candidate plus fidelity, without the seed.
A submission finds its pool through `pool_of(job)`, or through its job hash when `pool_of` is `None`.
A job with no pool raises `InvalidJobError`.

By default each new submission key draws one recording from its pool with replacement.
The draw uses an RNG seeded from the replay `seed`, the pool key, and the number of earlier draws from that pool.
The same seed gives the same draws.
Submissions of other pools, and the worker count, do not change a pool's draws.
`seed` is required for a resampling replay, so a run states it.

Resampling reuses recorded noise. It cannot create variance beyond the recorded pool.
A small pool therefore understates run-to-run noise, and a reused recording is a repeat of an old observation.
It is not a new sample and must not count as one.
Every `ReplaySnapshot` carries `recording_hash`, `pool_key`, `reuse_count`, and `resampled`.
`reuse_count` is the number of earlier submissions served that same recording.
`resampled` is true when it is positive.
`MockExecutor.resampled_submissions` counts those submissions.

`strict=True` is the opt-in exact replay.
Each pool is consumed once, in input order, and an exhausted pool raises `InvalidJobError`.
No receipt is resampled.
Do not duplicate a recording to supply more draws. Resample instead, so the reuse stays visible.

Repeated keys return the same job ID without drawing.
Changed inputs with the same key raise `SubmissionConflictError`.
Admission consumes a draw even if the replay is later cancelled or expires in the queue.
A submission rejected during validation consumes nothing.

## Virtual time and capacity

`MockExecutor(recordings, workers=1, gpu_lanes=1, seed=..., strict=False, pool_of=None)` supplies identical simulated worker slots.
Every job needs a worker.
A job marked `gpu=True` also needs a GPU lane.
A GPU job that waits for a lane holds no worker, so CPU jobs behind it still start.
Otherwise the queue is FIFO.
The real Mac has one GPU lane.
To simulate many GPU workers, set `gpu_lanes` to that number on purpose; the S2 report states the value it used.
The default UTC epoch is January 1, 2000. It is a simulation origin, not a measured date.
`now` gives simulated seconds since that epoch.

`submit`, `poll`, `cancel`, `collect`, `wait_any`, and `close` follow the [executor contract](EXECUTORS.md#contract).
Submission starts work when a worker, and a lane if needed, is free.
`poll` is a non-blocking observation. It never advances time.
A policy must wait with `collect` or `wait_any`. Sleep-polling never terminates here, because only waiting moves the clock.
`wait_any(job_ids)` advances the clock to the next terminal job among those IDs and returns its receipt.
It returns at once if one is terminal, and the smallest job ID wins ties.
`collect(job_id)` advances the clock until that job is terminal.
`advance(seconds)` processes events and returns newly terminal receipts in event order.
Other jobs can complete during any advance. Read their receipts with `poll` or `collect`.
No real-time sleep or process execution occurs.

The mock enforces queue and wall limits on the virtual clock.
Queue expiry wins a tie with a worker release.
A recording whose duration equals its wall limit completes.
A longer recording times out at the wall limit.
The default wall limit is 60 seconds, so set `Limits.wall_seconds` above the longest recording.
Cancellation releases its worker and lane immediately and records no reason, like the real executors.
Close cancels work without starting queued jobs.
Zero-duration recordings complete with `advance(0)`, `collect`, or `wait_any`.

Every `ReplaySnapshot` has `simulated=True` and its recording hash.
Its lifecycle times and durations are simulated.
`recorded_result` appears only when the successful recording completes unchanged.
A new timeout, queue expiry, or cancellation suppresses that result.
The original result keeps its metric values, observations, and artifact digests.
Those observations describe the source run, not the replay.
The mock does not recreate logs, outputs, exit codes, or RAM observations.
Fields without those observations stay absent.

## Offline example

The following source is an explicit synthetic fixture.
It does not describe a measured training run.

```python
import asyncio

from openloop.executors import MockExecutor, ProcessJob, RecordedRun, State
from openloop.ledger import Metric, Result, content_hash


async def example() -> None:
    job = ProcessJob(("not-executed",))
    result = Result({"score": Metric(0.5, "fixture units")})
    source = {"fixture": "example", "duration_seconds": 10, "result": result}
    recording = RecordedRun(job.hash, content_hash(source), 10, State.SUCCEEDED, result=result)
    executor = MockExecutor((recording,), workers=2, seed=0)
    try:
        first = await executor.submit(job, key="first")
        second = await executor.submit(job, key="second")
        receipt = await executor.wait_any([first, second])
        assert receipt.simulated
        assert receipt.recorded_result == result
        assert (await executor.collect(second)).resampled  # One recording, two draws.
        assert executor.now == 10
    finally:
        await executor.close()


asyncio.run(example())
```

## Validation and limits

The [S1 replay check](mock-replay-smoke.json) replays all 70 completed legacy-recipe recordings at 1, 10, 100, and 10,000 simulated workers.
It is labelled `"simulated": true` and records the hash of every source receipt and their combined hash.
Regenerate it, or check it for drift, with:

```sh
uv run python -m openloop.studies.mock_replay_smoke            # rewrites docs/mock-replay-smoke.json
uv run python -m openloop.studies.mock_replay_smoke --check    # exits 1 if the file differs
```

Flags: `--workers`, `--replay-seed`, `--strict`, `--gpu-lanes` (default: one per worker), `--source`, and `--output`.
The generator reads `research/s1/run-2026-10-11` read-only and uses `process_seconds`.
It pools recordings by candidate config hash and fidelity, so seeds share a pool.
The default report resamples, so its elapsed times differ from the raw recorded sums.
`--strict` reproduces the exact recorded sums.
A test regenerates the committed file and fails on drift.
The report does not measure real throughput or the shared starting recipe.
More than 70 workers cannot improve that fixed 70-job replay's simulated finish time.

Ownership, draw counters, and clock state are in memory. They do not survive a restart.
The model includes worker capacity, GPU lanes, and time limits only.
It does not model CPU quotas, RAM limits, transfer time, setup, evaluation, hardware contention, or monetary cost.
The caller must reserve experiment work through the allocation and loop contracts.
The mock does not write ledger records or run trusted evaluators.
It is not a T1 process-worker replacement: it supplies historical typed results, not worker log bytes.
Keep simulated receipts separate from native evidence and statistical sample counts.
S2 policy comparisons and native capacity checks remain planned.
