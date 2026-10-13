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
The recording hash covers all fields and replay version 1.

`RecordedRun.from_run(run, job_hash=...)` captures a terminal public ledger attempt.
Its source hash covers the full original run record.
The method rejects cache aliases, pending attempts, and missing execution durations.
The default duration is the recorded execution-stage duration.
That stage can include queue and orchestration overhead from the original runner.
Use `duration_metric="process_seconds"` to select T1's recorded process duration.
The selected observation must use seconds and the execution split.
The job binding is explicit. The method cannot reconstruct a native process command from ledger inputs.

The caller supplies a finite sequence of recordings.
Each new submission key consumes the next recording for that exact input hash.
Repeated keys return the same job ID without consuming another recording.
Changed inputs with the same key raise `SubmissionConflictError`.
Missing or exhausted recordings raise `InvalidJobError`.
Admission consumes a recording even if the replay is later cancelled or expires in the queue.
There is no implicit sampling with replacement.
Explicitly repeated recordings remain simulated copies of old evidence.

## Virtual time

`MockExecutor(recordings, workers=1)` supplies identical simulated worker slots.
Every job occupies one slot, including a job marked `gpu=True`.
The queue is FIFO. Different workers can finish at different times.
The default UTC epoch is January 1, 2000. It is a simulation origin, not a measured date.
`now` gives simulated seconds since that epoch.

`submit`, `poll`, `cancel`, `collect`, and `close` follow the [executor contract](EXECUTORS.md#contract).
Submission starts work when a slot is free.
Polling does not advance time.
`advance(seconds)` processes events and returns newly terminal receipts in event order.
`collect(job_id)` advances the shared clock until that job is terminal.
Other jobs can complete during that advance. Read their receipts with `poll` or `collect`.
No real-time sleep or process execution occurs.

The mock enforces queue and wall limits on the virtual clock.
Queue expiry wins a tie with a worker release.
A recording whose duration equals its wall limit completes.
A longer recording times out at the wall limit.
Cancellation releases its slot immediately. Close cancels work without starting queued jobs.
Zero-duration recordings complete with `advance(0)` or `collect`.

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
    executor = MockExecutor((recording, recording), workers=2)
    try:
        first = await executor.submit(job, key="first")
        second = await executor.submit(job, key="second")
        receipt = await executor.collect(first)
        assert receipt.simulated
        assert receipt.recorded_result == result
        assert (await executor.poll(second)).state == State.SUCCEEDED
        assert executor.now == 10
    finally:
        await executor.close()


asyncio.run(example())
```

## Validation and limits

The [S1 replay check](mock-replay-smoke.json) uses all 70 completed legacy-recipe recordings.
It verifies unchanged outcomes at 1, 10, 100, and 10,000 simulated workers.
The report records the source file hashes.
It does not measure real throughput or the shared starting recipe.
More than 70 workers cannot improve that fixed 70-job replay's simulated finish time.

Ownership and clock state are in memory. They do not survive a restart.
The model includes worker capacity and time limits only.
It does not model CPU quotas, RAM limits, transfer time, setup, evaluation, hardware contention, or monetary cost.
The caller must reserve experiment work through the allocation and loop contracts.
The mock does not write ledger records or run trusted evaluators.
It is not a T1 process-worker replacement: it supplies historical typed results, not worker log bytes.
Keep simulated receipts separate from native evidence and statistical sample counts.
S2 policy comparisons and native capacity checks remain planned.
