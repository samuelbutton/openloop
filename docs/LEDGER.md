# Experiment ledger

`openloop.ledger` provides a local SQLite ledger with schema version 1.
It uses only the Python standard library.
The existing `ledger.json` remains a separate setup record.
The ledger does not import that record automatically.

## Identity and schema

`ExperimentInputs` declares the source, dependencies, data, environment, evaluator, and loop hashes.
It also declares the tokenizer, configuration, seed, verification stage, fidelity, budget, and execution settings.
Each content reference must be a full lowercase SHA-256 digest.
Hash complete source and dependency snapshots, rather than mutable paths or model aliases.
Freeze the environment probe once for a campaign; do not regenerate its benchmark for each submission.

The input hash covers canonical JSON and the schema version.
Canonical JSON uses sorted keys, fixed separators, and UTF-8.
It rejects non-finite numbers and non-string object keys.
Nested input mappings and sequences are immutable after construction.
Integer and floating-point configuration values remain distinct inputs.

| Table | Content |
| --- | --- |
| `experiment` | Input hash and canonical manifest. |
| `attempt` | Execution identity, input hash, purpose, and retry reference. |
| `submission` | Submission identity, attempt reference, hypothesis, UTC time, reuse reference, and optional request key. |
| `parent` | Submission lineage edges. Parents must already exist. |
| `event` | Ordered stage and terminal events, UTC timestamps, and immutable payloads. |
| `probe` | Baseline and replication references, tolerances, comparison time, and observed differences. |

Database triggers reject updates and deletes.
The `attempt_state` view derives current status from events.
Admission uses one write transaction for inputs, attempts, submissions, parents, and the initial queue event.
Connections use foreign keys and write-ahead logging.
Unknown schema versions and unrelated databases are rejected.
Use one `Ledger` connection per thread.

## API

| Operation | Behavior |
| --- | --- |
| `submit(inputs, ...)` | Record a hypothesis and parents. Reuse completed results or pending normal work with identical inputs. |
| `get(run_id)` | Return an immutable snapshot of the submission, attempt, inputs, events, and result. Unknown IDs raise `KeyError`. |
| `lineage(run_id)` | Return unique ancestors before descendants. Include explicit parents, retry parents, and cache sources. |
| `query(...)` | Filter by status, input hash, or purpose. Support `limit` and `offset`. Exclude reused submissions by default. |
| `record_preparation(...)` | Record actual proposal or implementation times before submission. Reject overlapping stages and unknown timezones. |
| `start_stage(run_id, stage)` | Close the current stage and start a later stage in one transaction. |
| `complete(run_id, result)` | Publish trusted typed metrics, artifact hashes, and an optional verdict. Require execution and evaluation stages. |
| `fail(run_id, reason)` | Close the active stage and preserve the failure reason. |
| `probe(run_id, runner, ...)` | Force a replication, compare its results, and persist the comparison. |
| `probes(run_id)` | Read stored probe comparisons for a baseline. |

Every submission has an ID.
Reused submissions share an `attempt_id` and contain `reused_from`.
Their hypothesis and parent references remain separate records.
They cannot change the shared attempt.
They do not create another statistical sample.
Check `reused_from` before scheduling work.

A `request_key` makes repeated admission of one request idempotent.
Reusing that key with different inputs or provenance raises `ValueError`.
Different keys can reference the same cached attempt.
`purpose="replication"` always creates a new attempt.
`purpose="retry"` requires `retry_of` to reference a failed run with identical inputs.
The retry records the failed run as a parent.
A new seed or fidelity changes the input hash.

## Stage times and recovery

Stages are proposal, implementation, queue, execution, evaluation, and decision.
Submission starts the queue stage.
`start_stage` records the current stage's finish and the next stage's start.
Completion and failure close the active stage.
Stages can be omitted when no corresponding work occurs.
The ledger does not invent proposal or implementation times.

Events store UTC timestamps with microsecond precision.
Durations use a monotonic clock within one connection's lifetime.
After reopening, durations use wall time and carry `duration_clock="wall"`.
Wall-time durations can be affected by clock changes.
`run.stages` groups the recorded events by stage.

An interrupted write transaction leaves no partial submission or terminal result.
A committed running attempt remains visible after restart.
The coordinator must reconcile the worker before recording failure or requesting a retry.
The ledger does not assume that an interrupted coordinator stopped the worker.

## Determinism probe

The probe callback receives the baseline's exact `ExperimentInputs`.
It must perform the experiment and return a trusted `Result`.
The ledger records a new replication even when a cached result exists.
Callback errors create a failed replication and propagate to the caller.

The comparison checks metric names, values, units, directions, splits, sample counts, and artifact hashes.
For a metric, the permitted difference is `atol + rtol * abs(baseline_value)`.
Both tolerances default to zero.
Timing, costs, and verdicts are excluded from the comparison.
This check measures one repeat; it does not establish statistical confidence.

Replication completion and its comparison enter the ledger in one transaction.
A mismatch disables completed-result reuse for that input hash.
Pending work can still be shared.
The drift flag persists across restarts and later matching probes.
There is no automatic reset of that flag.

## Example

This example uses a synthetic evaluator and spends no API or GPU credit.
The example hashes identify toy snapshots, not the current training setup.

```python
from openloop.ledger import ExperimentInputs, Ledger, Metric, Result, content_hash

inputs = ExperimentInputs(
    source_hash=content_hash("toy source v1"),
    dependencies_hash=content_hash("toy dependencies v1"),
    data_hash=content_hash("toy data v1"),
    environment_hash=content_hash("toy environment v1"),
    evaluator_hash=content_hash("toy evaluator v1"),
    loop_hash=content_hash("toy loop v1"),
    seed=42,
    fidelity="screen",
    budget_unit="evaluations",
    budget_amount=1,
)


def evaluate(inputs: ExperimentInputs) -> Result:
    return Result({"toy_score": Metric(inputs.seed / 100, "fraction")})


with Ledger("ledger.sqlite3") as ledger:
    run = ledger.submit(inputs, hypothesis="Test the toy evaluator")
    if run.reused_from is None:
        ledger.start_stage(run.id, "execution")
        result = evaluate(inputs)
        ledger.start_stage(run.id, "evaluation")
        ledger.complete(run.id, result)
    report = ledger.probe(run.id, evaluate)
    assert report.matches
    assert ledger.get(run.id).status == "succeeded"
    assert run.id in {ancestor.id for ancestor in ledger.lineage(report.repeat_id)}
```

## Boundaries

Only trusted orchestration and evaluator code may publish results.
This API is not an agent security boundary or a statistical decider.
The caller must verify artifact bytes before recording their hashes.
Artifact bytes and metric-series files remain outside this first ledger implementation.
Campaign budgets, worker execution, held-out access control, and migration of setup records remain separate tasks.
