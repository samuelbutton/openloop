# Experiment ledger

`openloop.ledger` provides a local SQLite ledger with storage schema version 3.
It uses only the Python standard library.
The existing `ledger.json` remains a separate setup record.
The ledger does not import that record automatically.

## Identity and schema

`ExperimentInputs` declares the source, dependencies, data, environment, evaluator, and loop hashes.
It also declares the tokenizer, configuration, seed, phase, fidelity, budget, reproducibility, and execution settings.
Each content reference must be a full lowercase SHA-256 digest.
Hash complete source and dependency snapshots, rather than mutable paths or model aliases.
For `environment_hash`, hash the probe's `environment` field.
Do not hash its `calibration` field, because that field contains a timestamp and benchmark measurements.

The phase names the verification phase, such as `screen` or `held_out`.
A stage is a step of one attempt: queue, execution, or evaluation.
Reproducibility is `deterministic` when identical inputs are expected to give identical results.
It is `noisy` when repeats are expected to vary, for example from GPU nondeterminism.

The input hash covers canonical JSON and `IDENTITY_VERSION`.
Bump `IDENTITY_VERSION` only when the canonical encoding or a field's meaning changes.
That change alters every hash.
The storage `SCHEMA_VERSION` does not enter any hash.
Canonical JSON uses sorted keys, fixed separators, and UTF-8.
It rejects non-finite numbers and non-string object keys.
Nested input mappings and sequences are immutable after construction.
Integer and floating-point configuration values remain distinct inputs.

The candidate hash covers the identity version, source hash, dependencies hash, and configuration.
It ignores the seed, fidelity, phase, data, evaluator, and other experiment settings.
Use it to group the samples of one candidate across seeds.

`openloop.ledger.validation` is the public home of the shared validators and the `JSONValue` type.
It provides `check_digest`, `check_text`, `check_positive`, `coerce_enum`, `finite_number`, `plain`, `freeze`, and `freeze_object`.
Each raises `InvalidInputError`.
Loops and other callers import them from there, not from `identity`.

A `Result` holds `metrics`, which are the measurements being judged.
It also holds `observations`, which are operational measurements such as timing and throughput.
`compare_results` and decisions ignore observations, so repeated runs do not differ only by speed.
Results published before observations existed read back with none.

Allowed values come from enums: `Purpose`, `Status`, `Stage`, `PreparationStage`, `Direction`, and `Reproducibility`.
Methods accept enum members or plain strings.
SQL `CHECK` clauses are built from the same enums.

| Table | Content |
| --- | --- |
| `experiment` | Input hash, candidate hash, and canonical manifest. |
| `attempt` | Execution identity, input hash, purpose, and retry reference. |
| `submission` | Submission identity, attempt reference, hypothesis, UTC time, reuse reference, and optional request key. |
| `parent` | Submission lineage edges. Parents must already exist. |
| `event` | Ordered attempt events with an optional stage, UTC timestamps, and immutable payloads. |
| `preparation` | Proposal and implementation times for one submission. |
| `probe` | Baseline and replication references, tolerances, comparison time, observed differences, and cache invalidation. |
| `activity` | Append-only hypothesis, operator, implementation, binding, idea-failure, hypothesis-outcome, and spend records. These are not experiment samples. An index on `(subject_hash, sequence)` serves per-subject reads. |

Database triggers reject updates and deletes.
The `attempt_state` view derives current status from events.
Admission uses one write transaction for inputs, attempts, submissions, parents, and the initial queue event.
Connections use foreign keys and write-ahead logging.
Schema version 3 is unreleased, so its definition changes in place.

## Opening, migration, and frozen evidence

`Ledger(path, *, migrate=False, read_only=False)` never upgrades a schema implicitly.

| Open mode | Version 2 database | Version 3 database |
| --- | --- | --- |
| Default | Opened unchanged. Experiment and run APIs work. Activity APIs raise `SchemaError`, which tells the caller to pass `migrate=True` on a writable working copy. | Opened normally. |
| `migrate=True` | Upgraded to version 3 in one write transaction. Existing run records and identity hashes are unchanged. | Opened normally. |
| `read_only=True` | Opened through SQLite `mode=ro`. No DDL, no write, and no journal-mode change. Activity APIs raise `SchemaError`. | Opened read-only. Every write raises `ReadOnlyError`. |

`read_only` and `migrate` together raise `InvalidInputError`.
A read-only open of a missing file fails and creates nothing.
A new empty path is initialized at version 3 with write-ahead logging, as before.

Use `read_only=True` for every frozen evidence ledger, such as `research/**/ledger.sqlite3`.
Never migrate evidence in place; copy it first and migrate the copy.
A read-only open leaves the database file byte-identical, verified by SHA-256 on a copy of a real version 2 ledger.
A WAL database can still gain empty `-wal` and `-shm` sidecar files, because SQLite needs the shared-memory index to read.
A non-read-only open does not change the schema or `user_version` of a version 2 file.
It also skips the journal-mode change for existing files.
It still takes a read transaction and can create the same sidecars.
Closing the last writable connection to a WAL database checkpoints any pending `-wal` frames into the main file, which changes its bytes.
A database that was killed mid-write can therefore change on a writable open.
Any write through a writable connection changes the file.
Unrelated databases and unsupported schema versions raise `SchemaError`.
The activity kinds `hypothesis_outcome`, `spend_reservation`, and `spend_settlement` have structural invariants in the ledger.
An outcome (`verified`, `discarded`, `inconclusive`, or `screen_dropped`) names its hypothesis record as `parent_sequence`, appears once per hypothesis, and carries a `decision_hash` digest unless it is `screen_dropped`.
A reservation has no parent.
A settlement references a reservation of the same subject as `parent_sequence`, and each reservation is settled once.
The ledger does not read amounts, limits, or prices; the operators layer owns that policy.
There is no migration from the pre-release version 1 schema.
The [operator guide](OPERATORS.md#failure-records) describes `record_activity`, `activities`, and atomic bounded repair admission.
Use one `Ledger` connection per thread.

## Errors

| Exception | Raised for |
| --- | --- |
| `LedgerError` | Base class for all ledger failures. |
| `InvalidInputError` | Malformed values and values outside an enum. Also a `ValueError`. |
| `NotFoundError` | Unknown run IDs and missing parents. Also a `LookupError`. |
| `ConflictError` | A request key reused with different provenance, or a preparation stage recorded twice. |
| `InvalidTransitionError` | Stage order, terminal runs, reused submissions, retries of runs that did not fail, and probes of incomplete baselines. |
| `SchemaError` | Unrelated databases, unsupported schema versions, and activity calls on an unmigrated version 2 database. |
| `ReadOnlyError` | A write through a `read_only=True` ledger. |

## API

| Operation | Behavior |
| --- | --- |
| `submit(inputs, ...)` | Record a hypothesis and parents. Reuse completed results or pending normal work with identical inputs. |
| `get(run_id)` | Return an immutable snapshot of the submission, attempt, inputs, preparations, events, and result. |
| `lineage(run_id)` | Return unique ancestors before descendants. Include explicit parents, retry parents, and cache sources. |
| `query_all(...)` | Return every run matching the filters (status, input hash, candidate hash, or purpose) in submission order, from one read snapshot. Exclude reused submissions by default. Use it for any complete-evidence view, such as an allocator snapshot. |
| `query(limit=..., offset=...)` | Return one page of the same ordered result. `limit` is a required keyword with no default, so a caller can never be truncated by a hidden cap. A page shorter than `limit` is the last. |
| `record_preparation(run_id, stage, ...)` | Record actual proposal or implementation times for a submission. Reject overlapping stages and unknown timezones. |
| `start_stage(run_id, stage)` | Close the current stage and start a later stage in one transaction. |
| `complete(run_id, result)` | Publish trusted typed metrics, artifact hashes, and operational observations. Require execution and evaluation stages. |
| `fail(run_id, reason)` | Close the active stage and preserve the failure reason. |
| `record_probe(baseline_id, repeat_id, result, ...)` | Complete a replication, compare it with the baseline, and persist the comparison in one transaction. |
| `probes(run_id)` | Read stored probe comparisons for a baseline. |

`openloop.probe.run_probe` orchestrates a probe through this public API.

A result holds measurements only.
A decision compares several samples and belongs in a separate record.

Every submission has an ID.
Reused submissions share an `attempt_id` and contain `reused_from`.
Their hypothesis and parent references remain separate records.
They cannot change the shared attempt.
They do not create another statistical sample.
Check `reused_from` before scheduling work.

A `request_key` makes repeated admission of one request idempotent.
Reusing that key with different inputs or provenance raises `ConflictError`.
Different keys can reference the same cached attempt.
`purpose="replication"` always creates a new attempt.
`purpose="retry"` requires `retry_of` to reference a failed run with identical inputs.
The retry records the failed run as a parent.
A new seed or fidelity changes the input hash.

## Stage times and recovery

Attempt stages are queue, execution, and evaluation.
Submission starts the queue stage.
`start_stage` records the current stage's finish and the next stage's start.
Completion and failure close the active stage.
Stage events carry the `event.stage` column and keep durations in their payload.
`run.stages` groups attempt events by stage.

Preparation belongs to the submission, not the attempt.
The stages are proposal and implementation.
A cache-hit submission records its own preparation.
Preparation can be recorded in any attempt status.
It must finish by the submission time, and the proposal must finish before the implementation starts.
The ledger does not invent preparation times.
`run.preparations` lists them by start time.

Events store UTC timestamps with microsecond precision.
Durations use a monotonic clock within one connection's lifetime.
After reopening, durations use wall time and carry `duration_clock="wall"`.
Wall-time durations can be affected by clock changes.

An interrupted write transaction leaves no partial submission or terminal result.
A committed running attempt remains visible after restart.
The coordinator must reconcile the worker before recording failure or requesting a retry.
The ledger does not assume that an interrupted coordinator stopped the worker.

## Determinism probe

`run_probe` validates the tolerances and the baseline before it admits any work.
It submits a replication with the baseline as parent and starts execution.
It passes the baseline's exact `ExperimentInputs` to the runner, which must perform the experiment and return a trusted `Result`.
It then starts evaluation and calls `record_probe`.
The ledger records a new replication even when a cached result exists.
Any error after admission, including `KeyboardInterrupt`, fails the replication and propagates.

The comparison checks metric names, values, units, directions, splits, sample counts, and artifact hashes.
For a metric, the permitted difference is `atol + rtol * abs(baseline_value)`.
Both tolerances default to zero.
Timing and costs are excluded from the comparison.
This check measures one repeat; it does not establish statistical confidence.

`record_probe` requires a succeeded baseline.
The repeat must be a replication with the baseline as parent and the same input hash.
Replication completion and its comparison enter the ledger in one transaction.

A mismatch for `deterministic` inputs sets `invalidates_cache`.
That flag disables completed-result reuse for the input hash.
For `noisy` inputs a mismatch records variation and leaves reuse enabled.
A cached result is one earlier sample and never adds a sample.
Pending work can still be shared.
The flag persists across restarts and later matching probes.
There is no automatic reset of that flag.

## Example

This example uses a synthetic evaluator and spends no API or GPU credit.
The example hashes identify toy snapshots, not the current training setup.

```python
from openloop.ledger import (
    ExperimentInputs,
    Ledger,
    Metric,
    Reproducibility,
    Result,
    content_hash,
)
from openloop.probe import run_probe

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
    reproducibility=Reproducibility.DETERMINISTIC,
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
    report = run_probe(ledger, run.id, evaluate)
    assert report.matches
    assert ledger.get(run.id).status == "succeeded"
    assert run.id in {ancestor.id for ancestor in ledger.lineage(report.repeat_id)}
```

## Test coverage

Confirmed on October 10, 2026.
The first implementation checks are covered by these tests:

| Requirement | Evidence |
| --- | --- |
| Hash stability | [Golden identity digests](../tests/ledger/test_identity.py) and [property tests](../tests/ledger/test_identity_properties.py) check canonical ordering, immutable snapshots, reconstruction, and both hashes. |
| Cache hits | [Ledger tests](../tests/ledger/test_store.py) and [loop tests](../tests/loops/test_env.py) check shared attempt references, preserved provenance, no extra samples or charged work, request replay, and explicit replication. |
| Contract violations | [Ledger tests](../tests/ledger/test_store.py) and [loop tests](../tests/loops/test_env.py) reject invalid transitions, exceeded budgets, concurrent steps, frozen-input drift, and invalid output. Rejected work cannot publish a successful result. |
| SIGKILL during a transaction | [Store crash tests](../tests/ledger/test_store.py) kill the writer before submission or completion commits. They check rollback and safe subsequent writes. |
| SIGKILL during a live run | [Coordinator crash tests](../tests/loops/test_crash_recovery.py) kill a real process during execution and evaluation. They check committed evidence, SQLite integrity, foreign keys, pending references, reconciliation, retry lineage, and successful-result reuse. |

The live-run tests wait for a committed stage before sending SIGKILL.
They use temporary ledgers and CPU-only T0 jobs.
They do not use production data, GPU training, or paid services.
A restarted coordinator sees the interrupted attempt as `running`, with no result.
It explicitly records failure before it retries the same inputs.
The retry creates a new attempt and retains the failed attempt.
Database consistency is checked before and after recovery.
Automatic reconciliation, termination of orphan worker processes, and recovery from a host power loss are not established by these tests.

## Boundaries

Only trusted orchestration and evaluator code may publish results.
This API is not an agent security boundary or a statistical decider.
The caller must verify artifact bytes before recording their hashes.
Artifact bytes and metric-series files remain outside this first ledger implementation.
Decision records, campaign budgets, worker execution, held-out access control, and migration of setup records remain separate tasks.

### Query design

An earlier `query` defaulted to `limit=100`.
A caller that wanted all evidence got the first 100 runs without any signal.
Truncation is therefore an explicit choice now.
`query_all` is the only unbounded read, and `query` cannot be called without a `limit`.
Both share one ordering (submission order) and one SQL statement, so pages concatenate to the `query_all` result absent concurrent writes.
`query_all` reads ids and runs inside one read transaction, so it is a consistent snapshot.
