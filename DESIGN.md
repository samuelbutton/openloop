# openloop design

Status: proposed design. Date: October 8, 2026.

This document defines the first core and its plugin boundaries.
The [needs table](NEEDS.md#n1n10) gives the evidence for each need.
The design uses short sentences and consistent technical terms from [STE-100 Issue 9][ste100].

## Current state

The repo has a Python 3.12 package, an environment probe, and a FineWeb-Edu preparation tool.
It also has recorded MLX runs and tabular load reports.
The local `ledger.json` stores setup records.
It is not the append-only ledger specified below.

The first ledger implementation is available in `openloop.ledger`.
It records input manifests, submissions, attempts, lineage, preparation times, stage events, results, and determinism probes.
Experiment inputs declare whether results are deterministic or noisy.
`openloop.probe` runs determinism probes through the ledger API.
The [ledger guide](docs/LEDGER.md) describes its API and schema.
Campaign records, artifact storage, and spend records remain planned extensions.
The [loop interface and T0/T1 adapters](docs/LOOPS.md) are implemented.
T1 job generation and CPU stand-ins are checked; native training remains unverified.
The [local and container executors](docs/EXECUTORS.md) are implemented.
The [decider](docs/DECIDER.md) has frozen protocols, staged evidence, two screen policies, and a trusted final-evaluation guard.
Both policies are checked on simulated T0 families.
The mock executor, GCP stub, application spend gate, and plugins remain planned.
The [model configuration](config/models.toml) selects `gpt-6-astra` as proposer and `gpt-6.1-sol` through Codex CLI as implementer.
The [billing configuration](config/billing.toml) records a $10 monthly OpenAI limit and $10 of RunPod credit.
RunPod auto-pay is off.
These account controls do not implement the application spend gate.

## Terms

| Term | Meaning |
| --- | --- |
| Campaign | A research objective with a fixed protocol, budget, and selection rule. |
| Loop | A contract for experiments that change one declared surface. |
| Hypothesis | A proposed change with a predicted effect and evidence references. |
| Candidate | An immutable code and configuration snapshot that implements a hypothesis. |
| Experiment | A complete specification for one candidate, seed, phase, and fidelity. |
| Phase | The verification phase of an experiment, such as screen or held-out. |
| Attempt | One execution of an experiment. A retry creates a new attempt. |
| Replication | A deliberate additional measurement of an experiment. |
| Fidelity | A declared amount of experiment work, such as a training token count. |
| Evaluator | Trusted code that measures outputs against a frozen evaluation specification. |
| Decider | A rule that uses measurements to return a decision. |
| Ledger | The permanent records for experiment inputs, events, measurements, and decisions. |
| Artifact | A file with a recorded content hash, type, and size. |
| Held-out split | Data reserved for final evaluation after candidate selection. |

## Core and plugins

The core owns records, contract checks, execution limits, and decision records.
Plugins use typed records through the core API.
Plugins do not write storage tables directly.

| Core part | Responsibility | Needs |
| --- | --- | --- |
| Ledger | Store immutable inputs, lineage, events, measurements, artifacts, and decisions. | N1, N5 |
| Loop interface | Declare the mutable surface, frozen evaluator, data, fidelity, and budget unit. | N4, N9 |
| Executors | Run approved jobs within resource and access limits. Record lifecycle events and actual costs. | N3, N8, N10 |
| Decider | Apply a recorded statistical policy to validated measurements. | N2 |

Hash calculations, contract checks, and decision calculations are pure functions.
Adapters handle files, SQLite, processes, and network calls.
The first implementation uses one coordinator and SQLite.
It does not require a message broker or a distributed database.

### Plugins

| Plugin | Contract with the core | Needs |
| --- | --- | --- |
| Loop definitions | Supply a versioned loop specification, job construction, and trusted evaluation code. | N4, N9 |
| Allocation policies | Rank eligible jobs within the remaining budget. Start with FIFO; add successive halving and expected-value policies later. | N3 |
| Proposers | Produce hypothesis cards with evidence, predicted effects, and candidate requests. | N6 |
| Implementers | Produce candidate snapshots inside the declared mutable surface. Record repair attempts separately. | N6, N8 |
| Decision policies | Supply versioned thresholds and calculations through the decider contract. | N2 |
| Research memory | Derive findings, failed hypotheses, and evidence counts from ledger records. | N7 |
| Agent toolbelt | Expose ledger queries, run comparisons, hypothesis status, and replication requests through Python and MCP. | N6 |
| Researcher UI | Show lineage, measurements, costs, stage times, and pending approvals. Link each conclusion to evidence. | N5, N6 |

The proposer receives research evidence through ledger queries.
Failed runs remain available as diagnostic evidence.
Only decisions that pass the verification policy can support a verified finding.
An implementation error does not establish that a hypothesis is false.

## Ledger schema

SQLite stores small records and references.
Parquet artifacts store long metric series.
Git objects store source snapshots and diffs.
SHA-256 identifies artifact bytes and canonical manifests.
The target schema uses these logical tables:

| Table | Key and fields |
| --- | --- |
| `campaign` | `campaign_id`; protocol hash; objective; loop specification hashes; decision policy hash; budget limits; creation time. |
| `hypothesis` | `hypothesis_id`; campaign ID; text; predicted metric effect; evidence references; proposer request reference. |
| `candidate` | `candidate_hash`; hypothesis ID; source tree hash; diff artifact hash; configuration hash; dependency lock hash. |
| `candidate_parent` | Child candidate hash; parent candidate hash; relation such as improve, combine, or debug. |
| `experiment` | `experiment_hash`; candidate hash; loop specification hash; evaluator hash; data snapshot hash; tokenizer hash; environment hash; seed; phase; fidelity; reproducibility. |
| `attempt` | `attempt_id`; experiment hash; request key; purpose; retry parent; executor identity; declared resource limits. |
| `event` | Event ID; attempt or campaign ID; sequence number; event type; UTC time; duration; payload hash. |
| `artifact` | SHA-256; byte count; media type; storage location. An association record links each artifact to its owner and role. |
| `measurement` | Measurement ID; attempt ID; evaluator hash; split; metric name; value; unit; direction; sample count; series artifact hash. |
| `decision` | Decision ID; campaign ID; candidate and comparator hashes; policy hash; measurement references; effect; interval; verdict; reason; creation time. |
| `spend` | Entry ID; reservation ID; campaign and attempt references; provider; currency; event type; amount; price snapshot hash; usage artifact hash. |

Foreign keys connect all records.
Parent references form a directed acyclic graph.
The core rejects cycles and missing parents.
Each record has a schema version.
Identifiers for hypotheses, campaigns, attempts, and events distinguish occurrences from content.

Records are append-only after publication.
Corrections create new records with a reference to the old record.
Status views derive current state from events.
The core can rebuild these views without changing the original evidence.

The environment artifact contains the probe output and runtime versions.
The experiment manifest also records the budget unit, requested amount, resource class, and execution settings.
It records effective model identifiers, prompt hashes, token usage, and provider responses for agent calls.
The ledger never stores API secrets.
Provider aliases remain labeled as aliases.

### Identity and cache reuse

Canonical manifests use sorted JSON keys, UTF-8, fixed separators, and explicit schema versions.
Manifests reject non-finite numbers.
They declare defaults and units before hashing.
Paths, mutable tags, and timestamps do not substitute for content hashes.

The experiment hash covers all declared inputs that can affect the result.
These inputs include evaluator code, data membership, tokenizer, environment, seed, fidelity, and resource settings.
A cache hit references a completed, validated attempt with the same experiment hash.
It does not create another statistical sample.
Crashes and incomplete attempts are not cache hits.

Replication bypasses cache reuse and creates a new attempt with purpose `replication`.
A retry uses purpose `retry` and references the interrupted or failed attempt.
A new seed or fidelity creates a new experiment hash.
These distinctions permit deliberate drift checks without accidental duplicate samples.
Matching inputs do not guarantee matching outputs on nondeterministic hardware.

### Publication and recovery

The coordinator writes artifacts to temporary files and verifies their hashes.
It then publishes the files under their content hashes.
A SQLite transaction records the artifact references and completion event.
File publication and the database transaction are separate operations.
Recovery finds orphan files and incomplete attempts after a crash.

Queue admission, attempt creation, and budget reservation use one SQLite transaction.
A unique request key prevents duplicate admission of the same request.
Remote executors use idempotent submission keys when the provider supports them.
An unknown remote outcome stays unresolved until reconciliation.
The core does not assume exactly-once execution.

## Loop interface

Each loop publishes an immutable `LoopSpec` before a campaign starts.
The core checks its invariants before execution and after output collection.

| Specification field | Required content |
| --- | --- |
| Identity | Loop name, version, specification hash, and adapter source hash. |
| Mutable surface | Allowed files and configuration keys. All other inputs remain frozen. |
| Evaluator | Source hash, metric names, units, optimization directions, and output schema. |
| Data | Train, validation, and held-out manifest hashes; tokenizer hash when applicable. |
| Budget | Unit, per-job limit, campaign limit, and wall-time limit. Training work and elapsed time remain separate. |
| Fidelities | Named work amounts and permitted promotion paths. |
| Invariants | Protected file hashes, resource limits, artifact rules, and allowed network access. |
| Verification | Seed schedule, comparator, minimum effect, decision policy, and held-out release rule. |

The Python signature follows the single-use shape of Tinker Cookbook's [`Env`][tinker-env].
That interface has `initial_observation` and `step`; it has no `reset` method.
Openloop uses structured experiment actions instead of generated token IDs.
The contract is implemented in `openloop.loops`.
It is not a Tinker SDK adapter.

```python
from __future__ import annotations

from typing import Protocol


class LoopEnv(Protocol):
    @property
    def spec(self) -> LoopSpec: ...

    async def initial_observation(self) -> Observation: ...

    async def step(self, action: LoopAction) -> StepResult: ...
```

| Type | Content |
| --- | --- |
| `Observation` | Ledger evidence references, candidate identity, remaining budget, and permitted next actions. |
| `LoopAction` | Request key, seed, phase, fidelity, execution purpose, and retry reference. The candidate is fixed by the environment. |
| `StepResult` | Next observation, run and attempt references, run status, measurements, observed work, execution flag, and `episode_done`. |

One environment instance serves one candidate episode.
Steps can cover a screen and planned confirmation runs.
The current episode ends when its remaining allowance cannot fund another fidelity.
The later decider can end it after a terminal decision.
The coordinator owns shared storage and executors across episodes.
Cancellation records an event and preserves all collected evidence.

The core validates each action before it admits a job.
The loop adapter constructs a job from immutable inputs.
The executor produces artifacts.
The trusted evaluator reads those artifacts and produces typed measurements.
The decider returns a recorded decision.
Candidate code cannot supply its own authoritative score.

T0 uses planted outcomes to test decisions and failures.
T1 permits changes to declared training constants.
Its adapter enforces exact token budgets at two fidelities, 8m and 21m tokens, against the pinned upstream source.
The earlier upstream measurement used a time budget.
No native token-budget run has been measured through the new adapter.
Later loops use the same ledger and action types.
Combining two accepted changes creates a candidate with two parent references.
The combination requires its own verification.

## Executors

The executor contract provides `submit`, `poll`, `cancel`, and `collect` operations.
Each submission includes immutable inputs, an idempotency key, and resource limits.
Each result includes exit status, artifact references, usage, and lifecycle times.

| Backend | Purpose and boundary |
| --- | --- |
| Local process | Run trusted preparation tools and reviewed MLX jobs. Limit time and memory. Use one GPU training lane. |
| Restricted worker | Run candidate code in an isolated container or VM. Deny network access by default. Mount inputs read-only. |
| Mock | Replay recorded durations and outcomes at 1 to 10,000 simulated workers. Report simulated costs and throughput explicitly. |
| GCP stub | Check contract compatibility. Reject execution until a real backend is configured. |

A Git worktree isolates source changes; it is not a security boundary.
Unreviewed code cannot run through the trusted local process backend.
Restricted workers receive no API keys, host home directory, or container control socket.
The trusted evaluator runs outside the candidate's writable area.
Network exceptions require an explicit job policy.
The worker stops jobs that exceed their declared limits.

The first Mac implementation runs one MLX training job at a time.
T1 now uses the local executor and its user-wide GPU lock.
The container executor uses a pinned local Docker image.
Its source and explicit inputs are read-only. Temporary writes stay in the container, and
the candidate can write limited, hashed files to an outputs directory.
Executor ownership and idempotency remain in memory.
The [executor guide](docs/EXECUTORS.md) gives the limits and failure semantics.
RunPod remains unused until a GPU workload is approved.
Later training adapters can expose `forward_backward`, `optim_step`, `sample`, and `save_state` operations.
Those operations do not replace the experiment contract or create a new training framework.

## Decider

The first implementation is available in `openloop.decider`.
The [decider guide](docs/DECIDER.md) gives the rules, statistical assumptions, and [simulated T0 power and error evidence](docs/decider-t0.json).
The coordinator is in memory.
Durable campaign records and restricted held-out storage remain planned.

The campaign freezes the decision policy before comparison starts.
The policy declares sample counts, seed schedules, minimum effects, uncertainty calculations, and multiple-comparison rules.
The decider is a pure function of this policy and validated measurements.
It returns `promote`, `discard`, `inconclusive`, or `verified` with evidence references and a reason.
Execution failures have separate status records.

Screening uses validation data at low fidelity.
Confirmation uses the declared fidelity and seed schedule.
Comparisons use the same data and work budget.
The policy estimates run variation before it claims an improvement.
It reports effect size and an uncertainty interval.
Fixed-seed repeats measure different variation from distinct-seed runs.

The protocol defines the comparison family and false-discovery control before selection.
A selectable 2-sigma policy does not automatically establish control across repeated searches.
Too little evidence produces `inconclusive`.
The initial baseline is a reference, not a verified improvement.
The [upstream rigor result](research/autoresearch-mlx.md#exact-keepdiscard-semantics) demonstrates this distinction.

Final evaluation uses the held-out split only after candidate selection.
The proposer and candidate worker cannot access held-out data or intermediate held-out scores.
The evaluator records final results in restricted ledger records.
The campaign can release these records after the final selection closes.
Further adaptive search requires a new untouched evaluation set or a separately approved reuse protocol.

FineWeb-Edu uses the frozen three-shard [90/5/5 document split](data/fineweb-edu/README.md).
Its hash rule removes exact duplicate documents; it does not establish removal of near duplicates.
HPO-B keeps its original task meta-splits.
NAS-Bench-201 keeps its documented validation and test roles.
These splits serve different purposes and are not interchangeable.

An evaluation-design loop can propose a new evaluator version.
It cannot change the evaluator for an active comparison.
A new version starts a new protocol and requires baseline measurements.

## Run sequence

1. Register the campaign protocol, budget, and baseline references.
2. Record a hypothesis and its evidence references.
3. Record the candidate snapshot and its parent references.
4. Check the loop contract and frozen input hashes.
5. Find a valid cache result or reserve budget for a new attempt.
6. Submit the job and record its lifecycle events.
7. Collect artifacts and check their hashes.
8. Run the trusted evaluator and record measurements.
9. Apply the decision policy and record its result.
10. Update derived memory and return the next permitted actions.

Events record proposal, implementation, queue, execution, evaluation, and decision stages.
UTC timestamps support inspection across workers.
Monotonic durations measure elapsed time within a process.
Reports distinguish active work, queue delay, and total hypothesis-to-decision time.
Failed jobs retain logs, partial artifacts, exit status, and consumed budget.

### Spend gate

The application gate checks settled spend plus open reservations before each paid action.
It reserves a conservative maximum cost for each call or GPU job.
Retries require new reservations.
Unknown billing outcomes retain their reservations until reconciliation.
Prices, usage, currency, and actual cost enter the ledger.

The gate rejects requests that exceed the approved campaign or provider allowance.
It uses the current $10 account settings as upper limits unless the user changes them.
Provider enforcement can lag; the gate must not depend on immediate provider rejection.
Account limits and application limits remain separate recorded controls.

## First implementation checks

The first implementation must check stable hashes, cache references, deliberate replication, and contract violations.
It must also check recovery after forced process termination.
T0 must measure false acceptances and missed planted improvements.
T1 must complete the full recorded sequence before any verified gain claim.
Mock scale checks must preserve budgets and prevent duplicate admission.
Restricted execution checks must establish the N8 boundary before autonomous code runs.

Stable hashes, cache references, replication, contract violations, and ledger consistency after SIGKILL are checked.
The [ledger guide](docs/LEDGER.md#test-coverage) links the tests and defines their limits.
T0 statistical decisions are checked for power and error rates on [simulated families](docs/DECIDER.md#t0-validation).
The complete T1 verification sequence, mock scale checks, and broader restricted-execution validation remain planned checks.
Setup reports alone do not complete those checks.

[ste100]: https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf
[tinker-env]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/types.py
