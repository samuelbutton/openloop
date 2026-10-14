# Research operators

Implemented on October 14, 2026. Needs: [N6 and N8](../NEEDS.md#n1n10).
The [plugin source](../src/openloop/operators/) uses public loop, executor, ledger, and decider APIs.
The [offline tests](../tests/operators/) use scripted model replies and small CPU smoke tests.

## Hypothesis cards

`Proposer.propose(operator, baseline=..., evidence=..., parents=...)` writes a hypothesis card to the ledger.
All references name public ledger runs. Parents are supplied as run IDs and converted to candidate hashes.
The proposer receives only selected records. It does not read artifacts or arbitrary source files.
Held-out records, held-out metrics, and cache aliases are rejected before the provider call.
The baseline must be successful. Its frozen loop, environment, metric, unit, and fidelity define the prediction.
Other evidence must use the same loop and environment.
Only the declared validation metric enters the model context.
Failure reasons are diagnostic text, not instructions or evidence of idea merit.
Only the last 2,000 characters of a failure reason enter model context.
They appear in a field of kind `untrusted_data` that reports `truncated` and `original_chars`.
The ledger keeps the full text.

| Operator | Required parents | Purpose |
| --- | --- | --- |
| `draft` | None | Propose a new hypothesis from the selected evidence. |
| `improve` | One candidate | Propose a change to that candidate. |
| `debug` | One candidate and failed experiment evidence | Propose a repair for an observed experiment failure. |
| `combine` | At least two distinct candidates | Propose a combination that needs its own evaluation. |

The provider returns `ModelReply[ProposalDraft]`.
The draft contains a title, rationale, proposed change, finite predicted delta, and `probability_of_improvement`.
The probability is required and lies in 0 to 1. It means the chance that the measured paired median gain at the recorded fidelity is positive.
An optional `predicted_interval` `(low, high)` bounds the signed delta.
The delta means candidate metric minus baseline metric at the recorded fidelity.
A negative delta predicts improvement for a minimized metric. A positive delta predicts improvement for a maximized metric.
The coordinator supplies the evidence, parent identities, metric metadata, and fidelity.
The provider cannot substitute those fields.
The card hash covers all card fields and hypothesis version 2. Version 2 added the probability and interval.
Cards and provider receipts are recorded as activities, not measurements or verified wins.

The model ports are callables with typed requests and replies.
`ProposalRequest.prompt` and `ImplementationRequest.prompt` supply instructions and canonical JSON context.
`ModelReply` records the provider, returned model identifier, response ID, and a simulation flag.
It also carries the usage that the provider reported: `input_tokens`, `output_tokens`, `reasoning_tokens`, and `cached_input_tokens`.
Each count is a nonnegative integer or `None` when unreported. No value is invented.
`output_tokens` is the billed output total and includes reasoning tokens. Cached tokens are a subset of input tokens.
Provider adapters must report these values truthfully. Model aliases in `config/models.toml` remain aliases.
Each operator hashes the exact prompt text (`prompt_hash`, SHA-256) and records it with the reply, usage, and spend note in the ledger payload.
These plugins do not call OpenAI, Codex CLI, or another paid provider by default.
A deployment supplies an approved provider adapter and its request timeout and credential handling.
Spend control is described next.
The editor provider supplies replacement text. It must not run an unrestricted host coding agent.
No API key enters the sandbox.

## Spend gate

`Proposer` and `Implementer` need a `SpendGate` and a `CallBudget`, or the explicit argument `simulated_only=True`.
Omitting both raises `InvalidInputError`.
A simulated-only operator takes no gate. It refuses any reply that is not labeled `simulated`, records the refusal as a failure, and raises `SpendLimitError`.
That check runs after the call, so use this mode only with scripted providers.

`Pricing(provider, model, input_usd_per_million, output_usd_per_million)` is supplied by the caller and hashed into each reservation. No price is hard-coded.
`CallBudget(pricing, max_input_tokens, max_output_tokens)` declares the per-call maximum.
The prompt's UTF-8 byte length bounds its token count and must not exceed `max_input_tokens`.
The adapter must enforce `max_output_tokens` at the provider.

`SpendGate(ledger, limit_usd=...)` stores `SPEND_RESERVATION` and `SPEND_SETTLEMENT` activities.
Amounts are integer micro-USD, rounded up.

1. Before a paid call, the gate reserves `max_input_tokens` x input price plus `max_output_tokens` x output price.
   It refuses with `SpendLimitError` if settled spend, plus open reservations, plus the new reservation exceeds the limit. The provider is not called.
2. After a reply with reported input and output tokens, the gate settles at usage x pricing, with cached tokens billed at the input price.
   A reply labeled simulated settles at zero.
   A settlement may exceed its reservation if the provider overran. The record says so.
3. An error, cancellation, a missing usage count, or a reply from another provider keeps the reservation open at its full maximum.
   The failure or success record in the proposal or implementation stores the reason under `spend`.
4. Every attempt, including a retry, needs a new reservation.
   `SpendGate.settle_reported` closes an open reservation from a cost the coordinator confirmed in provider billing.
   `SpendGate.committed()` reports settled and open amounts.

The gate guards one coordinator process. Its read and write are separate ledger transactions, so concurrent processes can together exceed the limit.
The account cap in `config/billing.toml` remains the independent backstop.
An implementer attempt refused by the gate records an implementation failure and raises `SpendLimitError` instead of burning the remaining attempts.

## Implementation

`Implementer` requires a `ContainerExecutor` with `network=False`.
The coordinator supplies a pinned Git commit, a pinned cached image, exact editable base files, a smoke command, and resource limits.
The image must contain Python on `PATH` and the smoke-test dependencies.
It is not downloaded by the implementation operator.
The source commit must contain only candidate-safe material. Do not include credentials, held-out data, held-out reports, or trusted evaluation code.
The editable file list is an explicit coordinator policy. It is not inferred from configuration-key names in `LoopSpec`.

The provider returns `ModelReply[Patch]`, with full UTF-8 replacements for permitted files.
Unsafe paths, credential paths, unapproved files, unchanged patches, and oversized candidates are rejected.
Every attempt starts from the same commit and base files.
The worker checks the base file text before it applies replacements.
It copies the read-only Git snapshot to `/tmp/worktree`, applies the edits, and runs the smoke command there.
This is a materialized working tree inside Docker. It does not register a host Git worktree, create a commit, or change the user's checkout.
Docker supplies isolation; a working directory alone does not.

The smoke command inherits only a fixed `PATH`, scratch `HOME`, and disabled Python bytecode writes.
It must not start GPU training or paid services without separate approval.
After a smoke pass, the worker checks that protected source files and candidate text did not change during the smoke test.
It publishes only the edited files to `/outputs`.
The coordinator verifies exact output membership, bytes, and artifact hashes before it records success.
A smoke pass checks implementation execution. It does not establish statistical merit or resistance to a malicious smoke-test bypass.
Trusted experiment evaluation still runs through the loop and decider contracts.

There is one initial attempt and at most two retries per card hash.
The ledger assigns attempt numbers and admits one owner atomically.
The cap persists across calls and restarts.
The source commit, base-file hash, image, smoke command, worker hash, and resource limits remain frozen across retries.
Each retry receives the preceding failure reason and the original base files.
It cannot change the hypothesis to escape the retry cap.
Provider errors, invalid patches, smoke failures, and cancellation produce implementation-failure records.
After three failures, `implement` returns `None` without a provider call. A later call cannot allocate a fourth attempt.
The retry prompt quotes the preceding failure as an `untrusted_data` field limited to its last 2,000 characters. The ledger keeps the full reason.
Cancellation propagates after cleanup and logging.
An unresolved start blocks another owner. It does not become a success or an idea failure after a crash.
The owner confirms that the executor job is gone, then calls `Implementer.reconcile(card, reason, executor_job=...)`.
It appends a failure linked to the unresolved start and raises `InvalidTransitionError` if no attempt is unresolved.
The reason is operational. It is not evidence about the idea.

## Failure records

The core's `activity` table keeps proposal and implementation records separate from experiment attempts and scores.
`Ledger.record_activity` and `Ledger.activities` are its public API.
The payload records the card, request hash, model receipt, source policy, attempt, executor job, artifacts, and observed failure reason as applicable.
UTC occurrence timestamps are recorded by the ledger. Missing durations and monetary costs stay absent.
Started, progress, success, and failure activities preserve implementation history.
An implementation success stops further repairs for that card.

The coordinator calls `bind_candidate(ledger, implementation, inputs)` before confirmation starts.
This freezes the association between the produced file snapshot and the loop's experiment candidate identity.
It is a trusted coordinator assertion because adapters derive source and configuration identities differently.
The implementation snapshot hash covers the base commit and replacement text. File modes follow the base commit.
It is distinct from the loop's experiment candidate hash.

`record_idea_failure(ledger, implementation, decision)` accepts a trusted, measured confirmation discard for that bound candidate.
It checks successful, distinct, paired baseline and candidate runs from the same comparison context.
Smoke failures, failed experiments, cache aliases, inconclusive verdicts, unmeasured allocation drops, and held-out decisions cannot become idea failures.
`decide` lists the `2 * sample_count` paired runs first, as baseline then candidate for each seed, and the noise-calibration runs after them.
The first group is checked as pairs. The remainder must be distinct, successful confirmation-phase runs of the protocol baseline in the pairs' comparison context, and the decision must list some.
The decider remains responsible for the frozen protocol and statistical validity of the decision.
Implementation-failure and idea-failure activities have different kinds and references.
Neither category is inferred from the other.

## Outcomes and calibration

`record_outcome(ledger, card, decision)` records exactly one `HYPOTHESIS_OUTCOME` for a card, as a child of its hypothesis record.
The candidate must already be bound with `bind_candidate`, and the decision must name that candidate.

| Decision | Outcome |
| --- | --- |
| Held-out `VERIFIED` | `verified` |
| Confirmation or held-out `DISCARD` with samples | `discarded` |
| Confirmation or held-out `INCONCLUSIVE` | `inconclusive` |
| Screen `DISCARD`, an allocator drop, or a confirmation excluded by a top-k limit | `screen_dropped` |

Screen and confirmation promotions are not terminal and are rejected.
`inconclusive` is for a schedule that the coordinator stops extending.
The record stores the decision and its content hash (`decision_hash`), the protocol hash, stage, prediction, and observed effect.
The observed effect has the decision's median gain, its signed delta, the interval, and the sample count.
A signed delta is candidate minus baseline: the negated gain for a minimized metric.
A repeat call with the same decision returns the same record. A different decision is a `ConflictError`.

`record_idea_failure` stays a separate record. It also records the discarded outcome from the same decision.
If an outcome already exists, it must be the discarded outcome of that decision. Once an idea failure exists, only that discarded outcome can be recorded.
A repeat call completes the outcome after an interruption between the two writes.

`calibration_summary(ledger)` is a pure read. It reports outcome counts and scores only `verified` and `discarded` outcomes that have a measured gain.
Inconclusive outcomes have an incomplete schedule. Screen drops rest on two noisy pairs or on no measurement. Both are excluded from scores but counted.

- `brier_score`: mean squared difference between `probability_of_improvement` and the indicator that the observed median gain is positive.
- `mean_signed_error`: mean of predicted delta minus observed delta.
- `interval_covered` and `interval_total`: how often the observed delta fell inside a stated predicted interval.

Verified outcomes use the held-out decision. Outcomes are reports for the operator and never enter proposer context.
A score of `None` means nothing was scorable, not zero.

## Offline card example

This example uses simulated T0 evidence and a scripted provider. It starts no model call or training run.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from openloop.ledger import Ledger
from openloop.loops import T0, ExperimentEnv, LoopAction
from openloop.operators import ModelReply, ProposalDraft, Proposer


async def provider(request):
    return ModelReply(
        ProposalDraft(
            "Move closer", "Planted T0 fixture", "Change coordinates", -0.1, 0.6
        ),
        "scripted", "fixture", "example-1", simulated=True,
    )


async def example(ledger):
    loop = T0()
    env = ExperimentEnv(
        loop, ledger, runner=loop.run, config={"coordinates": [0, 0]},
        environment_hash="a" * 64, budget=1,
    )
    baseline = await env.step(LoopAction(0))
    card = await Proposer(loop.spec, ledger, provider, simulated_only=True).propose(
        "improve", baseline=baseline.run_id, parents=(baseline.run_id,),
    )
    assert card.prediction.delta == -0.1
    assert len(ledger.activities(subject_hash=card.hash)) == 1


with TemporaryDirectory() as directory, Ledger(Path(directory) / "runs.sqlite3") as ledger:
    asyncio.run(example(ledger))
```

## Verification and limits

[Ledger tests](../tests/ledger/test_activities.py) check bounded repairs, concurrent admission, idempotency, migration, and SIGKILL inside and after a transaction.
A version-2 ledger is not migrated when it is opened. Opening it with the default arguments leaves its experiment APIs working and makes the activity APIs raise `SchemaError`.
`Ledger(path, migrate=True)` migrates a writable working copy to version 3, which adds the activity table and append-only triggers.
The migration preserves existing run records, identity hashes, and cache links.
Open frozen evidence with `Ledger(path, read_only=True)`; it changes no schema, row, or journal mode.
Migration was tested only on temporary databases. No existing local ledger was opened for this change.
Version 1 remains unsupported.

The CPU implementation stand-in checks the wrapper and operator contracts. It does not establish Docker isolation.
The existing [executor checks](EXECUTORS.md#validation) cover the container policy and opt-in real-container tests.
Live provider adapters, actual Codex CLI editing, and native training were not run or verified for this change.
No test calls a paid model. The spend gate was exercised only with scripted providers and caller-chosen test prices.
The gate does not verify that an adapter honored `max_output_tokens`, and it does not reconcile settled costs against provider invoices.
The outcome and calibration records use simulated T0 evidence; they show that the bookkeeping is correct, not that model probabilities are calibrated.

## Real-container validation

[`tests/operators/test_sandbox_real.py`](../tests/operators/test_sandbox_real.py) runs the real `Implementer`, a network-disabled `ContainerExecutor`, and the real `smoke_worker.py` inside Docker.
It uses scripted providers with `simulated_only=True`; no model is called.
The tests are opt-in. `OPENLOOP_TEST_PYTHON_IMAGE` must name a pinned, cached image with `python` on `PATH`.
The reference image is the official `python:3.12-slim` for linux/arm64, pinned as `python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`.
Use a `--basetemp` inside the repository, because Docker Desktop may not share `/private/tmp`:

```sh
OPENLOOP_TEST_PYTHON_IMAGE=python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 \
  uv run pytest tests/operators/test_sandbox_real.py --basetemp=$PWD/.pytest_cache/sandbox
```

They check the following:

- A passing patch publishes exactly the edited files with matching hashes.
- A smoke failure records an implementation failure, and the retry receives its reason.
- Three failures exhaust the cap. A later call returns `None` without a provider call.
- These smoke commands are rejected: one that writes extra files or symlinks to `/outputs`, one that edits or deletes a protected file, and one that rewrites the candidate file.
- Network access is denied.
- Non-ASCII text with CRLF line endings round-trips byte-exactly.
- No container remains afterwards.

The default suite uses only the `SandboxStandIn`.

### T1 smoke check

Docker on macOS has no Metal, so the sandbox cannot run MLX training.
[`smoke_checks.t1_smoke_command()`](../src/openloop/operators/smoke_checks.py) supplies a CPU-only smoke command for T1 code changes.
It compiles the edited `train.py` in memory, without writing bytecode or executing it.
It then checks the structure that the T1 token-budget adapter requires: every declared training setting as a module-level constant, the seed call, the progress and startup assignments, the time-budget stop rule, and the `total_tokens` boundary.
A pass means only that the file parses and keeps the adapter's anchors.
It says nothing about numerics, runtime errors, or the merit of the idea.
The trusted T1 `build_job` re-validates the source outside the sandbox with the authoritative `training_program`.
