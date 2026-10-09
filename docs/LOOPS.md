# Loop interface

Implemented on October 9, 2026. Needs: N1, N2, N4, N8, and N9 in [NEEDS.md](../NEEDS.md).

`openloop.loops` defines a shared `LoopEnv` contract with `initial_observation` and `step`.
`ExperimentEnv` implements this contract through the public ledger API.
One instance serves one fixed candidate.
It has no reset method.

## Contract

| Record | Content |
| --- | --- |
| `LoopSpec` | Frozen source, dependencies, data, evaluator, tokenizer, execution settings, mutable keys, metrics, and an ordered tuple of named fidelities. |
| `LoopAction` | Seed, phase, fidelity, request key, execution purpose, and optional retry reference. |
| `Observation` | Candidate hash, evidence references, remaining work allowance, unit, and the names of fidelities the allowance can still fund. |
| `Job` | Immutable experiment inputs and an optional generated training program. |
| `StepResult` | Observation, submission and attempt references, run status, measured result, observed work, execution flag, and episode completion flag. |

The candidate configuration and parents enter the environment constructor.
They remain fixed during the episode.
A loop declares one or more fidelities, ordered by strictly increasing work, with unique names.
A fidelity name describes work, such as `"4x"` or `"21m"`, not a verification phase.
An action names a fidelity, or omits it to use the lowest.
An unknown name raises `ContractError`.
The action's `phase` is separate: `Phase.SCREEN` (the default) or `Phase.CONFIRM`.
Both enter the experiment hash independently, so any fidelity can be used in either phase.
Selection actions permit screening and confirmation only.
Held-out evaluation is a later task.

Before admission, the environment checks the candidate, job, and remaining allowance.
It deducts the full fidelity amount from the allowance before awaiting the runner.
It checks the frozen specification again before publishing measurements.
Failures and cancellation preserve a failed attempt and keep the deduction.
They do not produce a successful sample.
These are experiment-work limits, not money limits.

The allowance belongs to one `ExperimentEnv` instance.
It lives in memory only, does not survive a restart, and is not recorded in the ledger.
Durable campaign budgets are planned.
The check against the allowance runs before the ledger lookup and reads no ledger state to predict reuse.
It is therefore conservative: an action that exceeds the remaining allowance is refused even when its result would be a free cache hit.

The environment records execution and evaluation stages in the ledger.
`StepResult.status` distinguishes the cases.
`SUCCEEDED` marks a completed run, whether executed now or reused from the cache; reused runs consume no new work.
`QUEUED` or `RUNNING` marks a reference to pending work with no result yet, and no duplicate work executes.
Use `ledger.get(run_id)` to inspect that run later.
`FAILED` marks a request-key replay of a run that failed earlier; it has no result.
An explicit replication creates a new attempt, including for noisy T1 inputs.
A request-key replay returns its original step snapshot.
It does not execute or charge work again.

The environment rejects concurrent steps within one episode.
It does not schedule a shared GPU across several episodes.
The later executor must enforce that resource limit.
The decider will supply promotion rules, statistical verdicts, and final evaluation.

## T0: planted truth

T0 has a known quadratic objective and a configurable Gaussian noise standard deviation.
The default optimum is `(1, -1)` and the default standard deviation is `0.1`.
The only mutable setting is `coordinates`.
The oracle returns the exact squared distance to the optimum.
The runner adds seeded Gaussian noise.
Only the sampled loss enters the ledger result.

The fidelity `"1x"` uses one observation.
The fidelity `"4x"` averages four observations.
Its noise standard deviation is half that of `"1x"`.
The budget unit is `observations`.
The runner seeds its noise from the hash of the full experiment inputs, not from the seed alone.
Identical inputs reproduce identical samples, so the specification declares `deterministic` reproducibility.
A different seed, candidate, fidelity, or phase receives independent draws.
This keeps paired comparisons honest: candidates at the same seed do not share noise, and a confirmation does not reuse its screen's draw.
T0 results are labeled as simulated in the input manifest.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from openloop.ledger import Ledger, content_hash
from openloop.loops import T0, ExperimentEnv, LoopAction, Phase


async def example(ledger: Ledger) -> None:
    workload = T0()
    env = ExperimentEnv(
        workload,
        ledger,
        runner=workload.run,
        config={"coordinates": [0, 0]},
        environment_hash=content_hash({"example": "synthetic context"}),
        budget=5,
    )
    initial = await env.initial_observation()
    screen = await env.step(LoopAction(seed=42))
    confirm = await env.step(LoopAction(seed=43, fidelity="4x", phase=Phase.CONFIRM))
    assert initial.remaining_budget == 5
    assert screen.work_units == 1
    assert confirm.work_units == 4
    assert confirm.episode_done


with TemporaryDirectory() as directory:
    with Ledger(Path(directory) / "example.sqlite3") as ledger:
        asyncio.run(example(ledger))
```

This example creates only an isolated temporary ledger.
It spends no API or GPU credit.
Real runs must hash their actual environment identity.

## T1: autoresearch-mlx

`TrainingSource.load(repository)` verifies `train.py`, `prepare.py`, and `uv.lock` against the [recorded upstream pin](../research/autoresearch-mlx.md).
The revision is `766a25ff22afa799efd8d0aa450a4348e4749df2`.
The adapter changes an in-memory AST copy.
It does not change upstream files or frozen evidence.

The initial mutable surface consists of the `TrainingSetting` enum in [training.py](../src/openloop/loops/training.py).
It covers depth, model dimensions, attention windows, batch sizes, Adam settings, learning rates, and schedule ratios.
Arbitrary source edits are not accepted as candidate settings.
The sequence length remains 2048.
The final evaluation remains 1,572,864 validation tokens with batch size 256.
The upstream evaluator retains its memory-bounded microbatches.

| Fidelity | Training tokens | Updates at the default 65,536-token batch |
| --- | ---: | ---: |
| `"8m"` | 8,388,608 | 128 |
| `"21m"` | 20,971,520 | 320 |

These budgets are initial design choices, not calibrated run times.
Batch sizes must divide both fidelity budgets and gradient accumulation.
The adapter rejects partial final batches.
Every optimization update counts toward the budget, including the first update.
The learning-rate schedule uses trained tokens, rather than elapsed time.
The stopping rule uses trained tokens only.
The requested seed replaces the upstream fixed seed.

T1 uses the [frozen FineWeb-Edu snapshot](../data/fineweb-edu/README.md).
The earlier bring-up used upstream validation shard 6542.
T1 uses the frozen validation membership and therefore requires a new baseline.
It replaces the upstream document iterator with the frozen train and validation memberships.
Packing and tokenization retain the upstream implementation.
The adapter changes only constants, the seed, the token stop, the token-based progress and remaining-token expressions, the tokenizer path, and the time-budget message.
It removes the upstream evaluator import.
It does not edit log strings.
Raw shards contain documents from all splits.
The reader scans these shards but yields and tokenizes only the selected documents.
It does not open held-out membership files or accept held-out requests.
This reader is not a file access sandbox.

The existing upstream tokenizer is a frozen external asset.
It was trained on the upstream corpus, not solely on the new train membership.
Its pickle and token-byte lookup both enter the tokenizer hash.
The worker verifies those bytes before loading them.

### Native worker

Constructing T1 and generating jobs do not start training.
The explicit `T1.run` method starts a child process.
The caller must first approve native GPU training.
The `python` constructor argument names the worker interpreter.
The coordinator can run in any environment.
The worker interpreter must have the runtime synchronized with the pinned upstream `uv.lock`.
The openloop source must be importable in that runtime.
For this checkout, that interpreter is `research/autoresearch-mlx/.venv/bin/python`.
The path is machine-specific, so it does not enter the loop specification or any hash.
The worker checks installed versions against the locked dependencies, which `dependencies_hash` covers.
The worker rebuilds the loop with its own interpreter path and obtains an identical specification.

The worker checks the Python 3.12 dependency branch before importing MLX.
It rejects missing or mismatched packages.
It verifies the manifest snapshot hash, the membership file hashes, the raw shard sizes, and the tokenizer.
It does not hash raw shards, which total 6.4 GB.
The reader compares every document it yields with its membership digest, so corruption in any document that a job uses is rejected.
Corruption in an unused document does not affect the job.
`Corpus.verify()` hashes every raw shard and remains available for explicit audits.
Then it executes the generated training program.
Finally, the frozen evaluator measures the trained model on validation data.
Candidate settings cannot replace this evaluator or its data iterator.

The child receives only `PATH` and the openloop source path in its environment.
Upstream garbage-collector settings therefore affect the child, not the coordinator.
A one-hour wall limit stops stalled jobs independently of the token budget.
Cancellation kills the child and waits for termination.
A nonzero exit raises `ContractError`.
Its message contains the exit code and the last 4,000 characters of stdout and of stderr, each labelled.
Upstream prints `FAIL` to stdout before it exits, so the stdout tail is necessary.
The ledger failure reason therefore stays bounded.
This trusted worker is not the planned sandboxed executor.
It must receive reviewed source snapshots.

Native training and BPB scoring both occur inside the trusted job.
The ledger execution stage covers that job.
The ledger evaluation stage validates its receipt and publishes typed measurements.
The result records `val_bpb` as its metric.
Observed training seconds and tokens per second are `observations`, which comparisons ignore.
Training seconds include every trained update, including startup.
The throughput excludes setup and final validation.
T1 declares `noisy` reproducibility.
Cache references still represent prior samples; noise estimation requires explicit replications.

## Validation and remaining work

The [loop tests](../tests/loops) use tiny frozen data and CPU-only training stand-ins.
They check both fidelities, exact token stops, token schedules, seed replacement, cache semantics, rejected actions, and failure records.
They also check data memberships, tokenizer drift, dependency mismatches, and protected evaluation budgets.
Property-based tests check the planted optimum and independence from simulated machine speed.

A [golden test](../tests/loops/test_pinned_source.py) generates both fidelity programs from the real pinned source.
It uses the baseline configuration, seed 42, and tokenizer directory `/frozen/tokenizer`.
It compiles them and compares their SHA-256 digests with pinned literals.
It also checks the seed, the token stop, the token-based progress, the startup exclusion, and the absence of the evaluator.
It skips when `research/autoresearch-mlx` is absent or differs from the pin.

The [compile-only report](loop-validation.json) records successful compilation of the pinned upstream source at both fidelities on October 9, 2026.
Its installed runtime matched all 12 locked dependencies, including MLX 0.31.0.
Neither check executed GPU training.
Regenerate the report with this command.
It compiles the programs and imports no MLX.

```sh
PYTHONPATH=src research/autoresearch-mlx/.venv/bin/python -m openloop.loops.t1_report \
  --upstream research/autoresearch-mlx --corpus data/fineweb-edu \
  --tokenizer-dir ~/.cache/autoresearch/tokenizer --verify-runtime \
  --output docs/loop-validation.json
```

`--verify-runtime` checks installed versions with the worker's `verify_runtime`.
Omit it outside the upstream runtime.
The `adapter_hash` field changes whenever an adapter file changes, so regenerate the report after such a change.

A successful worker run returns only its receipt.
The worker log is discarded.
Artifact storage does not exist yet, so no run keeps its training log.
Native training, measured throughput, fidelity rank correlation, and the noise floor remain unverified.
Executor isolation, the shared GPU lane, statistical decisions, and held-out release remain separate tasks.
