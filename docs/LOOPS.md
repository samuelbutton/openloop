# Loop interface

Implemented on October 9, 2026. Needs: N1, N2, N4, N8, and N9 in [NEEDS.md](../NEEDS.md).

`openloop.loops` defines a shared `LoopEnv` contract with `initial_observation` and `step`.
`ExperimentEnv` implements this contract through the public ledger API.
One instance serves one fixed candidate.
It has no reset method.

## Contract

| Record | Content |
| --- | --- |
| `LoopSpec` | Frozen source, dependencies, data, evaluator, tokenizer, execution settings, mutable keys, metrics, and two fidelities. |
| `LoopAction` | Seed, fidelity, request key, execution purpose, and optional retry reference. |
| `Observation` | Candidate hash, evidence references, remaining work allowance, unit, and permitted fidelities. |
| `Job` | Immutable experiment inputs and an optional generated training program. |
| `StepResult` | Observation, submission and attempt references, measured result, observed work, execution flag, and episode completion flag. |

The candidate configuration and parents enter the environment constructor.
They remain fixed during the episode.
The action's fidelity also selects the experiment's phase.
Selection actions permit screening and confirmation only.
Held-out evaluation is a later task.

Before admission, the environment checks the candidate, job, and remaining allowance.
It reserves the full fidelity amount before awaiting the runner.
It checks the frozen specification again before publishing measurements.
Failures and cancellation preserve a failed attempt and the reservation.
They do not produce a successful sample.
These are experiment-work limits, not money limits.

The environment records execution and evaluation stages in the ledger.
Completed cache references consume no new work.
Pending cache references return an unresolved run reference without executing duplicate work.
Use `ledger.get(run_id)` to inspect that run later.
An explicit replication creates a new attempt, including for noisy T1 inputs.
A request-key replay returns its original step snapshot.
It does not execute or reserve work again.

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

Screening uses one observation.
Confirmation averages four observations.
Its noise standard deviation is half the screening standard deviation.
The budget unit is `observations`.
Identical seeds reproduce identical samples, so the specification declares `deterministic` reproducibility.
Different seeds supply independent draws for later statistical tests.
T0 results are labeled as simulated in the input manifest.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from openloop.ledger import Ledger, content_hash
from openloop.loops import ExperimentEnv, LoopAction, T0


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
    screen = await env.step(LoopAction(seed=42, fidelity="screen"))
    confirm = await env.step(LoopAction(seed=43, fidelity="confirm"))
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
| Screen | 8,388,608 | 128 |
| Confirm | 20,971,520 | 320 |

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
The explicit `T1.run` method starts a child process using the active Python interpreter.
The caller must first approve native GPU training.
Use the runtime synchronized with the pinned upstream `uv.lock`.
The openloop source must be importable in that runtime.
For this checkout, that runtime is `research/autoresearch-mlx/.venv/bin/python` with `src` on `PYTHONPATH`.

The worker checks the Python 3.12 dependency branch before importing MLX.
It rejects missing or mismatched packages.
It verifies the frozen train/validation files and tokenizer.
Then it executes the generated training program.
Finally, the frozen evaluator measures the trained model on validation data.
Candidate settings cannot replace this evaluator or its data iterator.

The child receives only `PATH` and the openloop source path in its environment.
Upstream garbage-collector settings therefore affect the child, not the coordinator.
A one-hour wall limit stops stalled jobs independently of the token budget.
Cancellation kills the child and waits for termination.
This trusted worker is not the planned sandboxed executor.
It must receive reviewed source snapshots.

Native training and BPB scoring both occur inside the trusted job.
The ledger execution stage covers that job.
The ledger evaluation stage validates its receipt and publishes typed measurements.
The result records `val_bpb`, observed training seconds, and tokens per second.
Training seconds include every trained update, including startup.
The throughput excludes setup and final validation.
T1 declares `noisy` reproducibility.
Cache references still represent prior samples; noise estimation requires explicit replications.

## Validation and remaining work

The [loop tests](../tests/loops) use tiny frozen data and CPU-only training stand-ins.
They check both fidelities, exact token stops, token schedules, seed replacement, cache semantics, rejected actions, and failure records.
They also check data memberships, tokenizer drift, dependency mismatches, and protected evaluation budgets.
Property-based tests check the planted optimum and independence from simulated machine speed.

The [compile-only report](loop-validation.json) records successful compilation of the pinned upstream source at both fidelities on October 9, 2026.
Its installed runtime matched all 12 locked dependencies, including MLX 0.31.0.
Neither check executed GPU training.
Native training, measured throughput, fidelity rank correlation, and the noise floor remain unverified.
Executor isolation, the shared GPU lane, statistical decisions, and held-out release remain separate tasks.

## Files in this change

- `src/openloop/loops/__init__.py`: public API.
- `src/openloop/loops/models.py`: contracts and immutable records.
- `src/openloop/loops/env.py`: ledger orchestration.
- `src/openloop/loops/files.py`: read-only source fingerprints.
- `src/openloop/loops/t0.py`: planted objective and seeded samples.
- `src/openloop/loops/training.py`: pure token-budget transformation.
- `src/openloop/loops/corpus.py`: frozen membership reader.
- `src/openloop/loops/t1.py`: MLX adapter and child-process runner.
- `src/openloop/loops/t1_worker.py`: runtime checks and trusted scoring.
- `tests/loops/conftest.py`: offline source and data fixtures.
- `tests/loops/test_env.py`: episode tests.
- `tests/loops/test_models.py`: contract identity and validation tests.
- `tests/loops/test_t0.py`: planted-truth tests.
- `tests/loops/test_training.py`: token-control properties.
- `tests/loops/test_corpus.py`: frozen membership tests.
- `tests/loops/test_t1.py`: adapter and worker tests.
- `docs/LOOPS.md`: this guide and implementation decisions.
- `docs/loop-validation.json`: compile-only and installed-version evidence.
- `README.md`: package entry point.
- `DESIGN.md`: implemented and planned boundaries.
