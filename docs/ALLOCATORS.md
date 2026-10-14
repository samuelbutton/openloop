# Allocation plugins

Implemented on October 12, 2026. These plugins address [N3](../NEEDS.md#n1n10).
The source is in [`openloop.allocators`](../src/openloop/allocators/).
The [offline tests](../tests/allocators/) check the policies through T0 and a small NAS-Bench-201 fixture.
These checks are simulations. They do not establish a policy's search performance.

## Contract

`Allocator.select(snapshot) -> Allocation` is a pure function.
The snapshot contains one frozen `LoopSpec`, one environment hash, offered `Job` records, and public ledger `Run` records.
It also contains the remaining work budget and the number of free worker slots.
All records and policy parameters are immutable.
The caller must construct jobs through the trusted workload adapter.
The allocator checks each manifest against the loop specification.
It does not check candidate programs or read benchmark outcomes.

An allocation contains ordered jobs, a policy hash, `work_units`, a `status`, and a `reason`.
The hash includes the policy type, every declared parameter, and allocator version 2.
It is a policy identity, not a decision record in the ledger.
No allocation can exceed the snapshot's remaining budget or free slots.
Every job costs its full declared fidelity amount.
Training tokens, epochs, and lookup counts are separate units. They are not currency costs.

The caller must supply complete, current ledger evidence for this loop and environment.
Complete means every run: pass `Ledger.query_all()`, never a paginated `Ledger.query(limit=...)` page.
A truncated list makes finished work look fresh, and the coordinator would run it again.
Other scopes are rejected. Confirmation results are excluded from selection evidence.
Only validation metrics with the declared name, unit, and direction can supply scores.
Cache aliases count once per attempt. Genuine additional attempts count separately.
Cached, queued, running, and failed input hashes are excluded from new allocations.
A failure does not supply a score. The coordinator can retry failed work through the core API.
An allocation only selects new screen work. The decider controls confirmation and final evaluation.

The coordinator must reserve the batch budget before dispatch.
It must rebuild the snapshot after admission and completion.
The snapshot's remaining budget excludes committed work: jobs that are queued or running, and any retry the coordinator queued.
Allocators price only jobs that have not started.
These plugins do not implement durable reservations, worker dispatch, or crash recovery.
Two callers can select the same jobs from the same stale snapshot.
The existing ledger and executor admission rules remain necessary.
An allocation never establishes a keep or discard decision.

### Allocation status

Every allocation states why it holds the jobs it holds, or none.
`AllocationStatus` has these members:

| Status | Meaning | Final? |
| --- | --- | --- |
| `selected` | The allocation holds jobs. | No |
| `done` | The policy has nothing left to run. A halving bracket is complete, or no fresh job exists. | Yes |
| `waiting` | Jobs the rung needs are queued or running. | No |
| `blocked` | A failed input needs a retry (`on_failure="block"`). | No |
| `budget_exhausted` | The remaining budget cannot fund the next step. | No |
| `no_slots` | No worker slot is free (`slots=0`). | No |
| `missing_jobs` | The caller did not offer a job that the policy needs. Nothing starts. | No |

Only `selected` holds jobs. Greedy, random, and bandit policies report `done`, `budget_exhausted`, or `no_slots` when they hold none.

## Policies

| Policy | Frozen parameters | Selection rule |
| --- | --- | --- |
| `Greedy` | Fidelity name | Select candidates without measurements first. Then prefer the best mean validation metric at that fidelity. |
| `RandomSearch` | Fidelity name and nonnegative seed | Rank fresh inputs by a hash of the seed and full input hash. Caller order does not affect the order. |
| `Bandit` | Fidelity name and raw metric lower and upper bounds | Normalize the directed score to [0, 1]. Rank by the UCB1 index. Select candidates without measurements first. |
| `SuccessiveHalving` | Candidate hashes, `Rung` (fidelity and seeds) per rung, integer `eta >= 2`, and `on_failure` | Complete a rung for every active candidate. Keep `max(1, floor(n / eta))` candidates for the next rung. |
| `Hyperband` | Candidate pool, rungs, `eta`, random seed, and `on_failure` | Freeze disjoint random halving brackets. Run them in order, from the lowest starting fidelity to the highest. |

Greedy and Bandit compare only one fidelity and candidates in the offered pool.
Different fidelities never enter one mean or UCB index.
Input hashes break their score ties. Candidate hashes break promotion ties.
Fresh seeds supply more measurements. Allocation does not invent seeds or remap them.
The caller must declare seeds that the workload accepts.

`RandomSearch` ranks exactly the jobs it receives.
The caller decides whether that pool holds distinct candidates or replicates of one candidate.
The class is not named `Random`, so it does not shadow `random.Random`.

Bandit uses `mean(reward) + sqrt(2 * log(total_pulls) / candidate_pulls)`.
The bounds must be declared before selection.
An out-of-range score is clipped to the bounds, never raised.
The allocation `reason` counts the clipped scores, so the event stays visible.
A clipped score is a lower bound on the real reward, so clipping keeps the index finite and the run alive.
Widen the bounds in a new policy if clipping occurs often.
The [UCB1 paper](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf) gives the bounded-reward rule.
UCB1 minimizes cumulative regret over repeated pulls.
It spends pulls on promising arms, and it does not identify the best arm or give an error bound.
Use the decider to certify a winner.
Parallel batches use the same completed evidence for all jobs in the batch.
They do not have the paper's sequential regret guarantee.
Repeated deterministic lookups do not provide independent noise samples.

Halving uses the first completed successful attempt for each candidate's declared rung seed.
Later replications cannot revise that score.
Each rung declares a fidelity and one or more distinct seeds.
The seeds can differ between rungs.
There is no checkpoint resumption. A 12-to-200 epoch NAS promotion costs 200 more epochs.

### Rung-atomic budget

A rung is useful only when every active candidate has a score.
A partly funded rung wastes its compute, because no promotion follows.
Halving therefore starts a rung only if the remaining budget covers the full cost of all jobs in it that have not started.
Otherwise it selects nothing and reports `budget_exhausted`.

The snapshot defines committed work.
Jobs that are queued or running are already reserved by the coordinator, so they are excluded from the remaining budget and are not priced again.
A rung that is partly started stays consistent: each call prices only the jobs still unstarted.
Free slots can still split a rung over several calls, and the budget check holds on each call.
The check covers one rung. It does not reserve budget for later rungs.
A bracket can still stop at a later rung with `budget_exhausted`.
Compare the declared bracket cost with the budget before you start if that matters.
Retries of failed inputs are the coordinator's operations. Their cost is not in this check.

Hyperband runs brackets in order, and it stops at the first rung that it cannot afford.
It does not skip to a cheaper later bracket.
A skip would spend the budget on a different mix of brackets than the declared plan, and equal-compute comparisons between runs would stop being interpretable.

### Failure policy

`on_failure` is a declared parameter, and it enters the policy hash.

| Value | Effect of a terminal failure at a rung |
| --- | --- |
| `block` (default) | The rung waits until the coordinator retries the input. Status is `blocked`. |
| `rank_worst` | The candidate ranks below every scored candidate and is never promoted. The allocator starts none of its other seeds at that rung. |

A failure is terminal when the input has a failed attempt and no queued, running, or completed attempt.
In both modes the failure stays in the ledger, and it is never a score or a statistical sample.
Under `rank_worst`, the promotion count stays `floor(n / eta)` of the candidates that entered the rung.
Failed candidates do not fill the freed places, so a rung can promote fewer candidates.
If every candidate fails, the bracket ends with `done`.

The default is `block`.
A failure is often infrastructure: a killed worker, an out-of-memory error, a full disk.
That is not evidence against the candidate ([CONTRIBUTING.md](../CONTRIBUTING.md#statistical-honesty)), and `rank_worst` would drop a good candidate for it.
Choose `rank_worst` only when the failure is a property of the candidate, such as divergence at high learning rate.
Then a retry would repeat the same failure, and blocking the cohort would stall the bracket for nothing.

### Multiple seeds per rung

A one-sample rung promotes on one noisy score.
The best of n noisy scores is biased high: this is the winner's curse.
At the T1 noise of about 0.027 BPB, that selection effect is large next to the gaps between candidates.
A rung with several seeds promotes on the mean of those seeds.
Every seed must complete, or fail under the failure policy, before the rung promotes.
For example, `Rung("8m", (0, 1, 2))` costs three times a one-seed rung at the same fidelity.
The mean has about `1/sqrt(k)` of the single-seed noise for `k` seeds.

The trade-off is cost against selection bias.
More seeds per rung cost proportionally more budget and leave fewer candidates for a fixed budget.
Fewer seeds cover more candidates but promote more lucky candidates.
A mean over seeds reduces the bias but does not remove it.
Neither choice tests a hypothesis.
Confirmation does that, with fresh seeds.
A single-seed rung stays valid: `Rung("1x", (0,))`.

Hyperband follows the bracket counts in [Algorithm 1](https://jmlr.org/papers/volume18/16-558/16-558.pdf).
For `L` declared rungs, bracket `s` starts at rung `L - 1 - s` with `ceil(L * eta**s / (s + 1))` candidates.
The last bracket is random search at the highest fidelity.
The candidate pool must be large enough for all disjoint brackets.
The plan samples without replacement. It does not silently reduce bracket sizes.
`brackets(spec)` exposes this plan before execution.

This is a finite-pool, declared-grid Hyperband variant.
Geometric grids reproduce the published resource ratios and bracket counts.
T1's 8m/21m grid and NAS-Bench-201's 12/200 grid are not geometric.
The variant uses those exact amounts. Its brackets need not consume equal work.
It does not claim the paper's theoretical guarantees on these grids.
For a single fidelity, Hyperband has one random-search bracket with one candidate.
Use `RandomSearch` for a larger single-fidelity search, such as HPO-B.

## Screening for the decider

A finished halving or Hyperband run is a screen for the [decider](DECIDER.md#allocator-screens).
`SuccessiveHalving.outcome(snapshot)` and `Hyperband.outcome(snapshot)` return a `ScreenOutcome` once every bracket is complete.
They raise `ContractError` before then.

- `candidates` is the whole declared pool. Declare it before allocation starts.
- `survivors` are the candidates that finished the final rung, best first.
  Pass `survivors=k` to keep the best k.
  Final-rung scores rank across Hyperband brackets, because every bracket ends at the same rung.
  A candidate that failed under `rank_worst` is not a survivor.
- `evidence` maps each candidate to the ledger run ids that scored it.
  Candidates that the allocator never ran have no references.
- `seeds` is every seed that the declared rungs use.

Record the outcome with `Campaign.record_external_screen`, once for each candidate, under `Policy.ALLOCATOR`.
The campaign enforces the order and checks the seeds.
Allocator seeds must not meet the confirmation, held-out, or confirmation-noise seeds.

The error bound survives because confirmation uses fresh seeds, and the pool and the survivor set are fixed before any confirmation score exists.
An allocator promotion is not a test, and it carries no error rate.
[DECIDER.md](DECIDER.md#why-the-error-bound-holds) states the argument and its limits.
[`tests/studies/test_t0_allocator.py`](../tests/studies/test_t0_allocator.py) runs the whole path on T0.
That test is simulated and small.
It checks the plumbing and one planted gain. It does not measure error rates.

## Offline example

This example runs only simulated T0 work. It uses a temporary ledger.
Construct cohort hashes from normalized, built jobs.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from openloop.allocators import AllocationSnapshot, Hyperband, Rung
from openloop.ledger import Ledger
from openloop.loops import T0, ExperimentEnv, LoopAction

loop = T0()
environment = "a" * 64
rungs = (Rung("1x", (0,)), Rung("4x", (1,)))
configs = tuple(loop.normalize_config({"coordinates": [i, 0]}) for i in range(6))
jobs = tuple(
    loop.build_job(loop.spec.inputs(config, LoopAction(seed, rung.fidelity), environment))
    for config in configs
    for rung in rungs
    for seed in rung.seeds
)
policy = Hyperband(tuple(loop.spec.candidate_hash(c) for c in configs), rungs, seed=42)

with TemporaryDirectory() as directory, Ledger(Path(directory) / "runs.sqlite3") as ledger:
    envs = {
        loop.spec.candidate_hash(config): ExperimentEnv(
            loop, ledger, runner=loop.run, config=config,
            environment_hash=environment, budget=100,
        )
        for config in configs
    }
    remaining = 100
    while True:
        snapshot = AllocationSnapshot(
            loop.spec, environment, jobs, ledger.query_all(), remaining, slots=2,
        )
        allocation = policy.select(snapshot)
        if not allocation.jobs:
            break
        remaining -= allocation.work_units  # Reserve before dispatch.
        for job in allocation.jobs:
            inputs = job.inputs
            asyncio.run(envs[inputs.candidate_hash].step(
                LoopAction(inputs.seed, inputs.fidelity),
            ))
```

Native T1 runs, S2 policy comparisons, and mock-executor scale projections remain unverified.
