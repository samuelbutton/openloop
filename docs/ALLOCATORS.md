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

An allocation contains ordered jobs, a policy hash, and `work_units`.
The hash includes the policy type, parameters, and allocator version 1.
It is a policy identity, not a decision record in the ledger.
No allocation can exceed the snapshot's remaining budget or free slots.
Every job costs its full declared fidelity amount.
Training tokens, epochs, and lookup counts are separate units. They are not currency costs.

The caller must supply complete, current ledger evidence for this loop and environment.
Other scopes are rejected. Confirmation results are excluded from selection evidence.
Only validation metrics with the declared name, unit, and direction can supply scores.
Cache aliases count once per attempt. Genuine additional attempts count separately.
Cached, queued, running, and failed input hashes are excluded from new allocations.
A failure does not supply a score. The coordinator can retry failed work through the core API.
An allocation only selects new screen work. The decider controls confirmation and final evaluation.

The coordinator must reserve the batch budget before dispatch.
It must rebuild the snapshot after admission and completion.
These plugins do not implement durable reservations, worker dispatch, or crash recovery.
Two callers can select the same jobs from the same stale snapshot.
The existing ledger and executor admission rules remain necessary.
An empty allocation can mean completion, a pending rung, missing offered jobs, failed work, or insufficient budget.
It does not establish a keep or discard decision.

## Policies

| Policy | Frozen parameters | Selection rule |
| --- | --- | --- |
| `Greedy` | Fidelity name | Select candidates without measurements first. Then prefer the best mean validation metric at that fidelity. |
| `Random` | Fidelity name and nonnegative seed | Rank fresh inputs by a hash of the seed and full input hash. Caller order does not affect the order. |
| `Bandit` | Fidelity name and raw metric lower and upper bounds | Normalize the directed score to [0, 1]. Rank by the UCB1 index. Select candidates without measurements first. |
| `SuccessiveHalving` | Candidate hashes, screen action per rung, and integer `eta >= 2` | Complete a rung for every active candidate. Keep `max(1, floor(n / eta))` candidates for the next rung. |
| `Hyperband` | Candidate pool, rung actions, `eta`, and random seed | Freeze disjoint random halving brackets. Run them in order, from the lowest starting fidelity to the highest. |

Greedy and Bandit compare only one fidelity and candidates in the offered pool.
Different fidelities never enter one mean or UCB index.
Input hashes break their score ties. Candidate hashes break promotion ties.
Fresh seeds supply more measurements. Allocation does not invent seeds or remap them.
The caller must declare seeds that the workload accepts.

Bandit uses `mean(reward) + sqrt(2 * log(total_pulls) / candidate_pulls)`.
The bounds must be declared before selection. Out-of-range evidence is rejected, not clipped.
The [UCB1 paper](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf) gives the bounded-reward rule.
Parallel batches use the same completed evidence for all jobs in the batch.
They do not have the paper's sequential regret guarantee.
Repeated deterministic lookups do not provide independent noise samples.

Halving uses the first completed successful attempt for each candidate's declared rung action.
Later replications cannot revise that score.
Each rung has an explicit fidelity and seed. The seeds can differ between rungs.
A pending or failed candidate blocks promotion until its rung has a successful result.
Already recorded successful measurements can close a rung without new work.
There is no checkpoint resumption. A 12-to-200 epoch NAS promotion costs 200 more epochs.

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
Use `Random` for a larger single-fidelity search, such as HPO-B.

## Offline example

This example runs only simulated T0 work. It uses a temporary ledger.
Construct cohort hashes from normalized, built jobs.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from openloop.allocators import AllocationSnapshot, Hyperband
from openloop.ledger import Ledger
from openloop.loops import T0, ExperimentEnv, LoopAction

loop = T0()
environment = "a" * 64
rungs = (LoopAction(0, "1x"), LoopAction(1, "4x"))
configs = tuple(loop.normalize_config({"coordinates": [i, 0]}) for i in range(6))
jobs = tuple(
    loop.build_job(loop.spec.inputs(config, action, environment))
    for config in configs
    for action in rungs
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
            loop.spec, environment, jobs, ledger.query(), remaining, slots=2,
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
