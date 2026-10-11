# S1 part 1: T1 calibration

Prepared on October 10, 2026. Need: [N2](../../NEEDS.md#n1n10).
Status: [stopped at the owner's request](run-2026-10-10/stopped-by-owner.json) on October 10, 2026.
Two pilot runs completed. The third run was cancelled.
The owner previously approved the [frozen batch](approval-2026-10-10.json).
Local execution must not resume without a new instruction.
The task is not complete until all runs and the analysis finish.
No T1 noise estimate or fidelity rank correlation has been measured yet.
The partial [8m](run-2026-10-10/run-001.json) and [21m](run-2026-10-10/run-002.json) pilot scores are 2.159976 and 1.745289 BPB.
These single scores are not noise estimates.
The [interruption record](run-2026-10-10/interrupted.json) preserves the cancellation.

The [frozen plan](plan-2026-10-10.json) declares 70 local runs.
Its hash is `642d3befd545b3723bcc580f9f47260f6259eb52148bb5aec0f81e2bc99983ea`.
It fixes source, evaluator, data, tokenizer, environment, configurations, seeds, and run order.
The worker uses the pinned MLX 0.31.0 environment.
The observed host is an Apple M4 Pro with 24 GiB RAM.
Preflight verifies raw shard hashes and train/validation membership hashes.
It checks all locked package versions and compiles every training program.
It imports no MLX backend and starts no training.
It does not open the held-out membership file.

## Measurements and work

| Measurement | 8m runs | 21m runs | Seeds |
| --- | ---: | ---: | --- |
| Baseline execution variation | 10 | 10 | Ten separate executions at seed 42. |
| Baseline distinct-seed variation | 10 | 10 | Seeds 1000 through 1009. |
| Five additional ranking candidates | 15 | 15 | Three matched seeds per candidate: 1000, 1001, 1002. |

This gives 20 baseline runs per fidelity.
The six-candidate rank panel also includes the baseline's first three distinct-seed runs.
Those runs do not execute again or count twice in a noise estimate.
Every new execution has purpose `replication`, including fixed-seed runs.
Cache aliases are not accepted as new observations.

The ranking candidates change one baseline setting each:

| Candidate | Setting |
| --- | --- |
| `baseline` | Pinned upstream settings. |
| `matrix_lr_half` | `MATRIX_LR = 0.02` |
| `matrix_lr_double` | `MATRIX_LR = 0.08` |
| `embedding_lr_half` | `EMBEDDING_LR = 0.3` |
| `embedding_lr_double` | `EMBEDDING_LR = 1.2` |
| `weight_decay_zero` | `WEIGHT_DECAY = 0.0` |

The candidate panel is an initial calibration panel.
It is not a sample of all possible training changes.
It cannot establish rank preservation for new architectures or other datasets.
Do not change it after scores become available.

Training work is capped at 1,027,604,480 tokens.
Each run also evaluates 1,572,864 validation tokens.
The full evaluation work is 110,100,480 tokens.
The first two jobs are a baseline run at each fidelity.
The remaining order is shuffled with a fixed seed to spread time drift.
One local executor serves one GPU lane.
The study has a 12-hour wall cap; each job also retains T1's one-hour wall cap.
Cancellation stops the active worker.
The study stops on the first failed job.
It never drops failed jobs to make a complete report.

No API calls or cloud GPU jobs are used.
Host GPU occupancy and wall time still require approval under [AGENTS.md](../../AGENTS.md).
Earlier ClimbMix runs are not samples for this FineWeb-Edu study.
Their source, evaluator, token stopping rule, and data differ.
Their throughput is not a measured duration for this plan.

## Run and records

Prepare a new plan without training:

```sh
uv run python research/calibrate_t1.py \
  --upstream research/autoresearch-mlx \
  --corpus data/fineweb-edu \
  --tokenizer "$HOME/.cache/autoresearch/tokenizer" \
  --worker-python "$PWD/research/autoresearch-mlx/.venv/bin/python" \
  --plan research/s1/plan-2026-10-10.json
```

The prepared file already exists; this command refuses to overwrite it.
Use a new filename for a new plan.
After approval, repeat the command with:

```sh
  --execute-approved --output research/s1/run-2026-10-10
```

Execution repeats preflight and checks equality with the prepared plan.
The output directory must be new.
The runner uses `ExperimentEnv`, `T1`, and `LocalExecutor` through their public APIs.
It creates a separate `ledger.sqlite3` in that output directory.
It does not change any earlier ledger or frozen evidence.
It records each completed run in a separate JSON receipt and preserves executor logs.
The ledger retains unsuccessful attempts.
An interrupted study records its error and completed-run count.
It does not produce the full analysis report.
Automatic retries and resumption are not implemented.
Do not rerun the whole study silently after an interruption.

## Analysis

The [pure analysis](../../src/openloop/studies/s1.py) requires all planned successes.
It rejects missing runs, changed inputs, failed attempts, cache aliases, duplicate attempts, and held-out scores.
It reports raw validation BPB, evidence references, sample means, and sample standard deviations with denominator `n - 1`.
The fixed-seed and distinct-seed estimates are separate.
Distinct-seed variation includes both seed effects and execution variation.
It is not an isolated estimate of seed effects.
Ten observations per noise type give limited precision.

The fidelity statistic is Spearman correlation across the six candidate means.
Each candidate mean uses the same three seeds at each fidelity.
Tied values use average ranks.
Constant rankings produce an absent correlation.
Ranking baseline seeds alone would measure noise agreement, not candidate rank preservation.

Noise intervals use 2,000 deterministic bootstrap draws within each noise group.
The rank interval uses 2,000 bootstrap draws of matched seed blocks.
Each draw uses the same seed indices for all candidates and both fidelities.
This preserves pairing.
These are descriptive 95% percentile intervals, not exact coverage guarantees.
Three seed blocks and six candidates give a preliminary rank check only.
An undefined bootstrap rank makes the interval absent; it is not silently dropped.
No hypothesis promotion or held-out claim occurs in S1 part 1.

The tests use synthetic ledger scores and CPU fixtures only.
They check run counts, token caps, separate noise groups, ties, correlation, invalid evidence, and the prepare-only CLI.
Native pilot execution is verified at both fidelities.
The full noise and rank measurements remain incomplete after the owner's stop request.
