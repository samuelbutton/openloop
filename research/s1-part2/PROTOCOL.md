# S1 part 2 protocol: M4 Pro lineage replication

Pre-registered October 12, 2026, before any audit run.
Plan hash: `df0ff4818c75396ad176a72460a1a1c57a436c544fcbf92de7b61ddbd402357b` ([plan](plan-2026-10-12.json)), superseded by the [Amendment (October 13, 2026)](#amendment-october-13-2026) below, plan hash `5f0badf3342f156eed23b965cb2603da59c7ae6d02a2ec822d242f81a91c9e43` ([plan](plan-2026-10-13.json)).
Need: [N2](../../NEEDS.md#n1n10). Study: S1, part 2.

This is a replication on one machine.
It does not judge the authors.
The fork author's published runs used a 16-core GPU.
This machine has an Apple M4 Pro with 20 GPU cores and 24 GiB.
The time budget fixes seconds, not steps, so faster hardware takes more steps.
Absolute BPB differences from the publication are expected.
The October 10 [reproduction](../reproductions/m4-pro-2026-10-10/README.md) of the winner gave 1.410165 against 1.429396 published.

## Claims under test

Source: the [Mar 14, 2026 M4 Pro wiki page](https://github.com/elementalcollision/autoresearch/wiki/Experiment-Results-Mar-14-2026-M4-Pro/3337ce96ae8c5410fa79088a4273804d7f33bdd8)
at revision `3337ce96ae8c5410fa79088a4273804d7f33bdd8`, page SHA-256 `c25ecb0521459beb1095a1b74ae88db5622ed8680edfc9110676ffe0275e8061`.
Each published number is one run at seed 42.

| Arm | Commit | Published BPB | Published status |
| --- | --- | ---: | --- |
| `baseline` | `5d903d0027c39cb37b7dd9d9b70151c64756262a` | 2.026063 | baseline ("Unmodified master, Pro tier defaults") |
| `bc5cdc1` | `bc5cdc17092555a96a4a3dfd6eb985c512d67018` | 1.439990 | keep (#1) |
| `1d83776` | `1d8377683d9a5b1af14f5933e6f5130c54e94460` | 1.429396 | keep, the published winner (#4) |
| `27b9d9f` | not in the public repository | 1.429607 | discard, "tied" (#7, MLP ratio 1.0) |
| `000e04d` | not in the public repository | 1.430230 | discard, "tied" (#11, softcap 12) |
| `8f00506` | not in the public repository | 1.432641 | discard, "slightly worse" (#9b, warmdown 0.65) |
| `cc51630` | not in the public repository | 1.432399 | discard, "slightly worse" (#10a, window "SL") |

The baseline commit is the parent of `bc5cdc1` on master, and `bc5cdc1` is the parent of `1d83776`.
Both facts were checked in the fork's history.
The history does not contain `27b9d9f`, `000e04d`, `8f00506`, or `cc51630`.
Those commits were never pushed to any public branch.
GitHub's API also reports no commit for these IDs.
Their parent therefore cannot be read from history.
The log lists them after experiment 4, whose commit is the incumbent best.
The audit assumes each was tested as a one-line change from `1d83776`.
That is an assumption, not a verified fact.

### Reconstructed arms

The four missing arms are rebuilt from the `1d83776` source with one exact text replacement each.
Each replacement must match exactly once or the run fails.

| Arm | Replacement in `train_mlx.py` |
| --- | --- |
| `27b9d9f` | `MLP_RATIO = 1.5` becomes `MLP_RATIO = 1.0` |
| `000e04d` | `softcap = 15` becomes `softcap = 12` |
| `8f00506` | `WARMDOWN_RATIO = 0.7` becomes `WARMDOWN_RATIO = 0.65` |
| `cc51630` | `WINDOW_PATTERN = "SSSL"` becomes `WINDOW_PATTERN = "SL"` |

The reconstructions follow the wiki's descriptions.
They are not the authors' bytes, so each "not a win" claim is tested for the described change, not the exact lost commit.
Reports label these arms "reconstructed".

## Comparisons

Six comparisons are registered.
Gain is parent BPB minus child BPB, so positive means the child improved.

| # | Child | Parent | Claim |
| --- | --- | --- | --- |
| 1 | `bc5cdc1` | `baseline` | win |
| 2 | `1d83776` | `bc5cdc1` | win |
| 3 | `27b9d9f` | `1d83776` | not a win (tied) |
| 4 | `000e04d` | `1d83776` | not a win (tied) |
| 5 | `8f00506` | `1d83776` | not a win (slightly worse) |
| 6 | `cc51630` | `1d83776` | not a win (slightly worse) |

No other comparison is registered.
Any further contrast is exploratory and must be labeled that way.

## Procedure

Each run executes the arm's own `train_mlx.py`, `prepare.py`, evaluator, Muon backend, and 300-second training budget.
The data is the original ClimbMix cache: ten training shards plus validation shard 6542.
Its 13 file hashes are checked against the [recorded manifest](../artifacts/autoresearch-mlx/data_manifest.json) before the study and after every run.
The validation set is the unchanged 5,242,880-token evaluation.
No held-out data is used in phase A.

The only change to any program is the training seed.
Each upstream script contains exactly one call, `mx.random.seed(42)`.
The runner replaces it with `mx.random.seed(N)`.
It fails unless the text contains exactly one `.seed(` call.
The same transformation applies to every arm.
The hash of each transformed program is stored in its run record and ledger entry.
The seed sets model initialization.
The upstream data loader has no random shuffle, so every seed sees identical training data in identical order.
Measured seed spread therefore excludes data-order variation, and the intervals below are narrower than a full-variance estimate.

Seeds are 2000 through 2004 for every arm, a paired design.
They are disjoint from the publication's seed 42 and from the T1 calibration seeds.
This gives 7 arms times 5 seeds, or 35 runs.
Run order is a fixed shuffle (shuffle seed 20261012) that interleaves arms to spread thermal and time drift.
The order is in the plan.
One GPU lane serves all runs.
Each job has a 1800-second wall limit, a 23 GiB resident-memory limit sampled every 5 seconds, and 16 MiB per log stream.
The wall limit exceeds the 900-second limit used on October 10 because the baseline arm (64K-token batch) has no measured timing and the published log reports it exceeding physical memory (32 GB on a 24 GB machine).
The wall limit only stops pathological runs.
Training time is the script's own 300-second budget.

Each run is a ledger experiment with purpose `replication` and reproducibility `noisy`.
The ledger is new and lives in the output directory.
Each run also writes a JSON receipt.
Receipts hold the executor job ID, state, times, resident memory, log hashes, and the parsed summary: BPB, steps, and training seconds.
Executor logs stay in the output directory.
Failures, timeouts, and interruptions are recorded, not hidden.
A failed run is an implementation or environment event, not evidence about a hypothesis.
The study stops after three consecutive failures or a 12-hour cap.
Step count is recorded for every run, because the budget fixes time and hardware speed sets steps.

## Analysis

The analysis is [`openloop.studies.claims_audit`](../../src/openloop/studies/claims_audit.py), written and tested before any run.
For each comparison it reports:

- paired gains per seed, restricted to seeds that succeeded in both arms;
- mean gain, median gain, and sample standard deviation of gains;
- the exact paired sign test p-value (`openloop.decider.sign_p_value`), raw and Bonferroni-adjusted across the 6 comparisons;
- the paired t-test p-value, raw and Bonferroni-adjusted;
- a 95% t interval for the mean gain and a Bonferroni-adjusted interval at level 1 - 0.05/6;
- the published single-run delta, and whether the AutoScientists gate (`openloop.decider.autoscientists_gate`) would promote that delta given the parent arm's measured seed standard deviation;
- each arm's mean BPB, mean steps, and sample standard deviation across seeds, which is the arm-level noise estimate.

Five paired seeds cannot give significance from the sign test alone after correction.
The smallest possible exact p-value with five pairs is 1/32 = 0.031, and times 6 it is 0.19.
The sign test is reported as a consistency check.
The decision rules use the Bonferroni-adjusted interval and require all pairs to agree in sign for a win.

## Decision rules

Let L and U be the Bonferroni-adjusted interval for the mean gain.

For claimed wins (comparisons 1 and 2):

- **replicated**: L > 0 and the raw exact sign p-value is at most 0.05, which means all five pairs improved;
- **reversed**: U < 0;
- **not replicated**: neither.

For claimed not-wins (comparisons 3 to 6):

- **actually worse**: U < 0;
- **actually better**: L > 0;
- **indistinguishable**: the interval contains 0.

"Indistinguishable" is low-powered with five seeds.
Reports give the interval, so readers see which differences the data excludes.
Equivalence is not claimed.

The headline numbers are the share of the 2 claimed wins that replicate and, for the 4 not-wins, the counts in each category.
The report also states which published deltas the gate would not promote.
Fewer than 3 paired seeds in a comparison yields "insufficient data".

## Limits

- One machine, one chip, and one macOS and MLX version.
  Its 20 GPU cores differ from the publication's 16.
- Four arms are reconstructions.
- The published run order, hardware state, MLX version, and package versions are unknown.
  The upstream `uv.lock` does not pin MLX.
  Observed versions are in the plan.
- Seeds vary initialization only (see Procedure).
- Five seeds give wide intervals.
  A non-replication means this test did not detect the effect, not that the effect is absent.
- Phase A uses the same validation shard that the authors used for selection.
  It tests replication, not generalization.
- The wiki's baseline row lists 9 training steps.
  The program cannot stop before 11 steps, because its stopping rule is `step > 10`.
  This audit records the discrepancy and does not resolve it.

## Phase B: held-out evaluation (superseded by Amendment 2)

Phase B would score each final model on a ClimbMix shard that no lineage run used.
Training used shards 0 to 9 and validation used shard 6542.
The proposed shard is `shard_05000.parquet`, chosen now by the author before any result.
Its size is 91,420,038 bytes (about 87 MiB), from an HTTP HEAD request.
Nothing has been downloaded.

Phase B needs owner approval for the download.
It also needs new code.
The runs above save no checkpoints.
Phase B therefore needs either of these, decided **before** launching the 35 runs:

1. A second recorded program transformation that appends an in-process evaluation on the held-out shard after the existing final evaluation, using the original evaluator, and logs that score.
   The shard must live outside `~/.cache/autoresearch/data`.
   The upstream loader treats every parquet file in that directory as training data.
2. A transformation that saves final weights, plus a separate evaluation script.
   Weights need about 90 MB per run.

Without either, phase B means repeating all 35 runs.
Phase A as registered here does neither.
The held-out score would be read once, after the phase A analysis closes.

## Ethics and release

The fork author has not reviewed these results.
Before any public post, share the report with the maintainer of `elementalcollision/autoresearch` and allow time to respond.
Describe the results as a replication on different hardware.
Publish all runs, including failures.

## Amendment (October 13, 2026)

This amendment supersedes the sections above wherever they conflict.
The original text stays as pre-registered on October 12.

### What changed

1. **The `baseline` arm is dropped.**
   The study now has 6 arms (`bc5cdc1`, `1d83776`, `27b9d9f`, `000e04d`, `8f00506`, `cc51630`) at the same 5 seeds (2000 to 2004), which is 30 runs.
2. **Comparison 1 (`bc5cdc1` against `baseline`) is removed.**
   Five comparisons remain: `1d83776` against `bc5cdc1` (claimed win), and each of the four reconstructed arms against `1d83776` (claimed not a win).
   Bonferroni adjustment uses m = 5 (intervals at level 1 - 0.05/5).
   The smallest possible exact sign-test p-value with five pairs is 1/32, and times 5 it is 0.16, so the sign test still cannot reach 0.05 after correction.
   The decision rules are unchanged, except that "claimed wins" is now comparison 1 alone (`1d83776` against `bc5cdc1`), and "claimed not-wins" are comparisons 2 to 5.
3. **The run order is a new fixed shuffle** with shuffle seed 20261013, recorded in the new plan.
4. **A memory guard is added** (below).
5. Sources, the seed transformation, per-job limits, the 12-hour study cap, the three-consecutive-failure stop, and the analysis rules are unchanged.
   The study runs fresh in a new output directory, `research/s1-part2/run-2026-10-13`.

### Why

The baseline configuration (Pro-tier defaults: depth 6, device batch 16, 64K-token batch, MLP ratio 4x) cannot run on this 24 GB machine without exceeding memory.
The published log itself reports 32.1 GB peak on a 24 GB machine.
The original protocol already said so in Procedure but did not act on it.
On October 12, macOS memory pressure reached red during run 2 of 35, which was the `baseline` arm at seed 2002.
The owner stopped the study (see [`run-2026-10-12/stopped-by-owner.json`](run-2026-10-12/stopped-by-owner.json)).
Running this configuration further risks the machine, so it is removed rather than retried.

### Timing of the decision

The decision was made before any result of the remaining comparisons was seen.
Only one run had completed, run 1 (`27b9d9f`, seed 2004).
That result has not been analysed and is **not used**.
The new study re-runs every arm and seed, including that one, in a new output directory.
The stopped output directory stays unchanged as evidence of the stop.
Run 2 was left in ledger state `running` with no result, and no interruption record was written by the runner.
The owner did not authorize resuming it.

### Claim not audited

The published improvement of `bc5cdc1` over the unmodified baseline (2.026063 to 1.439990 BPB) is **not audited**.
No statement about it follows from this study.
Reports must say so, and must not describe the audit as covering the full lineage.
`bc5cdc1` remains in the study as the parent of `1d83776`, so the second published win (1.439990 to 1.429396) is still tested.

### Memory guard

The runner samples system memory about every 5 seconds while each job runs.
Each sample reads `sysctl -n kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical) and the system-wide free percentage from `memory_pressure`.
The thresholds are named constants in `openloop.studies.claims_audit` and appear in the plan under `limits.memory_guard`, so they are pre-registered:

| Constant | Value | Meaning |
| --- | ---: | --- |
| `MEMORY_SAMPLE_SECONDS` | 5 | sampling interval during a job |
| `PRESSURE_CRITICAL` | 4 | a reading at this level trips the guard |
| `FREE_PERCENT_FLOOR` | 15 | free percentage below this counts as a low sample |
| `FREE_PERCENT_CONSECUTIVE_SAMPLES` | 2 | consecutive low samples that trip the guard |

The warning level (2) alone never trips the guard, because ordinary training runs at warning.
A single low-free sample never trips it either.

On a trip the runner:

1. cancels the current job through the executor, which kills its process group;
2. marks that run failed in the ledger with reason "memory guard" and writes its run receipt with the same failure;
3. writes `memory-guard-stop.json` with the time, the reason, the readings, the run index, arm and seed, and the thresholds;
4. writes the usual `interrupted.json`;
5. stops the whole study. No further run starts, and nothing is retried automatically.

A reader that cannot read the memory state also trips the guard, and the study will not start if the first reading fails.
The guard samples only while a job runs.
It cannot stop a spike shorter than the sampling interval, and pressure can reach critical between samples.
A guard stop is an environment event, not evidence about any hypothesis.
Restarting after a guard stop needs a new owner decision, a new output directory, and a documented amendment.

## Amendment 2 (October 13, 2026): seed completion and held-out phase

Plan hash: `738a84c5bf2b1b1b151221cac36aaf8c1c8ce0ade95f2635b068cc80696732d6` ([plan](plan-heldout-2026-10-13.json)).
This amendment adds two follow-ups.
It changes nothing in the October 13 study or its [report](run-2026-10-13/report.json).
Those files stay as evidence.

### Timing and honesty about what is confirmatory

This amendment is written **after** the [run-2026-10-13](run-2026-10-13/report.json) results were seen, and before any run it describes.
The author knows the results below.
They shape which follow-ups are worth running, so the following status applies.

- Run 30 (arm `8f00506`, seed 2004) failed after 0.01 s.
  The Python install under `/tmp` that the worker used was purged by macOS.
  This is an environment failure, not an experiment result.
- The published winner `1d83776` against `bc5cdc1` had mean gain +0.0092 BPB, 4 of 5 seeds better, and a Bonferroni-adjusted interval that includes zero (finding: not replicated).
- All four reconstructed discards were indistinguishable from `1d83776`.

**Slot A (seed completion) is confirmatory of the October 13 protocol.**
It re-executes one slot of the registered design.
It was chosen by the failure, not by any outcome.
Its program is byte-identical to the failed run's program (the plan checks the SHA-256 against run 30's receipt).
**Phase B (held-out) is a follow-up chosen after the validation result.**
Its hypothesis, shard, seeds, test and decision rule are fixed here before any held-out score exists.
The held-out shard has never been scored by any model, so its scores are new data.
But the choice to test this claim, and not the other four, comes from seeing that this claim was the only published win in the audit and the only one that did not replicate on validation.
Any statement about other comparisons from phase B is exploratory.
Phase B does not reopen the October 13 findings.

### A. Seed completion

Re-run exactly one slot: arm `8f00506`, seed 2004, with no change to the program, the seed transformation, the data cache, the limits or the memory guard.
The run is recorded as a re-execution of an environment-failed slot (`kind: seed_completion`).
Its receipt names the failed receipt and its SHA-256.
It is not a retry chosen by outcome: the slot is the one that failed for an environment reason, and it is run exactly once.
If it fails again for an environment reason, it is not run a third time without a new decision.

The analysis recomputes the October 13 report for all five comparisons from the October 13 receipts plus this one receipt.
It uses the unchanged `openloop.studies.claims_audit.analyze_audit`, the same Bonferroni adjustment (m = 5) and the same decision rules.
It writes a **new** file, `run-heldout-2026-10-13/report-seed-completion.json`.
The original `report.json` is never overwritten.
Only comparison `8f00506` against `1d83776` can change, because it gains its fifth pair.
The other four comparisons must reproduce their October 13 numbers exactly.

### B. Held-out phase for the winner claim only

**Claim tested:** `1d83776` improves on `bc5cdc1` (published 1.439990 to 1.429396 BPB).
No other claim is tested in phase B.

**Held-out shard.** `shard_05000.parquet` from the same Hugging Face dataset and URL pattern that `prepare.py` uses.
It is not one of the shards in the [recorded manifest](../artifacts/autoresearch-mlx/data_manifest.json) (shards 00000 to 00009 and 06542).
It is stored in `data/climbmix-heldout/`, outside `~/.cache/autoresearch/data`, because the upstream loader treats every parquet file in that directory as training data.
Its URL, size, SHA-256 and download time are in [`data/climbmix-heldout/manifest.json`](../../data/climbmix-heldout/manifest.json).
The runner checks the shard hash before the study and after every run.

**Runs.** Seeds 3000 to 3004, both arms at each seed (paired), which is 10 runs.
The seeds are disjoint from 42, from 2000 to 2004 and from the T1 calibration seeds.
Order is a fixed shuffle (shuffle seed 20261014), recorded in the plan.
The seed-completion run is job 1; the 10 held-out runs follow in shuffled order.
One GPU lane serves all jobs, with the same 1800-second wall limit, 23 GiB memory limit, 16 MiB log limit, three-consecutive-failure stop and memory guard as the audit.
The study cap is 4 hours (the expected run time is about 1.3 hours).
The 1800-second wall limit leaves room for the extra evaluation, which adds one pass over 5,242,880 tokens.

**Program.** Each held-out run executes the arm's own published program with the same seed injection as before, plus one evaluation-only addition.
The addition is inserted exactly once, directly after the published final evaluation line.
It runs after training has ended and after the published `val_bpb` exists.
It points the `prepare` module's `DATA_DIR` and `VAL_FILENAME` at the held-out shard, then calls the arm's own `evaluate_bpb` with the arm's own tokenizer and device batch size.
The token count is the same 5,242,880 as the published validation evaluation.
It prints `heldout_bpb`.
No training code path changes, and `prepare.py` is unchanged.
The runner fails the job if the anchor occurs other than exactly once or if held-out code is already present.
The plan stores the hash of the added code, and each receipt stores the hash of the seeded program before the addition and of the final program.
Both arms run the same addition.
The runner parses `val_bpb` (published validation shard 6542) and `heldout_bpb` from each run's output and requires exactly one `heldout_bpb` line.
The ledger records both scores.

**Validation scores in this phase are additional samples.**
They come from the shard the authors used for selection.
They are reported next to the held-out scores to show whether the two views agree.
They never decide the phase B finding.
**The held-out score is the confirmatory one.**

**Pre-registered test.**
For each seed with both arms successful, the gain is `bc5cdc1` held-out BPB minus `1d83776` held-out BPB.
Positive means `1d83776` is better.
This is a single pre-specified comparison, so no Bonferroni adjustment applies.
The report gives:

- the mean gain, the median gain and the sample standard deviation;
- the 95% two-sided t interval for the mean gain (level 0.95);
- the exact one-sided sign test p-value (`openloop.decider.sign_p_value`) and the paired t-test p-value, both unadjusted, as consistency checks;
- the per-seed gains and the number of seeds where `1d83776` wins;
- the same statistics for the validation scores of this phase, labelled as additional samples;
- validation-versus-held-out agreement: the number of seeds where the two gains have the same sign, the correlation between the two sets of per-seed gains, and the mean held-out minus validation shift for each arm.

**Classification rules (by the held-out interval alone):**

- **replicated**: the interval lies entirely above 0;
- **reversed**: the interval lies entirely below 0;
- **inconclusive**: the interval contains 0;
- **insufficient data**: fewer than 3 seeds have both arms successful.

A failed run (executor or memory-guard failure) is an environment or implementation event.
It removes that seed's pair from the analysis and is not evidence about the hypothesis.
Failed runs are not retried automatically.

### Limits of phase B

- Five seeds give a wide interval.
  In the October 13 data, the standard deviation of the paired gain was 0.0126 BPB.
  At that spread, the mean gain of +0.0092 gives an interval of about -0.007 to +0.025 with five pairs.
  If the true effect is near the observed one, "inconclusive" is the most likely outcome.
  A non-replication means this test did not detect the effect, not that the effect is absent.
- Seeds vary initialization only (see Procedure), and every seed sees the same training data and the same held-out tokens.
  The interval excludes variation from data order and from the choice of held-out documents.
- The held-out shard is one sample of 5,242,880 tokens from one shard.
- The two arms differ in more than one setting (batch size and step count; see the arm table), so a held-out result is about the arm pair, not about any single setting.
- `1d83776` is the audit's incumbent for four other comparisons.
  Phase B says nothing about them.
- The extra evaluation adds a little time to `total_seconds` and may raise the reported peak memory.
  It does not change `training_seconds`, `num_steps` or `val_bpb`, because it runs after all three are fixed.
  Whether it leaves `val_bpb` unchanged in practice is not verified before the run.
  The validation scores of this phase are checked against the October 13 spread for sanity.

## Amendment 3 (October 13, 2026): valid re-run of the Amendment 2 jobs

Plan hash: `e3fcb2b05b2b8fb5d0ac48b61c0389be5306514d12ede4d66114c042032f2db3` ([plan](plan-heldout-2026-10-13b.json)).
This amendment is written before any run it describes.
It changes nothing in the October 13 study, in Amendment 2's plan, or in the files of the invalidated Amendment 2 run.
Those stay as evidence.

### What happened

The Amendment 2 run ([run-heldout-2026-10-13](run-heldout-2026-10-13/stopped-invalid-sleep.json)) was stopped by the coordinator on the owner's instruction after 4 of 11 jobs.
The owner closed the laptop lid.
macOS slept and ran throttled dark-wake periods.
The published 300 s training budget is measured with wall-clock time (`time.time` per step in `train_mlx.py`), and wall-clock time kept advancing.
Runs therefore ended after 359 to 543 steps, where the October 13 runs took about 844 steps (`1d83776`, `8f00506`).
`caffeinate -i` does not prevent lid-close sleep.
The stop record lists the evidence for runs 1 to 3.
Run 4 overlapped the sleep window.

### Invalidation

All 4 receipts in `run-heldout-2026-10-13` are **invalid by an environment event** and are **not used** in any analysis.
Nothing in that directory changes.

Honest disclosure: two held-out scores from the invalid runs were seen (`1d83776`, seeds 3000 and 3001: 1.506821 and 1.66406) before this amendment.
They come from runs that trained for 506 and 359 steps, so they say nothing about the hypothesis.
They are excluded by the rule below, which uses only throughput and clock evidence.
The same slots are re-run: the claim, the shard, the seeds (3000 to 3004 for both arms, plus slot A), the order (shuffle seed 20261014), the programs, the analysis and the classification rules are exactly those of Amendment 2.
Re-running the same slots is justified because the invalidation is environment-based and was not chosen by any outcome.
Seeing two invalid scores cannot be fully undone, so a reader should weigh the held-out result with that in mind.

### New rules (validity, not outcome)

An **attempt** is one execution of a slot.
An attempt is **invalid** when either rule holds.
The thresholds are named constants in `openloop.studies.validity` and appear in the plan under `validity`.

| Constant | Value | Meaning |
| --- | ---: | --- |
| `DRIFT_SAMPLE_SECONDS` | 5 | sampling interval during a job |
| `DRIFT_LIMIT_SECONDS` | 5 | cumulative drift above this invalidates the attempt |
| `DRIFT_QUIET_TOLERANCE_SECONDS` | 1 | a sample with more drift than this is not quiet |
| `SETTLE_SECONDS` | 60 | quiet time required before a re-execution starts |
| `STEP_FRACTION` | 0.85 | minimum `num_steps` as a fraction of the arm's reference |
| `MAX_ATTEMPTS_PER_SLOT` | 3 | attempts allowed for one slot |

1. **Sleep guard.**
   While a job runs, the runner samples `time.time()` and `time.monotonic()` about every 5 seconds.
   On macOS the monotonic clock does not advance during sleep.
   Drift is the wall-clock elapsed time minus the monotonic elapsed time, accumulated over positive increments only.
   When cumulative drift exceeds 5 s, the runner cancels the job through the executor (which kills its process group), marks the ledger run failed with reason `invalid: machine sleep (drift X s)`, and writes a receipt with `validity.valid: false`.
   A final sample after the job ends catches sleep in the last interval.
2. **Step-count check.**
   After a job succeeds, it is invalid if `num_steps` is below 0.85 times its arm's reference.
   The reference is the median `num_steps` of that arm's [run-2026-10-13](run-2026-10-13/report.json) receipts, frozen in the plan: `1d83776` 844 (floor 717.4), `8f00506` 844 (floor 717.4), `bc5cdc1` 445 (floor 378.25).
   The reason is that a wall-clock budget lets throttling shorten training silently.
   This rule uses throughput only, never BPB.
   The ledger records the run as failed with reason `invalid: throttled`.
   The invalid summary stays in the receipt as `observed_summary`, and `summary` is null.

The memory guard is unchanged and stays active.
A memory-guard trip still stops the study.

### Re-execution

An invalid attempt is re-executed for the **same slot** (same arm, same seed, same program hash), up to 3 attempts per slot.
Before each re-execution the runner waits until the machine shows no drift above the tolerance for 60 seconds.
Each attempt has its own ledger run, work directory and receipt (`run-NN-<kind>-<arm>-s<seed>-aN.json`).
Invalid attempts stay as evidence.
If a slot has 3 invalid attempts, the study stops with `slot-exhausted` in `interrupted.json`, and nothing continues without a new owner decision.
An ordinary training failure (nonzero exit, parse failure, limit) is **not** retried: it is recorded and counts toward the three-consecutive-failure stop, as before.
Invalid attempts do not count as failures there.
Re-execution is justified because the rule depends on the environment and on throughput only, so it cannot select favorable scores.
The study cap rises to 8 hours to leave room for re-executions.

### Analysis

Analysis uses **only valid attempts** (`validity.valid: true`).
It writes `report-seed-completion.json` and `heldout-report.json` with the Amendment 2 code, and an `attempts-report.json` that lists every attempt with its validity and the number excluded.
Classification rules, the single held-out test and the limits of phase B are as in Amendment 2.

### Limits of the guard

The guard samples about every 5 seconds, so it cannot see a stall shorter than the drift limit.
It detects sleep by clock drift, not by power events.
A throttling event without sleep is caught only by the step-count check.
Whether macOS pauses the runner process during sleep, and so delays the next sample, is not verified.
The final sample after completion covers that case.

## Amendment 4 (October 13, 2026): resume after a memory-guard stop

Plan hash: `4c6548cc1ad80e00b00452d7278078515489f0c9212c01f6a55a8f75610f551f` ([plan](plan-heldout-2026-10-13c.json)).
This amendment is written before any run it describes.
It changes nothing in the October 13 study, in Amendments 2 and 3, or in the files of the `run-heldout-2026-10-13` and `run-heldout-2026-10-13b` directories.
Those stay as evidence.

### What happened

The Amendment 3 run ([run-heldout-2026-10-13b](run-heldout-2026-10-13b/memory-guard-stop.json)) stopped on the memory guard during slot 5 of 11 (`bc5cdc1`, seed 3002).
The last readings were free memory 15%, 14% and 14% at pressure level 2 (warning).
The floor rule was free memory below 15% for 2 consecutive samples.
Other desktop applications (about 8 GB) were running next to training.
The guard cancelled slot 5, which has no result.
Slots 1 to 4 had completed and are valid under the Amendment 3 rules (drift near 0, steps at least 0.85 times the arm's reference).

The stop was an environment and safety event.
It was not chosen by any outcome, and the guard never saw a score.

### Memory-guard change

The free-memory floor changes from 15% to **5%**, still for 2 consecutive samples.
The owner asked for a less restrictive guard before this amendment ran; the floor was set to 5% (first drafted as 10%) before any slot of this plan executed.
Critical pressure (level 4) still stops the study immediately.
Warning pressure alone still does not stop it.
Rationale: the design says warning alone must not stop a run, but on this machine warning pressure coincides with 14% to 15% free memory.
The 15% floor therefore contradicted that design.
Critical pressure (red, the level at which the owner stopped the first launch) is the main protection; the 5% floor is only a last backstop.
The guard code is `openloop.studies.resume.FloorMemoryGuard`; the thresholds are in the plan under `limits.memory_guard`.
A guard stop is still never retried automatically.

### Retained slots and remaining slots

The 4 valid receipts of `run-heldout-2026-10-13b` are **retained**:

| Slot | Kind | Arm | Seed |
| ---: | --- | --- | ---: |
| 1 | `seed_completion` | `8f00506` | 2004 |
| 2 | `heldout` | `1d83776` | 3000 |
| 3 | `heldout` | `1d83776` | 3001 |
| 4 | `heldout` | `bc5cdc1` | 3000 |

They satisfy the Amendment 3 validity rules, which were fixed before they ran.
The plan freezes the SHA-256 of each retained receipt.
The remaining 7 slots (5 to 11, in the original order, same seeds, same programs) run in a **new** directory, `run-heldout-2026-10-13c`.
A retained slot is never run again.
The cancelled slot 5 attempt is not a result and is not retained; it is listed in the plan under `resume.not_retained`.
Slot 5 runs as attempt 1 in the new directory.
All other rules (shard, seeds, order, programs, validity, re-execution of invalid attempts up to 3 per slot, 8-hour cap) are those of Amendment 3.

### Resume checks

The runner refuses to start unless all of these hold:

- the Amendment 3 plan reproduces from the current sources and equals the stored plan, and the prior run's `plan.json` equals it;
- each retained receipt has `validity.valid: true`, a successful summary and no failure;
- each retained receipt matches its frozen slot (kind, arm, seed), both program hashes (seeded and final) and the arm's step floor, and for held-out slots carries a held-out score;
- each retained receipt's artifacts match their recorded SHA-256, and its summary equals the summary re-parsed from its stdout;
- the held-out shard hash equals the one frozen in the plan (it is also checked after every run);
- no slot has two valid receipts, and the output directory is new.

Any mismatch refuses the resume.

### Disclosure

One held-out pair was observed by the coordinator before this amendment: seed 3000 (`1d83776` 1.407197, `bc5cdc1` 1.425002 held-out BPB), from retained slots 2 and 4.
This amendment does not change any rule, so the observation cannot steer a choice.
The only rule that changes is the memory floor, and it was chosen from the stop record, not from any score.
A reader should still weigh the result with that observation in mind.

### Analysis

Analysis is unchanged: the same code, the same Bonferroni adjustment, the single pre-registered held-out test and the Amendment 2 classification rules.
It is computed over the **union of valid attempts** from both directories.
The new directory receives `report-seed-completion.json`, `heldout-report.json` and `attempts-report.json`.
The attempts report names the source directory of every receipt, the SHA-256 of each retained receipt, and the prior attempts that were not retained.
The ledger of the new directory holds only the runs executed there; the retained runs stay in the ledger of `run-heldout-2026-10-13b`.

## Amendment 5 (October 14, 2026 UTC): complete the held-out design after a GPU fault

Plan hash: `76a89aa76e7554d61b1b02c129ecb1289dbebbea8ba9982ee164f9f368136c51` ([plan](plan-heldout-2026-10-13d.json)).
This amendment is written before any run it describes.
It changes nothing in the October 13 study, in Amendments 2 to 4, or in the files of the `run-heldout-2026-10-13`, `run-heldout-2026-10-13b` and `run-heldout-2026-10-13c` directories.
Those stay as evidence.

### What happened

The Amendment 4 run ([run-heldout-2026-10-13c](run-heldout-2026-10-13c/finished.json)) finished all 7 slots it started, but slot 10 (held-out, arm `1d83776`, seed 3002) has no result.
The training process exited with signal 6 (exit code -6) after 160.6 s, and its stderr reports `kIOGPUCommandBufferCallbackErrorInnocentVictim`, a Metal GPU fault ([stderr](run-heldout-2026-10-13c/executor/b8dc7f5909e744b9ba871bc55bfad8a8/stderr)).
The receipt is [run-10-heldout-1d83776-s3002-a1.json](run-heldout-2026-10-13c/run-10-heldout-1d83776-s3002-a1.json).
The attempt was valid by the drift and step-count rules (drift 0.002 s; it did not run long enough to produce a step count), and no score was read.
This is an environment failure, not an outcome of the experiment.
By the Amendment 3 rule, an ordinary failure is not retried automatically, so the run ended with that pair unmatched.

### Why the design is incomplete

Amendment 2 specified 5 seeds (3000 to 3004) for each arm and a single held-out test over the paired gains.
Amendment 2's rule that fewer than 3 matched pairs is "insufficient" never applied, because 4 pairs existed.
The 4-pair result is nonetheless a deviation from the pre-registered 5-pair design, caused only by the fault.
The Amendment 4 result, read from [heldout-report.json](run-heldout-2026-10-13c/heldout-report.json): seeds 3000, 3001, 3003 and 3004, mean held-out gain +0.0194, 95% t interval [+0.0149, +0.0239], classified `replicated`.

### Disclosure

The coordinator had already seen the 4-pair held-out result when this amendment was written.
The unmatched parent run (`bc5cdc1`, seed 3002, validation BPB 1.40843, held-out BPB 1.413666) is the best score that arm reached in the phase.
It has no matched child run.
Completing the pair is therefore expected to reduce the estimated gain, not to inflate it: if `1d83776` at seed 3002 scores like its other seeds (validation BPB 1.405 to 1.410), the seed-3002 gain would be smaller than the 4-pair mean.
This is an expectation, not a guarantee; the result may go in either direction.
The re-run is requested by the owner, to obtain valid results for the whole design.
A reader should weigh that the 4-pair result had been seen before this decision.

### Decision and scope

The coordinator re-executes **only slot 10** (held-out, arm `1d83776`, seed 3002), as attempt 1 in a new directory, `run-heldout-2026-10-13d`.
The program (base and seeded hashes), the shard, the hyper-parameters, the guards and the analysis are identical to Amendment 4.
Nothing about slot 10 is changed to make it more likely to succeed or to score differently.
All 10 other slots are **retained** from the earlier directories and never run again:

- slots 1 to 4 from `run-heldout-2026-10-13b` (the Amendment 3 run);
- slots 5 to 9 and 11 from `run-heldout-2026-10-13c` (the Amendment 4 run).

The plan freezes the SHA-256 of each retained receipt.
The failed slot 5 attempt of `run-heldout-2026-10-13b` (memory guard) and the failed slot 10 attempt of `run-heldout-2026-10-13c` (GPU fault) are not retained and appear in the plan under `resume.not_retained`.

The runner is [`research/audit_heldout_complete.py`](../audit_heldout_complete.py).
It imports `audit_heldout_resume.py` and every older module unchanged, so the older plans still reproduce.
It adds multi-directory retention and the combined reports.
Guards, validity rules and limits are those of Amendment 4:

- memory guard: stop at critical pressure (level 4) or at free memory below 5% for 2 consecutive samples;
- sleep guard and step-count rule of Amendment 3, with re-execution of an attempt that is invalid for those reasons, up to 3 attempts for the slot, after 60 s of quiet;
- an ordinary failure is not retried;
- the 8-hour cap and the 1,800 s per-job limit.

The ledger of the new directory holds only the run executed there.

### Resume checks

The runner refuses to start unless all of these hold:

- the Amendment 3 and Amendment 4 plans reproduce from the current sources and equal the stored plans, and the latest run's `plan.json` equals the Amendment 4 plan;
- each retained receipt has `validity.valid: true`, a successful summary and no failure;
- each retained receipt matches its frozen slot (kind, arm, seed), both program hashes and the arm's step floor, and carries a held-out score;
- each retained receipt's artifacts match their recorded SHA-256, and its summary equals the summary re-parsed from its stdout;
- no slot has valid receipts in two directories;
- the held-out shard hash equals the one frozen in the plan;
- the output directory is new.

### If slot 10 fails again

If slot 10 fails again for an environment reason (a GPU fault, a memory-guard stop, or any other ordinary failure), it is **not retried further**.
The 4-pair result is then reported with this caveat: the pair of seed 3002 is missing for an environment reason, and the unmatched parent run is that arm's best score.
Invalid attempts under the Amendment 3 rules still re-execute up to the limit above, and a slot with 3 invalid attempts stops the study with `slot-exhausted`.

### Analysis and reporting

The 5-pair analysis is the pre-registered one: Amendment 2 specified 5 seeds, the same code, the single held-out test and the classification rules.
It is the **primary finding**.
The new directory receives:

- `report-seed-completion.json`: the same inputs as before, recomputed;
- `heldout-report.json`: the primary 5-pair analysis over all valid attempts of the three directories;
- `heldout-report-4pair.json`: the same analysis over the valid attempts retained from the earlier directories only, which reproduces the Amendment 4 four-pair numbers;
- `attempts-report.json`: every attempt in `run-heldout-2026-10-13b`, `-13c` and `-13d`, with its validity, failure reason, receipt SHA-256 and whether it was used in the analysis.

The final writeup reports the 4-pair and the 5-pair numbers side by side, with the 5-pair result as the primary finding.
If slot 10 fails again, the primary analysis has 4 pairs and equals the Amendment 4 result, with the caveat above.
