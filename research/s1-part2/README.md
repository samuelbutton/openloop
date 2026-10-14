# S1 part 2: public-claims audit, M4 Pro lineage

Prepared October 12, 2026; amended October 13 and 14, 2026 (five times). Need: [N2](../../NEEDS.md#n1n10). Status: the October 12 launch was stopped by the owner; the October 13 study finished 29 of 30 runs (run 30 failed for an environment reason); the Amendment 3 run kept 4 valid slots and stopped on the memory guard in slot 5; the Amendment 4 resume finished 6 of 7 slots (slot 10 crashed with a Metal GPU fault, so one held-out pair is unmatched); the Amendment 5 re-run of that one slot is prepared, not run.

**Final result (October 14, 2026).** Amendment 5 completed the pre-registered design. Held-out test of the published winner `1d83776` vs its parent `bc5cdc1`, 5 paired seeds (3000–3004): mean held-out gain **+0.01583 BPB**, 95% t interval [+0.00547, +0.02619], 5/5 seeds better, exact sign p = 0.03125 — **replicated** under the Amendment 2 rule. The earlier 4-pair estimate (+0.01940) was inflated by the missing seed-3002 pair (its gain was +0.00156). All four reconstructed "tied/slightly worse" discards are indistinguishable from the winner. See [heldout-report.json](run-heldout-2026-10-13d/heldout-report.json), [heldout-report-4pair.json](run-heldout-2026-10-13d/heldout-report-4pair.json), [report-seed-completion.json](run-heldout-2026-10-13d/report-seed-completion.json) and [attempts-report.json](run-heldout-2026-10-13d/attempts-report.json). Replication on one 20-core M4 Pro (published: 16-core); limited precision.
Read the [pre-registered protocol](PROTOCOL.md) first.
The current [frozen plan](plan-2026-10-13.json) has hash `5f0badf3342f156eed23b965cb2603da59c7ae6d02a2ec822d242f81a91c9e43`.
It supersedes [plan-2026-10-12.json](plan-2026-10-12.json) (hash `df0ff4818c75396ad176a72460a1a1c57a436c544fcbf92de7b61ddbd402357b`), which can no longer be reproduced because the runner changed.
See the protocol's [amendment](PROTOCOL.md#amendment-october-13-2026).

The audit re-runs six code versions from the elementalcollision/autoresearch M4 Pro lineage at five matched seeds each.
That is 30 local GPU runs of 300 seconds of training each.
The expected time is about 3.2 hours, with a 12-hour hard cap.
The `baseline` arm was dropped because it cannot run on 24 GB; the published baseline claim (2.026 to 1.440) is not audited.

## Status of the October 12 launch

`run-2026-10-12/` is evidence of a stopped study.
Run 1 (`27b9d9f`, seed 2004) completed; run 2 (`baseline`, seed 2002) pushed macOS memory pressure to red and the owner stopped it (`stopped-by-owner.json`).
Run 2 is left in ledger state `running` with no result.
That output is not used for analysis and must not be resumed, re-run into, or edited.


## Prepare (already done; no training)

The October 13 plan was written by the command below.
The October 12 plan was prepared the same way with the earlier runner.

```sh
git clone --filter=blob:none --no-checkout https://github.com/elementalcollision/autoresearch.git research/s1-part2/sources/fork
git -C research/s1-part2/sources/fork checkout 1d8377683d9a5b1af14f5933e6f5130c54e94460 -- .
uv run python research/audit_m4_pro.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-2026-10-13.json
```

The clone is git-ignored.
Preparation verifies the commits and their history, hashes every file, builds the four reconstructed programs, compiles all 30 seed-injected programs, verifies the ClimbMix cache, and records the worker packages.
It imports no MLX.
It refuses to overwrite an existing plan.

## Launch (requires owner approval for local GPU use)

Plug in AC power, close other GPU-heavy work, and run:

```sh
caffeinate -i uv run python research/audit_m4_pro.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-2026-10-13.json \
  --output research/s1-part2/run-2026-10-13 \
  --execute-approved
```

The command recomputes the plan and refuses to run if its hash, sources, cache, or packages changed.
The output directory must not exist.
It receives `plan.json`, `ledger.sqlite3`, one `run-NN-<arm>-s<seed>.json` receipt per run, `executor/` logs, `workdirs/` (git-ignored program copies), and `started.json`.
A clean finish writes `report.json` and `finished.json`.
An interruption writes `interrupted.json`.

The runner also guards memory.
During every job it samples `kern.memorystatus_vm_pressure_level` and the system-wide free percentage from `memory_pressure` every 5 seconds.
It stops the whole study when pressure is critical (level 4), or when free memory is below 15% for two consecutive samples.
Warning (yellow) alone does not stop it.
On a stop it kills the job's process group, marks the run failed with reason "memory guard", and writes `memory-guard-stop.json` and `interrupted.json`.
It never retries.
Do not launch again after a guard stop without a new owner decision.
The thresholds are in the plan under `limits.memory_guard`.

To analyze an interrupted output that has no `report.json` yet:

```sh
uv run python research/audit_m4_pro.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --summarize research/s1-part2/run-2026-10-13
```

Do not re-run a completed output directory; start a new one.
Share the report with the fork author before publishing it.

## Amendment 2 follow-ups (prepared, not run)

Read the protocol's [Amendment 2](PROTOCOL.md#amendment-2-october-13-2026-seed-completion-and-held-out-phase) first.
The audit finished 29 of 30 runs.
Run 30 (`8f00506`, seed 2004) failed in 0.01 s because macOS had purged the old Python install.
The follow-ups are one plan with 11 jobs: the re-execution of that slot, then the held-out phase for `1d83776` against `bc5cdc1`.
The frozen [plan](plan-heldout-2026-10-13.json) has hash `738a84c5bf2b1b1b151221cac36aaf8c1c8ce0ade95f2635b068cc80696732d6`.
It was written by `research/audit_heldout.py`, which imports `research/audit_m4_pro.py` and leaves it unchanged, so the October 13 plan still reproduces.

The held-out shard (`shard_05000.parquet`, 91,420,038 bytes) is downloaded and hashed.
It lives in git-ignored `data/climbmix-heldout/`; the tracked [manifest](../../data/climbmix-heldout/manifest.json) holds its URL, size and SHA-256.
Never move it into `~/.cache/autoresearch/data`: the upstream loader would train on it.

Prepare (already done; no training; refuses to overwrite):

```sh
uv run python research/audit_heldout.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-heldout-2026-10-13.json
```

Launch (requires owner approval for local GPU use; the output directory must not exist):

```sh
caffeinate -i uv run python research/audit_heldout.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-heldout-2026-10-13.json \
  --output research/s1-part2/run-heldout-2026-10-13 \
  --execute-approved
```

Expected time is about 1.3 hours (11 jobs at about 420 s), with a 4-hour cap.
The runner recomputes the plan and refuses to run if the hash, sources, cache, runtime packages or shard changed.
It refuses unless the worker packages and hardware equal those in the October 13 plan.
It uses the same memory guard as the audit.
Receipts are `run-NN-<kind>-<arm>-s<seed>.json`; `kind` is `seed_completion` or `heldout`.
A clean finish writes two new analysis files and `finished.json`:

- `report-seed-completion.json`: the five audit comparisons from the October 13 receipts plus the re-executed run.
  The original `run-2026-10-13/report.json` is never touched.
- `heldout-report.json`: the pre-registered held-out test for `1d83776` against `bc5cdc1`, with validation scores as additional samples.

To write both for an interrupted output (they refuse to overwrite):

```sh
uv run python research/audit_heldout.py --report research/s1-part2/run-heldout-2026-10-13
```

Do not re-run a finished output directory.
A guard stop or a failed slot needs a new owner decision and a documented amendment.

## Amendment 3 re-run (prepared, not run)

The Amendment 2 run was invalidated: the owner closed the lid, macOS slept, and the wall-clock training budget produced runs of 359 to 543 steps instead of about 844.
See [stopped-invalid-sleep.json](run-heldout-2026-10-13/stopped-invalid-sleep.json) and the protocol's [Amendment 3](PROTOCOL.md#amendment-3-october-13-2026-valid-re-run-of-the-amendment-2-jobs).
All 4 receipts in `run-heldout-2026-10-13` are invalid and unused.
The same 11 slots are re-run with a sleep guard, a step-count validity check and automatic re-execution of invalid attempts (at most 3 per slot).
The frozen [plan](plan-heldout-2026-10-13b.json) has hash `e3fcb2b05b2b8fb5d0ac48b61c0389be5306514d12ede4d66114c042032f2db3`.
It was written by `research/audit_heldout_valid.py`, which imports `research/audit_heldout.py` unchanged.

**Keep the lid open and keep the Mac on AC power for the whole run.**
`caffeinate -dims` prevents display, idle and disk sleep, and system sleep on AC.
Closing the lid still sleeps the Mac.
The guard detects that and re-runs the affected slot, but every sleep costs time, and three invalid attempts in one slot stop the study.

Prepare (already done; no training; refuses to overwrite):

```sh
uv run python research/audit_heldout_valid.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-heldout-2026-10-13b.json
```

Launch (requires owner approval for local GPU use; the output directory must not exist):

```sh
caffeinate -dims uv run python research/audit_heldout_valid.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-heldout-2026-10-13b.json \
  --output research/s1-part2/run-heldout-2026-10-13b \
  --execute-approved
```

Expected time is about 1.3 hours without re-executions, with an 8-hour cap.
Receipts are `run-NN-<kind>-<arm>-s<seed>-aN.json`; each has a `validity` object.
Analysis uses only receipts with `validity.valid: true`.
To write the reports for an interrupted output:

```sh
uv run python research/audit_heldout_valid.py --report research/s1-part2/run-heldout-2026-10-13b
```

Do not re-run a finished output directory.
A memory-guard stop, an ordinary failed run, or a slot with 3 invalid attempts needs a new owner decision and a documented amendment.

## Amendment 4 resume (finished; slot 10 failed)

The Amendment 3 run (`run-heldout-2026-10-13b`) stopped on the memory guard in slot 5 of 11.
Slots 1 to 4 are valid and retained; slots 5 to 11 run in a new directory with the free-memory floor lowered from 15% to 5%.
See [memory-guard-stop.json](run-heldout-2026-10-13b/memory-guard-stop.json) and the protocol's [Amendment 4](PROTOCOL.md#amendment-4-october-13-2026-resume-after-a-memory-guard-stop).
The frozen [plan](plan-heldout-2026-10-13c.json) has hash `4c6548cc1ad80e00b00452d7278078515489f0c9212c01f6a55a8f75610f551f`.
It was written by `research/audit_heldout_resume.py`, which imports `research/audit_heldout_valid.py` unchanged, so the Amendment 3 plan still reproduces.
The plan freezes the SHA-256 of each retained receipt.
Do not edit, move or re-run anything in `run-heldout-2026-10-13b`.

Prepare (already done; no training; refuses to overwrite; refuses if the prior run does not match):

```sh
uv run python research/audit_heldout_resume.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --resume-from research/s1-part2/run-heldout-2026-10-13b \
  --plan research/s1-part2/plan-heldout-2026-10-13c.json
```

Launch (requires owner approval for local GPU use; the output directory must not exist):

```sh
caffeinate -dims uv run python research/audit_heldout_resume.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --resume-from research/s1-part2/run-heldout-2026-10-13b \
  --plan research/s1-part2/plan-heldout-2026-10-13c.json \
  --output research/s1-part2/run-heldout-2026-10-13c \
  --execute-approved
```

Expected time is about 50 minutes without re-executions (7 slots), with an 8-hour cap.
Receipts are `run-NN-<kind>-<arm>-s<seed>-aN.json` in the new directory.
A clean finish writes `report-seed-completion.json`, `heldout-report.json`, `attempts-report.json` and `finished.json` in the new directory.
They combine the retained and the new valid attempts, and every attempt names its source directory.
To write them for an interrupted output:

```sh
uv run python research/audit_heldout_resume.py --report research/s1-part2/run-heldout-2026-10-13c
```

### Pre-run checklist

- [ ] Mac on AC power.
- [ ] Lid open for the whole run (`caffeinate -dims` does not prevent lid-close sleep).
- [ ] Quit Chrome, Codex, ChatGPT, Obsidian and other heavy applications; quit Docker.
- [ ] Optional: exclude `data/` and `research/` from Spotlight indexing (System Settings, Spotlight, Search Privacy), so indexing does not compete for memory and disk.
- [ ] The output directory `run-heldout-2026-10-13c` does not exist.

The memory guard now stops the study at critical pressure, or at free memory below 5% for 2 consecutive samples.
A guard stop, an ordinary failed run, or a slot with 3 invalid attempts needs a new owner decision and a documented amendment.

## Amendment 5 completion (prepared, not run)

Slot 10 of the Amendment 4 run (held-out, arm `1d83776`, seed 3002) crashed about 160 s into training with a Metal GPU fault (`kIOGPUCommandBufferCallbackErrorInnocentVictim`, exit -6; see [stderr](run-heldout-2026-10-13c/executor/b8dc7f5909e744b9ba871bc55bfad8a8/stderr)).
The held-out test therefore used 4 of the 5 pre-registered pairs.
The unmatched `bc5cdc1` seed-3002 run is that arm's best score in the phase, so the 4-pair gain is probably biased upward.
Amendment 5 re-executes only that slot under the identical program, guards and rules; see the protocol's [Amendment 5](PROTOCOL.md#amendment-5-october-14-2026-utc-complete-the-held-out-design-after-a-gpu-fault).
The frozen [plan](plan-heldout-2026-10-13d.json) has hash `76a89aa76e7554d61b1b02c129ecb1289dbebbea8ba9982ee164f9f368136c51`.
It was written by `research/audit_heldout_complete.py`, which imports `research/audit_heldout_resume.py` and the older modules unchanged, so earlier plans still reproduce.
It retains the 10 valid receipts of `run-heldout-2026-10-13b` (slots 1 to 4) and `run-heldout-2026-10-13c` (slots 5 to 9 and 11), re-verifies their SHA-256 values, and runs only slot 10.
Do not edit, move or re-run anything in `run-heldout-2026-10-13b` or `run-heldout-2026-10-13c`.

Prepare (already done; no training; refuses to overwrite; refuses if an earlier run does not match):

```sh
uv run python research/audit_heldout_complete.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-heldout-2026-10-13d.json
```

Launch (requires owner approval for local GPU use; the output directory must not exist):

```sh
caffeinate -dims uv run python research/audit_heldout_complete.py \
  --source research/s1-part2/sources/fork \
  --worker-python research/autoresearch-mlx/.venv/bin/python \
  --plan research/s1-part2/plan-heldout-2026-10-13d.json \
  --output research/s1-part2/run-heldout-2026-10-13d \
  --execute-approved
```

Expected time is about 6 minutes for the one slot, with an 8-hour cap.
The new directory receives `report-seed-completion.json`, `heldout-report.json` (the primary 5-pair analysis), `heldout-report-4pair.json` (the Amendment 4 four-pair numbers), `attempts-report.json` (every attempt in the 13b, 13c and 13d directories), `finished.json` and the receipt `run-10-heldout-1d83776-s3002-a1.json`.
To write the reports for an interrupted output:

```sh
uv run python research/audit_heldout_complete.py --report research/s1-part2/run-heldout-2026-10-13d
```

If slot 10 fails again for an environment reason, it is not retried; the 4-pair result is reported with the caveat in Amendment 5.
The final writeup reports both the 4-pair and the 5-pair numbers, with the 5-pair result as the primary finding.

### Pre-run checklist

- [ ] Mac on AC power.
- [ ] Lid open for the whole run (`caffeinate -dims` does not prevent lid-close sleep).
- [ ] Quit Chrome, Codex, ChatGPT, Obsidian, Docker and other heavy applications.
- [ ] The output directory `run-heldout-2026-10-13d` does not exist.
