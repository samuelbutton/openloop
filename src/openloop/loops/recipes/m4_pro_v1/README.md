# Shared T1 starting recipe: m4-pro-v1

Frozen on October 12, 2026. The [freeze record](freeze.json) fixes the source,
default settings, dependency lock, evaluator, tokenizer, data snapshot, and two
token fidelities. It also records compile-only program hashes at seed 42 and
tokenizer path `/frozen/tokenizer`. Run-specific paths change the program hash.

`TrainingSource.load(repository)` selects this recipe by default. The repository
must match the existing trevin-creator pin. Its `prepare.py` and `uv.lock` remain
the trusted evaluator and runtime lock. Use `shared_recipe=False` to load the
legacy AdamW recipe. This preserves its source and program hashes. Adapter hashes
can change with openloop code, so this option does not recreate an old input hash.

## Selection evidence

The source is the published M4 Pro winner from
[elementalcollision/autoresearch at commit 1d83776][winner]. The
[published experiment log][wiki] reports 1.429396 val_bpb. The local
[unmodified reproduction][reproduction] measured 1.410165 on October 10, 2026.
The published gains combine a narrower MLP, a smaller batch, and a longer cooldown.
The optimizer is Muon for transformer matrices and AdamW for embeddings and scalars.
The local reproduction checks that this complete recipe runs on our M4 Pro.
It does not isolate the contribution of each change.

This is the best source-backed M4 Pro recipe in the checked experiment log.
It is not a claim of a universal best recipe. The
[trevin-creator results][trevin] report different winners on Max-class machines
and incomplete transfer between machines. Their lower scores do not establish
that those recipes beat this one under T1's dataset and token budgets.
The S1 parameter panel is calibration evidence, not a published winner selection.

| Setting | Legacy T1 | Shared recipe |
| --- | ---: | ---: |
| Depth | 4 | 6 |
| Width at default depth | 256 | 384 |
| MLP width ratio | 4 | 1.5 |
| Optimizer for transformer matrices | AdamW | Muon |
| Tokens per update | 65,536 | 8,192 |
| Device batch | 16 | 4 |
| Cooldown fraction | 0.5 | 0.7 |
| Updates at 8m | 128 | 1,024 |
| Updates at 21m | 320 | 2,560 |

Learning rates, Adam betas, weight decay, and the attention-window pattern retain
the published defaults. The MLP ratio, logit cap, optimizer implementation, and
other model arithmetic are frozen source. The declared `TrainingSetting` surface
remains the candidate interface.

## Port and freeze

The two `.py.txt` files are exact upstream source snapshots. SHA-256 checks run
before composition. The upstream README states an MIT license; see
[LICENSE](LICENSE) and the repository's [NOTICE](../../../../../NOTICE).
Neither the pinned elementalcollision revision nor the karpathy/autoresearch
head ships a license file, so [LICENSE](LICENSE) is the standard MIT text with the
copyright lines for Andrej Karpathy and Trevin Peterson. It is not a verbatim
upstream file and does not name the elementalcollision contributors; NOTICE records
this open question. Credit belongs to Andrej Karpathy, Trevin Peterson, and the
elementalcollision contributors for the upstream models and optimizer port. The
openloop adapter uses the repository's Apache-2.0 license.

Provenance (upstream URL, revision, license) lives here and in NOTICE, not in a
header comment. A header would change the pinned bytes and their hashes.

The pure [composer](../../recipe.py) inlines the optimizer and fixes depth 6.
It removes hardware detection and estimated MFU output. It connects the published
data-loader call to T1's frozen loader interface. The model and optimizer arithmetic
remain unchanged. The composed training source has a separate enforced SHA-256 pin.
The composer counts every rewrite and raises `ContractError` unless each matches
exactly as often as `EXPECTED_REWRITES` declares, so a missing or surplus pattern
fails loudly instead of leaving upstream code unmodified.
A change to these bytes requires a new recipe version and freeze record.

The existing token adapter supplies the requested seed, token-based learning-rate
and weight-decay progress, and exact token stop. Muon momentum retains the
published step ramp from 0.85 to 0.95 over 300 updates. Every update counts toward
both work and training time. Final scoring uses T1's trusted `prepare.py`, not
the published script's evaluator or summary. T1 retains 1,572,864 validation
tokens, sequence length 2048, and its memory-bounded evaluation microbatches.
Held-out membership is not opened.

The new training bytes change candidate and experiment identities. Existing
ledger cache entries cannot substitute for this recipe. Existing data manifests,
ledgers, reproduction evidence, and legacy golden hashes remain intact.

## Validation and limits

The [tests](../../../../../tests/loops/test_recipe.py) check source integrity,
model AST equality, both token stops, startup timing, schedules, and distinct
identities. CPU optimizer checks cover tall, wide, and square matrices, zero
gradients, finite updates, and the independent first-step AdamW formula.
They use float32 NumPy arithmetic and do not certify native MLX bfloat16 behavior.
Both full programs compile. All 12 installed package versions match the existing
Python 3.12 / MLX 0.31.0 lock.

## Native validation

One native smoke run of this port completed on October 12, 2026. The record is
[run.json](../../../../../research/smoke/m4-pro-v1-2026-10-12/run.json); the
[smoke README](../../../../../research/smoke/README.md) gives the command and limits.

| Fidelity | Seed | val_bpb | Throughput | Time |
| --- | ---: | ---: | ---: | --- |
| 8m | 1000 | 1.436965 | 22.1k tokens/s | 380.2 s training, 405.4 s process (about 6.8 min) |

An 8m run takes about 6.8 minutes. A 21m run would take about 16 minutes; this
is an estimate from the 8m run, not a measurement.

- This is one sample. It is not a noise estimate, and it cannot rank candidates.
- The score uses the frozen FineWeb-Edu split. It is not comparable to the 1.410
  ClimbMix reproduction, which used a different dataset, budget, and evaluation size.
- The noise floor and rank correlation from the completed [S1 report][s1] apply to
  the legacy AdamW recipe. They do not transfer to this recipe. The decider's
  per-campaign calibration of 5 pairs measures this recipe's noise; no separate S1
  re-run is planned.
- The S1 calibration panel's learning-rate variants were designed for AdamW. Derive
  them again for Muon before reuse.
- `freeze.json` keeps its freeze-time field `native_training_executed: false`. It
  records the state when the recipe was frozen. Later validation lives in the smoke
  record. Its `adapter_hash` and `evaluator_hash` cover openloop adapter files, so
  they change with any adapter edit. The tests treat them as informational and pin
  every other field to the code constants.

## Checks and limits

The [tests](../../../../../tests/loops/test_recipe.py) check source integrity,
model AST equality, both token stops, startup timing, schedules, distinct
identities, and the composer's rewrite counts. They also check that `freeze.json`
agrees with the code constants: recipe revision and hashes, upstream hashes,
baseline settings, fidelities, program hashes, and, when the pinned upstream clone
is present, the source and dependency hashes.
CPU optimizer checks cover tall, wide, and square matrices, zero
gradients, finite updates, and the independent first-step AdamW formula.
They use float32 NumPy arithmetic and do not certify native MLX bfloat16 behavior.
Both full programs compile. All 12 installed package versions match the existing
Python 3.12 / MLX 0.31.0 lock.

The reference score is on ClimbMix with a 300-second training budget and 5,242,880
evaluation tokens. It is not the expected FineWeb-Edu T1 score. Memory beyond
sampled RAM and the noise floor of this port remain unmeasured.

[winner]: https://github.com/elementalcollision/autoresearch/blob/1d8377683d9a5b1af14f5933e6f5130c54e94460/train_mlx.py
[wiki]: https://github.com/elementalcollision/autoresearch/wiki/Experiment-Results-Mar-14-2026-M4-Pro/3337ce96ae8c5410fa79088a4273804d7f33bdd8
[trevin]: https://github.com/trevin-creator/autoresearch-mlx#longer-apple-silicon-runs
[reproduction]: ../../../../../research/reproductions/m4-pro-2026-10-10/README.md
[s1]: ../../../../../research/s1/run-2026-10-11/report.json
