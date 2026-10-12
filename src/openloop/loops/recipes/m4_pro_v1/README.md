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
before composition. Their MIT license is retained in [LICENSE](LICENSE).
Credit belongs to Andrej Karpathy, Trevin Peterson, and the elementalcollision
contributors for the upstream models and optimizer port. The openloop adapter
uses the repository's Apache-2.0 license.

The pure [composer](../../recipe.py) inlines the optimizer and fixes depth 6.
It removes hardware detection and estimated MFU output. It connects the published
data-loader call to T1's frozen loader interface. The model and optimizer arithmetic
remain unchanged. The composed training source has a separate enforced SHA-256 pin.
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

No training of the port has run. The reference score is on ClimbMix with a
300-second training budget and 5,242,880 evaluation tokens. It is not the expected
FineWeb-Edu T1 score. Performance, native memory use, and the noise floor of this
port remain unverified. The completed [S1 report][s1] applies to the legacy recipe.
Recalibrate the shared recipe before using a noise threshold for promotions.

[winner]: https://github.com/elementalcollision/autoresearch/blob/1d8377683d9a5b1af14f5933e6f5130c54e94460/train_mlx.py
[wiki]: https://github.com/elementalcollision/autoresearch/wiki/Experiment-Results-Mar-14-2026-M4-Pro/3337ce96ae8c5410fa79088a4273804d7f33bdd8
[trevin]: https://github.com/trevin-creator/autoresearch-mlx#longer-apple-silicon-runs
[reproduction]: ../../../../../research/reproductions/m4-pro-2026-10-10/README.md
[s1]: ../../../../../research/s1/run-2026-10-11/report.json
