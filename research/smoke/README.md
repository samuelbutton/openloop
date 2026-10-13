# Native smoke runs

Each directory here holds the evidence of one approved native run that checks a
shared recipe end to end on the local machine. These runs are smoke tests of
the pipeline and of throughput. They are not noise estimates and not benchmarks.
Do not edit or overwrite the files; `research/smoke_t1.py` refuses to reuse a
directory.

## m4-pro-v1-2026-10-12

One 8m run of the shared starting recipe `m4-pro-v1`
(see [the recipe](../../src/openloop/loops/recipes/m4_pro_v1/README.md)).

| Item | Value |
| --- | --- |
| Machine | Apple M4 Pro, 24 GiB RAM (25,769,803,776 bytes), macOS 26.6.2 |
| Fidelity, seed | `8m` (8,388,608 training tokens), seed 1000 |
| Data | Frozen FineWeb-Edu split, validation on 1,572,864 tokens |
| `val_bpb` | 1.4369654020537048 |
| Throughput | 22,065 tokens/s (`tokens_per_second` observation) |
| Training time | 380.2 s |
| Process time | 405.4 s (about 6.8 min per 8m run) |
| Peak sampled RAM | 348,602,368 bytes (excludes Metal and GPU memory) |

Files: [`run.json`](m4-pro-v1-2026-10-12/run.json) is the ledger record,
[`started.json`](m4-pro-v1-2026-10-12/started.json) and
[`finished.json`](m4-pro-v1-2026-10-12/finished.json) bracket the run,
`ledger.sqlite3` is the isolated ledger, `executor/` holds the worker logs, and
`m4-pro-v1-2026-10-12.log` is the runner's console log, which is empty.
The ledger run has purpose `replication`, so it never substitutes for a cache hit.

### Command

[`research/smoke_t1.py`](../smoke_t1.py) runs one job through `T1`, `ExperimentEnv`,
a `LocalExecutor`, and a new ledger. It starts local MLX GPU training and requires
`--approve-local-training`. The exact command, run from the repository root on October 12, 2026:

```sh
nohup caffeinate -i uv run python research/smoke_t1.py \
  --upstream research/autoresearch-mlx \
  --corpus data/fineweb-edu \
  --tokenizer "$HOME/.cache/autoresearch/tokenizer" \
  --worker-python "$PWD/research/autoresearch-mlx/.venv/bin/python" \
  --output research/smoke/m4-pro-v1-2026-10-12 \
  --approve-local-training > research/smoke/m4-pro-v1-2026-10-12.log 2>&1 < /dev/null &
```

The defaults supplied fidelity `8m` and seed `1000`.
`caffeinate -i` kept the Mac awake.

### Limits

- One sample. It says nothing about run-to-run noise. Do not use the score to
  compare candidates or to set a promotion threshold.
- It is on the frozen FineWeb-Edu split. It is not comparable to the 1.410 val_bpb
  of the ClimbMix reproduction in `research/reproductions/m4-pro-2026-10-10/`,
  which used a different dataset, a 300-second training budget, and 5,242,880
  evaluation tokens.
- The S1 part 1 noise floor and rank correlation apply to the legacy AdamW recipe.
  The decider's per-campaign 5-pair calibration measures this recipe's noise.
  No separate S1 re-run is planned.
- The 21m time, about 16 minutes, is an estimate from this run: 2.5 times the
  training seconds plus the observed setup and evaluation overhead. It was not measured.
- The record was produced with the adapter files as they were on October 12, 2026.
  Later edits to files in `ADAPTER_FILES` change `adapter_hash` and `evaluator_hash`
  of new runs, not of this record.
