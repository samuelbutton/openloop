# autoresearch-mlx bring-up

Upstream: https://github.com/trevin-creator/autoresearch-mlx

Pinned revision: `766a25ff22afa799efd8d0aa450a4348e4749df2`.
The clone lives at `research/autoresearch-mlx/` and is ignored by the parent repo.
Upstream source and lockfile are unchanged. The experiment uses upstream's
Python 3.12 environment (MLX 0.31.0), separate from openloop's MLX 0.32.3.
Logs and measured results live in `research/artifacts/autoresearch-mlx/`.

## Measured results — October 8, 2026

Hardware: Apple M4 Pro, 24 GiB unified memory, macOS 26.6.2, Python 3.12.13.
All four runs completed successfully, with upstream source unchanged.

| Run | val_bpb (lower is better) | Average timed tokens/sec | Training seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| Standalone unmodified experiment | 1.775823 | 61,344 | 300.2 | 312.5 |
| Rigor repeat 1 | 1.738882 | 61,420 | 300.9 | 310.7 |
| Rigor repeat 2 | 1.803971 | 61,385 | 300.0 | 309.4 |
| Rigor repeat 3 | 1.742396 | 61,999 | 300.2 | 309.9 |

Average timed throughput is `(num_steps - 1) * 65536 / training_seconds`,
excluding the first startup step from both tokens and time. These averages are
approximate because upstream prints training time to one decimal place. They
include the work in each timed training step (including loading the next batch),
and exclude setup and final evaluation. `total_tokens_M` is rounded and includes
the startup step, so dividing that field by training time would overstate the rate.
All runs reported peak MLX memory of 8600.2 MB; this is not total system memory.
Upstream's `mfu_percent=0` is a placeholder, not a measured utilization result.

**Rigor verdict: keep (baseline).** Its three-run mean was **1.761750**, with
population standard deviation **0.029889**. The standalone run is not one of
those three samples. There was no incumbent in the initially empty rigor ledger,
so keep and `p_better=1` were assigned automatically without bootstrapping.
These repeated fixed-seed results span 0.065089 BPB; they demonstrate run-to-run
variation, not an improvement caused by a code change or distinct-seed variance.

Evidence:

- [Preparation log](artifacts/autoresearch-mlx/prepare.log)
- [Standalone training log](artifacts/autoresearch-mlx/unmodified.log)
- [Rigor training/scoring log](artifacts/autoresearch-mlx/rigor.log)
- [Raw rigor ledger](artifacts/autoresearch-mlx/rigor_ledger.jsonl)
- [Machine-readable results](artifacts/autoresearch-mlx/results.json)
- [Source hashes and runtime versions](artifacts/autoresearch-mlx/provenance.json)
- [Cached data/tokenizer hashes](artifacts/autoresearch-mlx/data_manifest.json)

The local openloop `ledger.json` retains its first `environment` field and now
includes these results under `autoresearch_mlx`. Data shards remain in the local
upstream cache; the workbench stores their hashes rather than redistributing them.

## Reproduce

From the openloop repository:

```sh
git clone https://github.com/trevin-creator/autoresearch-mlx.git research/autoresearch-mlx
cd research/autoresearch-mlx
git checkout 766a25ff22afa799efd8d0aa450a4348e4749df2
uv sync --locked --python 3.12
uv run --locked prepare.py
uv run --locked train.py
uv run --locked rigor.py run "unmodified upstream baseline"
```

Preparation downloads the default ten training shards and pinned validation
shard 6542, then trains the default 8192-token BPE tokenizer. Data is cached in
`~/.cache/autoresearch/`. Training uses depth 4, sequence length 2048, batch
65536 tokens, device batch 16, and a 300-second budget. The first optimization
step is excluded from that budget; setup and final evaluation are additional.
Evaluation scores 1,572,864 validation tokens and splits the requested batch 256
into microbatches capped at 2 GiB of float32 logits.

## Exact keep/discard semantics

The basic [`program.md`](https://github.com/trevin-creator/autoresearch-mlx/blob/766a25ff22afa799efd8d0aa450a4348e4749df2/program.md#L94-L109)
protocol keeps a single run if its `val_bpb` is lower than the previous best;
equal or higher means discard and revert. Its earlier prose also allows a
human/agent judgment about simplification versus improvement. `train.py` itself
does not implement that decision or revert anything.

[`rigor.py`](https://github.com/trevin-creator/autoresearch-mlx/blob/766a25ff22afa799efd8d0aa450a4348e4749df2/rigor.py)
implements a different, explicit gate:

1. Compute the first seven hex characters of SHA-1 of the raw `train.py` bytes.
   If that hash has any prior ledger entry (even discard or crash), skip it.
2. Choose the kept ledger entry with the lowest sample mean as the incumbent.
3. Run `uv run train.py` up to `--seeds` times (default 3). A nonzero exit or
   missing final numeric `val_bpb` records `crash` and stops.
4. If there is an incumbent and the first sample is **greater than or equal to
   its mean**, discard immediately, recording `p_better=0`. No uncertainty test
   precedes this shortcut.
5. With no incumbent, automatically keep the completed baseline, recording
   `p_better=1`. That value is a convention, not a calculated probability.
6. Otherwise independently resample incumbent and candidate samples with
   replacement 20,000 times, preserving each group's original sample count.
   Python's bootstrap RNG is fixed at 1234. A bootstrap trial wins only if the
   candidate mean is **strictly lower** than the incumbent mean.
7. Keep iff `wins / 20000 >= --confidence` (default 0.95); otherwise discard.
   The decision uses the unrounded fraction; the ledger rounds it to four
   decimals. Reported dispersion is population standard deviation.

Important limits: the repeated runs do not change the training seed:
[`train.py` fixes it at 42](https://github.com/trevin-creator/autoresearch-mlx/blob/766a25ff22afa799efd8d0aa450a4348e4749df2/train.py#L400).
The hash excludes `prepare.py`, dependencies, hardware, and data. `results.tsv`
does not seed the rigor ledger. `rigor.py` only appends samples and verdicts to
`rigor_ledger.jsonl`; it does not edit code, manipulate Git, or enforce a timeout.
This initial scoring establishes a baseline, not evidence of an improvement.
