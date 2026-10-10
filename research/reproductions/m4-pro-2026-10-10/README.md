# Public M4 Pro reproduction

Date: October 10, 2026. Status: sanity check passed.

The unmodified public winner produced **1.410165 val_bpb**.
The published reference is **1.429396**.
The absolute gap is **0.019231 BPB**.
It is below the 0.05 BPB debug threshold declared before the result.
This threshold is a practical sanity check, not a statistical decision rule.
No further training was required for this check.

## Reference and inputs

The reference is from [elementalcollision/autoresearch](https://github.com/elementalcollision/autoresearch/blob/1d8377683d9a5b1af14f5933e6f5130c54e94460/train_mlx.py).
It is not the trevin-creator fork used in the [earlier bring-up](../../autoresearch-mlx.md).
The [pinned wiki](https://github.com/elementalcollision/autoresearch/wiki/Experiment-Results-Mar-14-2026-M4-Pro/3337ce96ae8c5410fa79088a4273804d7f33bdd8) identifies winning commit `1d83776`.
The full commit is `1d8377683d9a5b1af14f5933e6f5130c54e94460`.
The [reference record](reference.json) retains the wiki revision, page hash, configuration, and debug threshold.

The run used the original `train_mlx.py`, `prepare.py`, hardware detector, and Muon optimizer.
All four SHA-256 hashes were checked before training and again after it.
None of these files changed.
The recipe uses depth 6, width 384, a 1.5x MLP, and an 8,192-token batch.
The device batch is 4 and the cooldown fraction is 0.7.
The seed is 42, the sequence length is 2048, and the vocabulary has 8192 tokens.
The model has 21,922,092 parameters.

The data is the original ClimbMix cache: ten training shards and validation shard 6542.
It uses the existing tokenizer and token-byte lookup.
All 13 cached file hashes match the recorded bring-up snapshot.
The cache still matches those hashes after the run.
The tokenizer preparation algorithm uses the same corpus, text cap, split pattern, vocabulary, and token-byte method.
No tokenizer was retrained.
The publisher did not publish input-byte hashes, so identical bytes on the publisher's machine are not verified.

The original evaluator scored **5,242,880 validation tokens**, with batch size 4.
It was not replaced or shortened.
No frozen FineWeb-Edu membership file or held-out data was used.

## Measured result

| Field | Public reference | Local reproduction |
| --- | ---: | ---: |
| val_bpb | 1.429396 | **1.410165** |
| GPU cores | 16 | 20 |
| Training steps | 751 | 824 |
| Training seconds | 300-second budget | 300.2 |
| Reported peak MLX memory | 4511 MiB | 4510.8 MiB |
| Total script seconds | Not published | 377.4 |
| Timed training tokens/s | Not published | approximately **22,186** |

The local machine has an M4 Pro, 24 GiB of RAM, and macOS 26.6.2.
It was connected to AC power before the run.
The runtime was Python 3.12.13 with MLX and mlx-metal 0.31.0.
The [provenance record](run-01/provenance.json) lists all observed package versions and input hashes.
The historical `uv.lock` does not include MLX and differs from its project metadata.
It does not establish an exact publisher runtime.
This run reused the existing bring-up environment and records its versions explicitly.
It does not claim a historical dependency match.

The source excludes updates numbered 0 through 10 from its training timer.
Timed throughput is therefore `(824 - 11) * 8192 / 300.2`.
The total is 6,660,096 timed tokens.
The rate is approximate because the reported training time is rounded to 0.1 seconds.
It excludes setup and final evaluation.
The source's MFU field is an estimate and is not used as a measured utilization result.

The local executor enforced one GPU lane, a 900-second process wall limit, 16 GiB of sampled resident RAM, and 16 MiB per log stream.
RAM was sampled every five seconds to reduce observer overhead.
Resident RAM does not measure separate Metal allocations.
The worker exited with code 0 and produced no stderr output.

Evidence:

- [Full stdout](run-01/jobs/78c6ec79a28443c2aeb73473e445e63e/stdout).
- [Full stderr](run-01/jobs/78c6ec79a28443c2aeb73473e445e63e/stderr).
- [Execution receipt and log hashes](run-01/execution.json).
- [Final numeric summary](run-01/summary.json).
- [Comparison and post-run hash audit](assessment.json).

## Diagnosis and limits

The earlier 1.76 BPB result used a different fork and recipe.
That run used AdamW, depth 4, a 4x MLP, and a 65,536-token batch.
Its evaluator used 1,572,864 validation tokens.
The public reference uses Muon, depth 6, a narrower MLP, and a much smaller batch.
The earlier result was not a reproduction of this reference.
Its gap cannot be treated as an environment failure without matching those inputs.

This run executes the correct pinned recipe and falls within the declared tolerance.
It fits the expected memory range and performs 73 more steps than the reference.
The higher step count is consistent with the extra GPU cores.
That causal explanation is not proven by one run.
The lower BPB is not evidence of a new algorithmic improvement.
There is one fixed-seed sample, no confidence interval, and no measured noise floor.

The frozen FineWeb-Edu T1 adapter was not changed or trained.
Its separate native baseline and the later recipe port remain pending tasks.
Existing frozen artifacts, data manifests, setup records, and SQLite ledgers were not modified.

## Reproduce

Run these commands from the openloop checkout.
The script requires explicit approval because it starts local GPU training.
Choose a new output directory for each run; existing evidence is never overwritten.

```sh
git clone https://github.com/elementalcollision/autoresearch.git /tmp/m4-pro-reference
git -C /tmp/m4-pro-reference checkout 1d8377683d9a5b1af14f5933e6f5130c54e94460
uv run python research/reproduce_m4_pro.py \
  --source /tmp/m4-pro-reference \
  --python research/autoresearch-mlx/.venv/bin/python \
  --output research/reproductions/m4-pro-2026-10-10/run-02 \
  --approve-local-training
```

The script checks the pinned source and existing cache before execution.
It captures runtime versions and retains the executor's logs and receipt.
The [tests](../../../tests/research/test_reproduce_m4_pro.py) use synthetic summaries and no GPU.
They check malformed results, evidence overwrite refusal, and virtual-environment interpreter selection.
