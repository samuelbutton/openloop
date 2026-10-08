# Frozen FineWeb-Edu corpus

This is a bounded local corpus for BUILDPLAN.md's tiny-model experiments and
data-mixture studies: the first three shards of the 10BT sample, not all of
FineWeb-Edu or the full 10BT sample. Raw text and membership Parquet files stay
local and are ignored by Git. The source specification and frozen manifest are
small, reviewable artifacts.

Frozen October 8, 2026 at dataset revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
The three source shards total 6,456,837,861 bytes and 2,182,000 documents.
Removing 18,340 exact duplicates leaves these memberships:

| Split | Unique documents | Membership SHA-256 |
| --- | ---: | --- |
| train | 1,947,744 | `0f9d8b16b96b7b9365c83b65cc4f76f51df697ad716f0903f719464f84678912` |
| val | 107,770 | `a50301f184f785777acecc6135e1b636db4ce7b482d19d44f0ad1a0f759078ba` |
| held_out | 108,146 | `8bdebdcfad0e045a2cafd5dd6e2ee1c3bf62084206a10bdd429352e26568c824` |

Snapshot SHA-256:
`93ea20c138c41e1e463324917a295a2d41b2fd1e4a1d01f690c889ff9d9ccb9f`.
[Manifest](manifest.json) records the individual source and membership file
hashes. All source bytes were verified against upstream checksums. An independent
full-membership audit checked assignments, reference bounds, document counts,
logical hashes, and absence of exact-content overlap between splits.

## Prepare and verify

From the repository root:

```sh
uv sync --locked
uv run python -m openloop.fineweb
uv run python -m openloop.fineweb --verify
```

Preparation uses the full upstream revision in `source.json`, not a moving
branch. Every downloaded shard must match the upstream byte size and SHA-256
before it is published to `raw/`. A frozen `manifest.json` is never regenerated
by a repeated preparation command: its artifacts are verified instead. Missing
raw files are downloaded from the same pinned source, and missing memberships
are restored only if their hashes match the existing manifest. Changed
source specifications require a new snapshot directory. Verification is offline.

## Split contract

Each document is identified by SHA-256 of its exact UTF-8 text. The first eight
digest bytes, interpreted as an unsigned big-endian integer modulo 10,000,
assign it to one of these half-open bucket intervals:

| Split | Buckets | Intended proportion | Use |
| --- | --- | ---: | --- |
| train | [0, 9000) | 90% | Training and training-data mixture changes |
| val | [9000, 9500) | 5% | Iterative evaluation and model selection |
| held_out | [9500, 10000) | 5% | Final confirmation after candidates are frozen |

Exact duplicate texts are retained only once, taking the first occurrence in
source-shard order and then row order. Identical content consequently cannot
cross split boundaries. Text is not normalized, and this does not establish
absence of near duplicates or related pages. Proportions are per unique document,
not exact row quotas or token proportions. Changing a tokenizer does not change
the split assignment.

`splits/*.parquet` contain references, not copied text: `shard_index` indexes the
manifest's ordered `shards` list, `row_index` is the zero-based physical Parquet
row in that source shard, and `text_sha256` identifies the selected content.
Future dataloaders must apply these memberships; reading whole raw shards would
mix the three splits. Tokenization and sequence packing happen **within** each
split. No tokenizer was trained and no held-out evaluation was performed here.

Each split records a Parquet-file SHA-256 and a logical membership SHA-256. The
logical hash streams records in source order as big-endian `uint16 shard_index`,
`uint64 row_index`, then the 32 text-digest bytes. It does not depend on Parquet
compression or metadata. The snapshot hash covers canonical JSON of the entire
manifest except `snapshot_sha256`, using sorted keys, compact separators, and
UTF-8 encoding. It binds source files, split memberships, policy, and preparation
format version. Individual source shards also retain their upstream SHA-256.

## Attribution

FineWeb-Edu was published by Anton Lozhkov, Loubna Ben Allal, Leandro von Werra,
and Thomas Wolf (2024), Hugging Face. Dataset DOI: `10.57967/hf/2497`.

[Pinned dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md).
The dataset uses [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/),
and its card also points to [Common Crawl's terms](https://commoncrawl.org/terms-of-use).
Openloop's Apache-2.0 source-code license does not replace the dataset license.
