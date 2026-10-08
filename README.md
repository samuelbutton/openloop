# openloop

An experiment workbench, starting with Apple silicon environment calibration.

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run pre-commit install
```

Python 3.12 is selected by `.python-version`. uv installs it if needed.

Run the checks:

```sh
uv run pre-commit run --all-files
uv run pytest
```

Source code lives in `src/openloop/`; tests live in `tests/`.
Commit `uv.lock` when changing dependencies so development and CI use the same versions.

## Environment ledger

On native Apple silicon macOS, `uv sync --locked` also installs MLX and mlx-lm.
Capture this machine's environment as the first ledger field:

```sh
uv run python -m openloop.env_probe --output ledger.json
```

The probe prints JSON with an `environment` field containing the chip, physical RAM
in bytes, macOS, Python and MLX versions, and a GPU matmul benchmark. The saved
`ledger.json` is local and ignored by Git; existing files are never overwritten.
Omit `--output` to print a fresh record without saving it.

The default benchmark multiplies two seeded 2048×2048 float32 matrices on Metal,
warms up three times, then measures ten runs. Inputs are evaluated before timing;
each matmul is evaluated and synchronized before the timer stops. Raw samples,
median latency, and GFLOP/s (`2*N³ / seconds / 1e9`) are recorded. This measures
warm matmul latency including dispatch and synchronization, not LLM throughput.
Use `--size`, `--warmup`, and `--repeats` to change the calibration workload.

Linux CI checks the portable logic and skips the Metal integration test.

## Frozen training data

The [FineWeb-Edu corpus](data/fineweb-edu/README.md) uses three pinned shards,
with exact-content duplicates removed and stable 90/5/5 train, validation, and
held-out memberships. Prepare or verify it from the repository root:

```sh
uv run python -m openloop.fineweb
uv run python -m openloop.fineweb --verify
```

The frozen manifest records source checksums, membership checksums, and a snapshot
hash. Raw data and membership Parquet files remain local.

NAS-Bench-201 and HPO-B are also downloaded and verified through their official
loaders. See the [tabular benchmark setup and load reports](data/tabular/README.md).
Their dependencies are isolated in the optional `benchmarks` group.

## License

Licensed under [Apache-2.0](LICENSE).
