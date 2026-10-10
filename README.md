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
uv run pyright
uv run pytest
```

Source code lives in `src/openloop/`; tests live in `tests/`.
Commit `uv.lock` when changing dependencies so development and CI use the same versions.
Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code; agents also follow [AGENTS.md](AGENTS.md).

## Environment ledger

On native Apple silicon macOS, `uv sync --locked` also installs MLX and mlx-lm.
Capture this machine's environment as the first ledger field:

```sh
uv run python -m openloop.env_probe --output ledger.json
```

The probe prints JSON with two fields. `environment` holds only stable facts that
can affect results: the chip, physical RAM in bytes, macOS, architecture, and the
Python and MLX versions. Hash this field for `ExperimentInputs.environment_hash`.
`calibration` holds the capture time and a GPU matmul benchmark; it is a
measurement, so it is excluded from identity. The saved `ledger.json` is local and
ignored by Git; existing files are never overwritten. Omit `--output` to print a
fresh record without saving it.

The default benchmark multiplies two seeded 2048×2048 float32 matrices on Metal,
warms up three times, then measures ten runs. Inputs are evaluated before timing;
each matmul is evaluated and synchronized before the timer stops. Raw samples,
median latency, and GFLOP/s (`2*N³ / seconds / 1e9`) are recorded. This measures
warm matmul latency including dispatch and synchronization, not LLM throughput.
Use `--size`, `--warmup`, and `--repeats` to change the calibration workload.

Linux CI checks the portable logic and skips the Metal integration test.

## Experiment ledger

[`openloop.ledger`](docs/LEDGER.md) records immutable experiment inputs and
append-only execution evidence in SQLite. It provides `submit`, `get`, `lineage`,
and `query`, with per-stage UTC timestamps, cache reuse, retries, and a
`candidate_hash` that groups one candidate's samples across seeds. Reused
submissions preserve provenance without adding samples.

[`openloop.probe`](src/openloop/probe.py) runs an explicit replication and records
the comparison. Inputs declare `reproducibility`: drift on `deterministic` inputs
disables result-cache reuse, while drift on `noisy` inputs is recorded as measured
variation.

See the [ledger guide](docs/LEDGER.md) for the schema, API, and a synthetic example.
The new SQLite ledger is separate from the setup record in `ledger.json`.

## Loop interface

[`openloop.loops`](docs/LOOPS.md) provides an `Env`-shaped contract through the
ledger API. T0 supplies a planted objective with seeded noise. T1 adapts the pinned
MLX training script to exact token budgets at two fidelities (8m and 21m tokens).
Candidate settings, frozen evaluators, and work limits are checked before publication.
The [loop guide](docs/LOOPS.md) includes an offline example and native-worker requirements.

The [public M4 Pro reproduction](research/reproductions/m4-pro-2026-10-10/README.md)
reached 1.410165 val_bpb with the unmodified published winner, against its 1.429396
reference. This uses the original ClimbMix evaluator; the frozen FineWeb-Edu T1
baseline remains separate.

## Executors

[`openloop.executors`](docs/EXECUTORS.md) provides local processes and committed Git
source inside Docker. Both expose `submit`, `poll`, `cancel`, and `collect`, with
idempotent keys, lifecycle timestamps, resource limits, and hashed logs.
The sandbox denies network access by default. T1 uses the local executor and one
shared GPU lane. See the [executor guide](docs/EXECUTORS.md) for trust boundaries.

## Decider

[`openloop.decider`](docs/DECIDER.md) freezes the comparison family, noise
calibration, and seed schedules. It offers noise-aware and AutoScientists screen
policies. Both use fresh-seed confirmation with a family error bound, followed by
one trusted held-out batch after selection closes. The coordinator is in memory;
held-out file isolation and durable campaign records remain host responsibilities.
The [CPU-only T0 study](docs/decider-t0.json) measures detection rate across effect sizes
and false-confirmation rate under nulls for each decision rule on simulated truth.

```sh
uv run python -m openloop.studies.t0_decider --trials 384 --output docs/decider-t0.json
```

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
