# Contributing to openloop

These rules apply to every change, whether a person or an agent writes it.
Openloop exists to make experiment results trustworthy.
Its own code and records must meet the standard it asks of experiments.
Expect expert infrastructure engineers and ML researchers to read every line.

[AGENTS.md](AGENTS.md) adds rules for coding agents.
[DESIGN.md](DESIGN.md) defines the architecture and terms.
[NEEDS.md](NEEDS.md) gives the reason for each capability.

## Setup and checks

```sh
uv sync --locked
uv run pre-commit install
uv run pre-commit run --all-files   # ruff lint and format
uv run pyright                      # static types
uv run pytest
```

A change is complete only when all checks pass locally.
CI runs the same commands.
Commit `uv.lock` with every dependency change.

## Scope

1. Each capability must trace to a need in [NEEDS.md](NEEDS.md).
   If no need supports a feature, do not build it.
2. Keep the core small: ledger, loop interface, executors, and decider.
   Allocation, proposers, memory, agent tools, and UI are plugins.
3. Plugins use the public core API.
   They never write SQLite tables or artifact storage directly.
4. Do not build training frameworks, schedulers, or storage systems.
   Use existing tools behind an interface.
5. Keep one change focused on one purpose.
   Put unrelated cleanups in a separate change.

## Architecture

1. **Pure core, thin adapters.**
   Hashing, validation, comparisons, and decisions are pure functions.
   Only adapters touch files, SQLite, processes, clocks, or the network.
   Pure modules must not import adapter modules.
2. **Dependency direction.**
   `identity` and `records` know nothing about storage.
   `store` depends on them.
   Orchestration, such as `openloop.probe`, depends only on the public `Ledger` API.
3. **Standard library in the core.**
   The ledger and decider use only the standard library.
   A new runtime dependency needs a stated reason in the change description.
   Optional workloads put their dependencies in a dependency group.
4. **One source of truth.**
   Define each set of allowed values once, as a `StrEnum`.
   Build SQL `CHECK` clauses, validation, and docs from that definition.
   Do not repeat literal tuples of allowed values.
5. **Explicit trust boundaries.**
   Name the boundary in the module docstring.
   Candidate code never produces its own authoritative score.
   A Git worktree is not a security boundary.

## Evidence and provenance

These rules protect the ledger.
A violation can invalidate every stored result.

1. **Records are append-only.**
   Never update or delete a published record.
   A correction is a new record that references the old one.
2. **Never invent evidence.**
   Record only times, costs, and measurements that were observed.
   If a value is unknown, leave it absent.
   Do not fill it with a default.
3. **Identity is content.**
   Identify inputs by SHA-256 of complete snapshots.
   Paths, tags, model aliases, and timestamps are not identities.
4. **Hashes are frozen.**
   Do not change canonical encoding, manifest fields, or field meaning without bumping `IDENTITY_VERSION`.
   Golden-hash tests pin known digests.
   If such a test fails, find the cause; do not just update the literal.
5. **Frozen data stays frozen.**
   Do not change anything that feeds `data/*/manifest.json`, the split policies, or recorded artifacts in `research/artifacts/`.
6. **Storage schema changes need a version.**
   Bump `SCHEMA_VERSION` and add an explicit migration step.
   Storage versions never feed into content hashes.
7. **Interrupted work stays visible.**
   A crash must leave either a complete transaction or nothing.
   A running attempt stays running until the coordinator reconciles it.
8. **Secrets never enter records, logs, tests, or tracked files.**
   `.env` stays local.
   Record model aliases as aliases, together with the identifiers that the provider returned.

## Statistical honesty

1. A cache hit references an earlier sample.
   It is never a new sample.
2. Declare `reproducibility` truthfully.
   GPU training is usually `noisy`.
3. Measure the noise floor before claiming an improvement.
   Report the effect size, the interval, and the sample count.
4. Selection uses validation data only.
   Held-out data is read once, after selection closes.
5. An implementation failure is not evidence against a hypothesis.
6. Label simulated results, such as mock-executor scale results, as simulated.
7. Record failures and negative results with the same care as successes.

## Python style

Target Python 3.12.
Ruff enforces formatting and lint rules.
Pyright enforces types.
These rules cover what the tools cannot check.

- **Types.** Annotate every public function and attribute.
  Use `X | None`, `type` aliases, and `collections.abc` types.
  Avoid `Any` and `cast`; if you need one, add a comment that gives the reason.
  The ledger uses pyright strict mode.
  Modules that validate untyped callers at runtime disable only `reportUnnecessaryIsInstance`, with a file-level pragma that gives the reason.
- **Records.** Use `@dataclass(frozen=True)` for value objects.
  Validate in `__post_init__`, so an invalid object cannot exist.
  Freeze nested mappings and sequences.
- **Errors.** Raise a specific exception from the package hierarchy, for example `openloop.ledger.errors`.
  Do not raise a bare `ValueError` or `Exception` for domain rules.
  Never catch broad exceptions unless you re-raise, or you record the failure and then re-raise.
- **APIs.** Make optional arguments keyword-only.
  Accept plain strings at the boundary, then convert them to enums once.
  Return immutable snapshots, never live database rows.
- **SQL.** Use parameterized queries only.
  Name the columns in every `INSERT`.
  Keep each multi-step write in one explicit transaction.
- **Names before comments.** Choose names that make comments unnecessary.
  Write comments to explain *why*: invariants, trade-offs, and surprising platform behavior.
  Do not narrate the code.
- **Docstrings.** Each module docstring states the module's purpose and boundary.
  Each public docstring states the contract, not the implementation.
- **Size.** Prefer small modules with one responsibility.
  Split a module before it reaches about 500 lines.
- **No dead code.** Do not commit commented-out code, unused parameters, or speculative hooks.
  Every `TODO` needs a linked issue.

## Tests

1. Test behavior through the public API.
   Mirror the source layout: `src/openloop/ledger/store.py` → `tests/ledger/test_store.py`.
2. Every bug fix adds a test that fails without the fix.
3. Test failure paths.
   After a rejected operation, assert that the stored evidence did not change.
4. Durability claims need crash tests, such as `SIGKILL` during a write.
   Concurrency claims need concurrency tests.
5. Use property-based tests (Hypothesis) for pure functions with invariants, such as canonical encoding and hashing.
6. The default suite is fast, deterministic, and offline.
   It uses no network, paid APIs, or GPU.
   Hardware-specific tests skip themselves on unsupported platforms.
7. Do not weaken a test so that it passes.
   If the expected behavior changes, say so in the change description.

## Documentation

1. Update the documentation in the same change as the code.
2. Write in short, declarative sentences with consistent terms ([STE-100][ste100]).
   Use the terms in the [DESIGN.md glossary](DESIGN.md#terms).
3. Keep implemented features and planned features clearly separate.
   Never describe a plan as done.
4. Link each factual claim to its evidence: a source, a log, or a recorded artifact.
5. Use absolute dates.

## Changes and review

- Explain *why* in the change description, and state what was verified and how.
- State what was not verified.
- Spending real money, such as API calls or GPU time, needs explicit approval from the owner.

[ste100]: https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf
