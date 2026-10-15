# Researcher UI v0

The UI plugin implements the first researcher view (N5, N6).
It reads the public ledger API and shares the toolbelt's run comparison logic.
It adds no runtime dependency, build tool, network asset, or ledger schema.

## Start

```sh
uv run python -m openloop.ui --ledger /absolute/path/ledger.sqlite3
```

Open the printed loopback URL. The default port is 8765.
Use `--port 0` to select a free port. Stop the server with Ctrl+C.
The ledger must already exist. The server does not create or migrate it.
Each request uses `Ledger(..., read_only=True)`. SQLite may create shared-memory
sidecars for a WAL database; see the [ledger guide](LEDGER.md).
The server binds only to `127.0.0.1`. It rejects foreign Host and Origin headers.
Only the fixed static assets and read-only JSON routes are available.
This is a local development server, not a public deployment server.

## Experiment DAG

Nodes represent submissions, with status, fidelity, and seed labels.
Solid edges show recorded parent links, including retry ancestry.
Dashed edges show cache reuse. A cache node does not add a measured sample.
Select a node with the mouse or keyboard to inspect its inputs, validation scores,
simulation marker, and recorded stage durations. A running stage has no invented
duration. Inspector buttons select a baseline or candidate for comparison.

The graph shows at most the latest 500 selection submissions.
A notice states when the graph is incomplete and counts parent references outside
the view. Curves still use all selection history. Point evidence retains full run
IDs even when those runs are outside the graph window.

## Curves and noise bands

The chart plots recorded endpoint scores against requested work budget.
Connecting lines are visual guides. They are not measurements between endpoints
or training traces reconstructed from artifact hashes.

Choose a comparison scope before comparing candidates. Scope retains loop, data,
evaluator, environment, tokenizer, phase, execution settings, budget unit, and
metric metadata. It varies candidate, seed, fidelity, and budget amount.
Different phases, metric sample counts, or simulation settings remain separate.
Each candidate has one line per fidelity and one point per budget on that line.
Lines never join endpoints from different fidelities.

The point mean gives each distinct seed equal weight. Repeat attempts at one seed
are averaged first. The noise band is the mean plus or minus one sample standard
deviation of the seed means. A single seed has no band. Cache aliases and failed
or unfinished attempts never contribute scores. Duplicate attempt references do
not increase the count. The point evidence table lists seed count, attempt count,
standard deviation, and original run IDs.

These bands describe observed seed variation. They are not confidence intervals,
the decider's paired noise calibration, or verification decisions.
Toggle candidate and fidelity lines in the legend. Select a point to open its evidence table.
Simulation markers remain visible in the scope and run inspector.

## Diff view

Select two runs, then use **Compare runs**. The view shows validation scores,
input changes, run status, and whether they reference distinct attempts.
It uses the [toolbelt comparison contract](TOOLBELT.md#run-diff).
Metric metadata, seed, context, and distinct attempts must match before a delta
is shown. A positive gain means improvement for either metric direction.
Missing or incomparable results have no invented delta. A diff makes no causal
or verification claim.

## Access and refresh

Only `screen` and `confirm` submissions and validation metrics are displayed.
Held-out scores, raw artifact files, and research memory are not released.
Ledger text is treated as data. The browser uses DOM text insertion instead of
interpreting it as HTML. The server sends a Content Security Policy, rejects
cross-site requests, and exposes no write route or arbitrary file path.
An unreadable, busy, or corrupt ledger returns a JSON error instead of a
dropped connection.

Refresh reads the ledger again. The displayed read time is the browser time of
the successful read, not an experiment timestamp. A failed refresh keeps the
previous view and shows an error. There is no automatic polling.
The interface supports narrow and desktop layouts and keyboard controls.
Approval queues, auto-ablation, anomaly analysis, and morning reports remain planned.

## Evidence

The [report tests](../tests/ui/test_reports.py) cover lineage, repeated seeds,
cache exclusion, context separation, single-seed points, and order invariance.
The [HTTP tests](../tests/ui/test_server.py) use temporary loopback servers and
verify routes, headers, read-only ledger bytes, and rejected writes.
Browser checks used a separate simulated fixture. They covered graph inspection,
seed bands, evidence disclosure, and a run diff. No training was required.
