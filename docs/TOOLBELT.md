# Agent toolbelt v0

`openloop.toolbelt` implements the first agent toolbelt plugin (N6).
It provides three operations through Python and an MCP stdio server.
The server uses the [official MCP Python SDK](https://py.sdk.modelcontextprotocol.io/).
The SDK is in the `toolbelt` dependency group. The `dev` group includes it so CI
can test the server. The Python toolbelt API does not load the MCP SDK.

## Start the server

```sh
uv sync --locked --group toolbelt
uv run --locked --group toolbelt python -m openloop.toolbelt \
  --ledger /absolute/path/working.sqlite3 \
  --repository /absolute/path/openloop
```

Use Git 2.45 or newer. The ledger must exist. The repository must be a Git
top-level directory with a commit. Supply both paths in the host configuration. Tool calls cannot change
them. Standard output contains MCP messages only. Diagnostics use standard error.
The process opens no HTTP listener and performs no network requests.

A host that accepts an `mcpServers` configuration can use this shape.
Replace each placeholder with an absolute path:

```json
{
  "mcpServers": {
    "openloop": {
      "command": "/absolute/path/openloop/.venv/bin/python",
      "args": [
        "-m", "openloop.toolbelt",
        "--ledger", "/absolute/path/working.sqlite3",
        "--repository", "/absolute/path/openloop"
      ]
    }
  }
}
```

Install the toolbelt dependency group first. No API key or model provider is needed.

## Tools

| Tool | Required arguments | Optional arguments | Result |
| --- | --- | --- | --- |
| `run_diff` | `baseline`, `candidate`: ledger run IDs | None | Input changes, validation metrics, comparable deltas, status and cache references. |
| `metric_query` | `metric`: metric name | `candidate_hash`, `loop_hash`, `fidelity`, `limit`, `offset` | Original successful validation scores with run references and comparison scope. |
| `repository_map` | None | `prefix`, `limit`, `offset` | Committed paths, blob hashes, sizes, and Python definitions with line numbers. |

Tools have read-only, non-destructive, idempotent annotations and no external
service access. Responses contain structured data and text from the SDK.
Input errors return MCP tool errors. They do not change the ledger.

`limit` defaults to 50 and accepts 1 to 200. `offset` starts at zero.
Paged responses contain `items`, `total`, and `next_offset`.
Continue until `next_offset` is null. There is no silent result cutoff.
Queries reflect current records on each call. Repository pages also name the
commit; restart pagination if it changes.

### Run diff

Input changes include source and dependency hashes, configuration, seed, data,
evaluator, environment, fidelity, phase, budget, and execution settings.
Metrics include raw baseline and candidate values. A delta is candidate minus
baseline. A positive gain means improvement for either metric direction.

Deltas require two distinct attempts with matching metric metadata, seed, and
comparison context. A run compared with itself or its cache alias is not comparable.
Source, dependencies, and candidate configuration may differ. Other context
changes produce `comparable: false` and null deltas. Missing measurements stay
null. Infinite arithmetic is rejected.
Cache submissions retain reuse links; they are not independent attempts.
`is_verification` is always false: a run diff does not establish a verified
improvement or explain its cause. Source hashes are reported as hashes. The tool
does not assume that they are Git commits or invent a source patch.

### Metric query

Rows retain units, direction, sample count, seed, fidelity, phase, work budget,
loop, data, evaluator, environment, and comparison-context hash.
Execution settings retain simulation and benchmark markers.
The tool excludes failed and unfinished runs, cache aliases, and replication probes.
It does not aggregate different scopes or count repeated submissions as samples.

Both ledger tools accept selection runs only (`screen` and `confirm`) and expose
metrics with `split="validation"` only. They do not release held-out metrics,
observations, raw logs, or research memory. The toolbelt grants no held-out reuse
permission to selection agents.

### Repository map

The map reads the committed `HEAD` tree. It ignores working-tree edits and
untracked files. Responses name the commit and set `working_tree_included: false`.
Python definitions include qualified names, definition kind, start line, and end
line. Source is parsed with `ast`; it is never imported or executed.
Syntax errors are reported without source excerpts.

The map excludes symlinks, submodules, dotenv names, credential and secret paths,
SSH/AWS/GPG directories, and common private-key extensions. It reads Git blobs,
not working-tree targets. It does not run checkout filters or hooks.
Git's [no-lazy-fetch option](https://git-scm.com/docs/git/2.45.0) prevents missing
objects in a partial clone from triggering a remote fetch or remote helper.
Paths that are not valid UTF-8 are omitted. Git failures, such as an older Git
or a repository owner that Git does not trust, report Git's own error message.
It is not a secret scanner. Use a repository that is appropriate for the agent.

Python blobs larger than 128 KiB remain in the map with `symbols_omitted`.
At most 200 symbols are returned per file; `symbols_total` and
`symbols_truncated` state when more exist. Git commands have a ten-second timeout.
The caller cannot supply a Git revision, shell command, or path outside the root.

## Python API

```python
from openloop.toolbelt import Toolbelt

tools = Toolbelt("/absolute/path/working.sqlite3", "/absolute/path/openloop")
diff = tools.run_diff("baseline-run-id", "candidate-run-id")
scores = tools.metric_query("val_bpb", limit=50)
source_map = tools.repository_map(prefix="src/openloop", limit=50)
```

Each ledger call uses a separate `Ledger(..., read_only=True)` connection.
The server never migrates a ledger, writes a result, starts an experiment, or
reads credential files. SQLite can create shared-memory sidecars when it reads
a WAL database; see the [ledger guide](LEDGER.md#opening-migration-and-frozen-evidence).

## Verification

The [toolbelt tests](../tests/toolbelt/) use temporary ledgers and synthetic Git
histories. They cover metric scope, gain direction, cache reuse, held-out rejection,
pagination, unchanged ledger bytes, committed source maps, path limits, skipped
blobs, and non-execution of Python files.
The official MCP client checks discovery, schemas, structured output, tool errors,
and real subprocess stdio connections in current and legacy protocol modes.
No paid model call, cloud job, or training run is required.
Desktop-host installation remains a separate host configuration step.
