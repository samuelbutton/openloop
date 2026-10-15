# T2 engineering evidence

Recorded on October 15, 2026. These are implementation checks, not study S6.
The [T2 guide](../../docs/T2.md) defines the recipe and its limitations.
The [training guide](../../docs/TRAINING.md) defines the numerical contract.

| Evidence | Scope |
| --- | --- |
| [Model snapshot](model-snapshot.json) | Pinned public Qwen2.5 0.5B 4-bit asset hashes; weights remain local. |
| [Current backend checks](backend-correctness-namespaced-2026-10-15.json) | 15 passing checks on MLX CPU, PyTorch CPU, and PyTorch MPS, with source hashes. |
| [Current smoke plan](smoke-plan-2026-10-15-namespaced.json) and [report](run-smoke-2026-10-15-namespaced/report.json) | Namespaced sample seeds and split token keys; 2 updates, 8/16 validation passes, mixed rewards and nonzero gradients. No measured quality gain. |
| [First smoke](run-smoke-2026-10-15/report.json) and [diagnostic](diagnostic-2026-10-15.json) | Tag-only format rejected correct bare answers. Retained as engineering failures. |
| [Plain-integer smoke](run-smoke-2026-10-15b/report.json) and [receipt-check repeat](run-smoke-2026-10-15-final/report.json) | Before the random-key overlap correction. Do not use as independent-rollout evidence. |
| [Initial backend checks](backend-correctness-2026-10-15.json) | 13 passing numerical checks before the added PRNG regression. |
| [Intermediate key-split checks](backend-correctness-split-keys-2026-10-15.json) and [smoke](run-smoke-2026-10-15-split-keys/report.json) | 15 tests; the smoke had two uniform-reward groups. Retained before final namespacing of sample seeds. |

Each run retains its plan, source snapshot, coordinator report, and executor logs.
SQLite ledgers and model/optimizer checkpoint binaries remain local and ignored.
The diagnostic's first launch failed before model loading because resolving the
interpreter symlink bypassed its virtual environment. The retry used that
environment's interpreter and produced the linked diagnostic. It ran inference
only, on already-used training tasks.

Scoring and sampling corrections followed observed engineering outcomes. These
records do not establish a training gain, noise floor, seed-level replication,
held-out accuracy, CUDA correctness, or performance of a 1.5B model.
