# openloop needs

Status: proposed needs. Evidence review date: October 8, 2026.

This document preserves the N1–N10 identifiers from the build plan.
The needs guide openloop development and the Act 3 proposal.
They are not requirements that Discovery Loop has approved.

Public statements explain the research goal.
Local measurements explain the current workbench.
Design inferences connect that evidence to proposed features.
Confidence describes support for a need, not completion of its acceptance check.

## N1–N10

| ID | Need | Evidence and inference | Confidence | Design | Acceptance check |
| --- | --- | --- | --- | --- | --- |
| N1 | Make every experiment traceable and reproducible. | [Discovery Loop][company] describes a repeated experiment cycle. We infer that unattended experiments need complete provenance. The plan also cites [BigGo][biggo]; that report was not accessible during this review. | High for provenance; the BigGo claim remains unchecked. | [Ledger](DESIGN.md#ledger-schema) | Recover each run's code, configuration, data, seed, environment, artifacts, and decision. Repeat a selected run and report drift. |
| N2 | Check whether an improvement is real before using it. | The [MLX README][mlx-readme] describes noisy scores. Our [unchanged runs](research/autoresearch-mlx.md) had a rigor standard deviation of 0.029889 BPB. This measures fixed-seed run variation. | High | [Decider](DESIGN.md#decider) | Reject planted false improvements in T0. Report effect size, uncertainty, and held-out results for selected T1 candidates. |
| N3 | Run many experiments in parallel and direct compute toward useful experiments. | [Discovery Loop][company] states a goal of thousands of parallel experiments. The [AASF report][aasf] describes allocation by expected value. | High for scale; allocation is a reported goal. | [Executors](DESIGN.md#executors) and [allocator plugins](DESIGN.md#plugins) | Compare allocation policies with recorded outcomes at 1 to 10,000 simulated workers. Separate simulation results from actual throughput. |
| N4 | Connect architecture, data, evaluation, and training loops. | The [AASF report][aasf] describes these connected loops. A shared contract and ledger are our design inference. | High | [Loop interface](DESIGN.md#loop-interface) | Run two loop types through one contract. Record both parents of a combined candidate. Test the combination before claiming a gain. |
| N5 | Reduce the time from hypothesis to verified decision. | The [AASF report][aasf] describes an iteration target of one hour instead of one week. Our first requirement is measurement. | High for the goal; the time target is untested. | [Run sequence](DESIGN.md#run-sequence) | Report total elapsed time and each stage's elapsed time. Compare those measurements with a recorded baseline. |
| N6 | Let agents operate the system while people set direction. | [Discovery Loop][company] describes AI systems that propose, run, and learn. Agent tools and human approval are design inferences. The plan cites [BigGo][biggo] for further context. | High for agent use; the interaction design is proposed. | [Plugins](DESIGN.md#plugins) | Use one typed Python API for agent and human tools. Require recorded approval when an action exceeds its approved limits. |
| N7 | Use previous results to select the next experiment. | [Discovery Loop][company] describes learning from evaluations. Research memory is our inference. The plan's Bayesian description is an analyst interpretation. | Medium-high | [Research memory](DESIGN.md#plugins) | Link each finding to decisions and samples. Keep failed hypotheses available without describing them as verified findings. |
| N8 | Contain autonomous experiment code. | The [AASF report][aasf] describes an agent that left its test environment. Restricted execution is our response to that reported risk. | High | [Executors](DESIGN.md#executors) | Test network denial, protected-file access, resource limits, and evaluator protection. A Git worktree alone does not satisfy this check. |
| N9 | Keep the core small and stable for a small team. | The [AASF report][aasf] describes Dean's small-core principle. [Discovery Loop][company] describes a lean team. Plugin boundaries are our design choice. | High | [Core and plugins](DESIGN.md#core-and-plugins) | Add a loop or allocation policy without changing ledger storage or executor lifecycle rules. |
| N10 | Support Google Cloud execution. | [Google][google] confirms its Cloud partnership. This supports portability as a need. It does not establish Discovery Loop's internal architecture. | High for the partnership; the backend design is inferred. | [Executors](DESIGN.md#executors) | First check the GCP stub against the executor contract. Later verify a real job, cancellation, artifact transfer, and cost records. |

## Scope and current evidence

The first implementation covers the ledger, loop interface, executors, and decider.
T0 supplies planted results for verification tests.
T1 supplies an MLX training experiment through the same interface.
Plugins follow after these two loops work.

Current evidence includes the [environment probe](src/openloop/env_probe.py), [MLX runs](research/autoresearch-mlx.md), and [frozen FineWeb-Edu splits](data/fineweb-edu/README.md).
The [tabular reports](data/tabular/README.md) confirm that NAS-Bench-201 and HPO-B load.
These checks do not establish allocator performance or a verified research improvement.
The [agent setup](AGENT_SETUP.md) records model choices and the approved account limits.

Training frameworks, cluster scheduling internals, and new storage systems are outside the core.
Chip design and domain science are outside this first implementation.
The GCP stub will not establish production cloud support.

## Evidence sources

| Source | Type | Review result |
| --- | --- | --- |
| [Discovery Loop][company] | Primary company statement | Accessible. Supports automated loops, parallel experiments, learning from evaluations, and a lean team. |
| [AASF coverage][aasf] | Secondary report, August 8, 2026 | Accessible. Supports the reported connected loops, latency goal, allocation goal, containment concern, and small-core principle. |
| [Google announcement][google] | Primary partner statement | Accessible. Confirms the founding investment and Cloud partnership. |
| [MLX README][mlx-readme] | Upstream technical source, pinned revision | Available in the local clone. Explains score variation and the limits of a running-best curve. |
| [Local MLX report](research/autoresearch-mlx.md) | Local measurements and source analysis | Records four unchanged runs and the exact upstream keep/discard rules. |
| [BigGo report][biggo] | Secondary source from the build plan | Not accessible during this review. Retained for traceability; its specific claims remain unchecked. |

The documents use short sentences and consistent terms from [STE-100 Issue 9][ste100].
The [design glossary](DESIGN.md#terms) defines the main technical nouns.

[company]: https://www.discoveryloop.com/
[aasf]: https://stanfordtechreview.com/articles/jeff-dean-first-talk-since-leaving-google
[google]: https://blog.google/company-news/inside-google/message-ceo/next-chapter-ai-momentum/
[biggo]: https://finance.biggo.com/news/e51246dd-cf55-4e14-b3d5-021583d75a8a
[mlx-readme]: https://github.com/trevin-creator/autoresearch-mlx/blob/766a25ff22afa799efd8d0aa450a4348e4749df2/README.md
[ste100]: https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf
