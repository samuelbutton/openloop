# Studies S2–S6

Protocol version 1. Prepared on October 15, 2026.

This document fixes the questions, comparisons, analysis rules, and execution
limits for S2–S6. Its Git commit must precede their first study run. It does not
authorize execution, paid API calls, or cloud spending. No S2–S6 study is run by
this change. Existing engineering tests and demonstrations are not study results.

The [design](DESIGN.md), [needs](NEEDS.md), and completed
[S1 calibration](research/s1/README.md) define the context. The
[S1 public-claims audit](research/s1-part2/README.md) has its own protocol and
amendments. This document does not preregister those earlier observations.

## Freeze and admission

Each study needs a committed `research/sN/manifest.json` before its first run.
The manifest instantiates the rules below. It must contain:

- This protocol's commit and SHA-256, study ID, manifest version, and creation time.
- Complete source snapshots, dependency locks, recipe, adapter, evaluator,
  tokenizer, data memberships, environment, and their content hashes.
- Arm matrix, ordered candidate pool or proposal grammar, task IDs, seeds,
  policy parameters, model identifiers, prompts, and initial memory snapshot.
- Every planned campaign, comparison family, stage count, minimum effect,
  error threshold, and maximum number of proposals. Use the rules below.
- Exact caps for tokens, attempts, process hours, API dollars, GPU dollars, and
  concurrency. Include calibration, baselines, confirmation, held-out work,
  failed attempts, and retries. Record owner approval separately.
- Trusted held-out custodian, proof of disjoint document/task membership,
  access controls, analysis code hash, and output locations.

No field may be left as an execution-time choice. Unavailable resources cause a
blocked study, not an undocumented smaller design. A machine-readable manifest
does not replace this protocol or permission to spend. Its hard limits may be
lower than the ceilings below; changing a comparison or sample size needs a
committed amendment before collecting affected evidence.

Register each campaign through `CampaignJournal.register` before its first
calibration or comparison run. Hash the manifest and decision protocol in its
record. A trusted coordinator must reserve budgets before dispatch; the allocator
alone does not reserve compute. Enforce a single local GPU lane. Use a new study ledger and
artifact directory. Do not write to S1 evidence or frozen data manifests.

## Common measurement and analysis rules

### Samples, costs, and failures

A loop seed is a complete search session, not one candidate evaluation. Matched
arms share initial conditions and proposal randomness where possible. Use
distinct experiment identities across phases. Cache aliases count as zero new
samples. Separate training seeds, policy seeds, and deterministic table lookups.

Charge all attempts to their arm, including compilation, failed implementations,
smoke tests, retries, calibration, and final evaluation. Record process time,
queue time, training tokens, API usage, and invoiced costs separately. Never
convert lookup wall time into GPU throughput. Report NAS historical training
seconds separately from actual lookup costs. A promoted NAS run costs its full
new fidelity; there is no checkpoint credit.

Allow an initial implementation attempt and at most two repairs. Infrastructure
failure permits one retry with the same inputs and seed. Retain both attempts.
Use the first valid result; never retry an unfavorable valid score. A further
failure leaves that slot missing. Implementation failures, infrastructure
failures, and evaluated idea failures have separate counts. No sample is dropped
for a low score. Missing pairs do not become zero gains. An incomplete primary
matrix is reported as incomplete and cannot support the planned superiority claim.

Stop at the first approved resource limit, a safety failure, or owner cancellation.
Reconcile interrupted attempts and publish the stop reason. Do not extend a run
because its result is close to a threshold. No efficacy or futility stopping is
allowed inside a fixed confirmation or held-out batch.

### Seeds and verification

Study number N defines seed base `N * 1_000_000_000_000`. Loop seeds are base +
0–29 (T1 uses 0–2). Assign each campaign an index j in the frozen matrix and
seed base `base + (j + 1) * 100_000`. Relative to that campaign base, assign ranges:
calibration +10,000–10,009 at each fidelity, screen +20,000–20,299,
confirmation +30,000 onward, and held-out +40,000 onward. Freeze the exact lists
in the manifest. A new campaign receives a separate campaign identity and fresh
independent draws. If any workload cannot provide independent draws, do not
apply the seed test to it.

T0 and native T1 require five distinct-seed baseline calibration pairs at each
used validation fidelity. Use the current shared
[m4-pro-v1 recipe](src/openloop/loops/recipes/m4_pro_v1/README.md) for T1.
S1's legacy AdamW noise estimates and rank correlation cannot calibrate it.
Before a T1 multi-fidelity search, measure a frozen six-candidate panel at three
matched seeds and both fidelities. Report Spearman correlation; if it is below
0.5, block multi-fidelity allocation until an amendment defines another proxy.
Include this prerequisite work in the manifest and cost totals.

Use validation only for proposal, allocation, screening, and candidate selection.
Use the [decider](docs/DECIDER.md) for fresh-seed confirmation. Freeze the family
before its confirmation starts. For adaptive proposal sessions, close the full
proposal batch before confirmation; a later batch is a new, pre-budgeted campaign.
The minimum effect for T1 is 0.005 BPB. For T0 it is 0.1 times its declared
distinct-seed noise standard deviation. Larger effects require an amendment.

Allocate total family-wise error 0.05 across the five studies: 0.01 each. Within
each study reserve 0.004 for candidate confirmation, 0.004 for final held-out
claims, and 0.002 for primary treatment contrasts. Let C be the manifest's fixed
maximum number of campaigns, including all arms, seeds, tasks, and proposal
batches. Each campaign receives confirmation alpha `0.004 / C` and held-out
alpha `0.004 / C`. Confirmation divides its allocation by the full declared
family size M (`confirmation_scope="family"`). Unused error allocations are not
recycled. No protocol instance alone enforces the study-wide budget; the host
must check this accounting before registration.

For each confirmation, fix `max(5, ceil(log2(C * M / 0.004)))` independent paired
seeds. For each final held-out batch, fix
`max(5, ceil(log2(C / 0.004)))` independent paired seeds. These counts permit an
all-positive sign test to pass; they do not guarantee power. Use the one-sided
exact sign test against the minimum effect, with ties counted as non-wins.
Report all paired gains, mean, median, exact sign-test p-value, and the decider's
order-statistic interval at the allocated threshold. Do not describe a test of
median gain as a test of mean gain.

Close selection before releasing one held-out batch for its final candidate.
Use an untouched, disjoint held-out bundle for each adaptive campaign; a new
seed or file name does not make previously released documents untouched. The
manifest must show that the existing FineWeb-Edu held-out membership is suitable,
or identify a separately frozen new bundle without changing the existing one.
No held-out score enters proposer context, allocation, or repair. Full research
memory records remain trusted-host evidence; a validation-only projection may
enter a later campaign only with a fresh held-out bundle, as specified for S3.
S1's released ClimbMix held-out scores are not fresh S2–S6 evidence.

For each study, freeze the number K of primary treatment contrasts. Use two-sided
paired sign-flip permutation tests at `0.002 / K` (enumerate up to 20 blocks;
otherwise use 100,000 permutations with RNG seed N). Exchangeability of arm
labels is an assumption, not a property of benchmark training seeds. Report
2,000 paired-block bootstrap percentile 95% intervals with RNG seed N, labeled
descriptive. At three T1 loop seeds, report every session and range; do not claim
inferential superiority. All other metrics and correlations are descriptive.
Do not pool results across datasets, fidelities, architectures, or hardware.

### Recorded-table limits

[T4](docs/T4.md) exposes validation outcomes, not a fresh held-out training run.
NAS seed availability differs by fidelity. Freeze eligible seed lists and never
call equal numeric seeds at different fidelities a paired training stream.
HPO-B has one authoritative lookup seed (0); policy seeds change search order,
not training noise. Use only meta-validation tasks. Meta-test remains sealed.
Neither repeated table queries nor policy restarts supply independent training
replicates. T4 endpoints below are recorded validation regret and search cost.
They cannot establish a native, held-out verified improvement.

## S2 — Where should the next compute hour go?

Compare six arms: greedy with the noise-aware gate; greedy with the selectable
AutoScientists gate; UCB1 bandit; random search; successive halving; Hyperband.
The last four use allocator screening and the common confirmation layer where
independent evidence exists. The AutoScientists arm is openloop's selectable
gate plus confirmation, not a reproduction of the complete published system.

- T0: all six arms, 30 matched loop seeds, a fixed 128-candidate pool per seed.
- NAS-Bench-201: greedy, bandit, random, halving, and Hyperband; 30 matched policy
  seeds, cifar10-valid, cifar100, and ImageNet16-120, 128 eligible architectures
  per task/seed, 12/200 epochs.
- HPO-B: greedy, bandit, and random; 30 matched policy seeds on the first ten
  meta-validation tasks ordered by numeric
  `(search_space_id, dataset_id)`. If fewer exist, use all and record the count.
  Use at most 128 configurations per task. Omit halving and Hyperband because
  the adapter has one fidelity. Both T4 panels omit noise gates: their recorded
  seeds do not supply the fresh calibration and confirmation design above.
  Greedy is a recorded-validation control, without a verified-gain claim.
- T1: greedy/noise, greedy/AutoScientists, and bandit at three matched loop seeds;
  random at seed 0 is a descriptive pilot. Ceiling: 300 screen attempts/session,
  plus the separately reserved calibration, confirmation, and held-out batches.

Freeze candidate pools by sorting eligible identities on
SHA-256 of UTF-8 compact JSON `[study_id, task_id, loop_seed, candidate_hash]`
and taking 128 (or all if fewer exist). Use no JSON whitespace or ASCII escaping.
Eligibility uses schema and availability, never outcome value. For T1, generate
the finite legal candidate pool before any study scores. LLM arms use the same
proposer, prompts, and initial cards. The random and Hyperband baselines need no
LLM; show both execution cost and end-to-end cost including card generation.
Freeze UCB bounds from the declared metric domain; log clipping. Halving uses
eta 2 and the adapter's legal rung grid. Hyperband uses that same grid.

Each arm receives the same work ceiling within a task: the declared cost of
300 maximum-fidelity jobs, including all stages and failures. Stop admission
before that total is exceeded; lower-fidelity jobs consume their actual declared
work. Verification must fit inside this common ceiling. Report unspent balances.

Primary endpoint on T0/T1: work to the first held-out verified improvement,
censored at the common budget. Compare capped work (budget when no win), and
report success frequency separately. T4: area under best-observed validation
regret versus charged work, using each table's fixed eligible-pool optimum only
in trusted post-run analysis. Normalize by the eligible pool's outcome range;
constant tasks have undefined normalized regret and remain listed as such.
Primary contrasts compare bandit with every eligible baseline, per task.

Replay the frozen successful and failed traces at 1, 10, 100, 1,000, and 10,000
mock workers, with 30 scheduler seeds. Freeze the trace assignment rule before
replay; never condition draws on future outcomes. Report simulated makespan,
idle time, throughput, and budget exhaustion. Trace shortages are reported,
not extrapolated into native scaling measurements.

## S3 — Does the loop learn from verified history?

Compare memory on versus memory off. Use the S2 bandit settings in both arms,
the same frozen proposer and implementer, and the same initial candidate grammar.
Run 30 matched loop seeds on T0 and the S2 T4 tasks, plus three on T1. Permit
100 proposals/session in five batches of 20. Freeze all five campaign slots,
their error budgets, and work caps before the session. Each slot's realized
candidate identities must be committed and registered before confirmation.

Memory on reads only a validation-only projection of verified findings from
closed earlier batches. The host must first verify the full record through
`ResearchMemory`; the projection excludes held-out scores, intervals, seeds,
sample references, and any held-out-only change fields. Freeze and test that
projection before execution; v0 does not add memory to proposer context itself.
Memory off receives the same context budget and an empty findings
section. Both retain identical mechanical ledger/cache access. No arm receives
held-out measurements or another arm's results. Later batches use disjoint
untouched held-out bundles. Reset memory between loop seeds.
Keep the two-repair limit and the model settings identical across arms.

Primary contrast: memory on minus memory off for duplicate proposal fraction
(repeat canonical candidate identity / all proposals, including rejected
duplicates). Secondary: final verified gain per total process hour on T0/T1,
recorded validation regret on T4, token/API cost, idea failures, and implementation
failures. Report invalid cards separately; they do not disappear from the
proposal denominator. Freeze identity normalization before proposals start.
The current T4 adapter cannot produce the required verified findings. Its memory
ablation is blocked until a trusted independent verification path is frozen;
empty-memory replay alone is a negative control, not evidence of memory efficacy.

## S4 — Are we overfitting evaluation?

Use all complete S2/S3 lineages with at least five checkpoints. Hash the eligible
lineage list before opening audit data. Add 30 T0 loop seeds each for planted
no-overfit and overfit controls, with the change point at checkpoint 5. Freeze
their truth parameters in the manifest; keep them hidden from the detector.
The current quadratic T0 oracle has no planted overfit transition. Freeze and
contract-test a separate simulation fixture before this control panel runs;
do not relabel ordinary noise draws as an overfit intervention.

At every checkpoint, define gap as claimed validation gain minus audit gain,
both relative to the lineage root. The detector fires after two consecutive
checkpoints with gap above twice the root calibration noise. Freeze this rule;
do not tune it on these lineages. Primary contrast: detection frequency on
planted overfit versus null lineages. Report false-alert frequency and detection
delay from the planted change point, including never-detected cases.

For native lineages, freeze all checkpoints using validation first. Then conduct
one sealed, non-adaptive audit batch over the fixed checkpoint list on a new
audit bundle. This is a retrospective detection study. Report apparent lead time
and gain gaps descriptively; it does not show that an online alert improved
search. No checkpoint audit feeds a later proposal. Online held-out monitoring
would require a separate protocol with disjoint audit releases.

## S5 — Do architecture and data-mixture wins compose?

Blocked until T3 exists behind the loop contract with frozen corpus memberships,
mixture semantics, and a trusted evaluator. Use validation to select one T1
architecture win A and one T3 mixture win D. Both must pass independent
confirmation before this study's combination is selected. Record both parents.
If either is absent, report that prerequisite outcome and do not substitute a
post-hoc near-winner.

Freeze a 2×2 design: baseline, A only, D only, A+D. Keep training tokens,
tokenizer, evaluator, and environment fixed. Use the seed count required by the
common rule, paired across all four cells. Reserve three comparison slots:
A versus baseline, D versus baseline, A+D versus baseline. Selection closes
before one fixed held-out batch evaluates the four cells on untouched data.

Let gA, gD, and gAD be paired gains against baseline. Primary treatment contrast
is interaction `gAD - gA - gD`. Report its per-seed values and interval. Positive
interaction means synergy; negative means interference. Call the combination
an improvement only if its own multiplicity-controlled decision passes. An
interval crossing zero is inconclusive about additivity, not proof of stacking.

## S6 — Do small wins survive scale-up?

Blocked until the CUDA T1 executor/evaluator port and T2 exist and pass contract
checks. Freeze CUDA source and numerical settings before scored runs. Calibrate
noise separately on each backend and budget; do not treat MLX/CUDA differences
as a candidate's gain. This study needs separate owner approval and exact GPU
and API spending limits. Account credit is not permission to spend it.

Select the top three confirmed T1 candidates by mean full-fidelity validation
gain from completed S2/S3 (hash tie-break), plus baseline. Freeze that panel
before any scale-up outcome. If fewer than three qualify, report insufficient
candidates and block the planned rank analysis. Run each at 1×, 2×, and 4× the
21m token budget on CUDA, using the common fixed seed counts and matched
within-backend pairs. Budget multipliers apply to exact adapter token amounts.

Primary descriptive endpoint: Spearman rank correlation of candidate mean gains
at MLX 1× versus CUDA 2× and 4×, with a paired-seed bootstrap interval. Include
all selected candidates, even failed transfers; missing outcomes leave a rank
analysis incomplete. Report same-budget backend gaps and adjusted per-candidate
gain tests separately. Do not generalize a three-candidate correlation to all
architectures.

T2 is a separate portability panel: one frozen 0.5B arithmetic/Countdown recipe
on the Mac and a frozen 1.5B counterpart on CUDA. Fix model/tokenizer revisions,
GRPO settings, prompt membership, generation budget, pass-rate evaluator, and
disjoint evaluation tasks before execution. Use five training seeds per size.
Report pass rate, task-level uncertainty, failures, throughput, and cost
descriptively. Model size and backend are confounded; this panel cannot establish
that the platform or a T1 architecture causes a T2 quality gain.

## Researcher time and publication

For each study, freeze one researcher question, answer rubric, and equivalent
manual/platform evidence packets before analysis. Use three matched cases per
method with counterbalanced order; record participant and case IDs. Stop the
clock at a correct evidence-linked answer, or censor at 30 minutes. Report every
timing and errors descriptively. Existing UI responsiveness is not this metric.

Publish the frozen manifests, amendments, source hashes, analysis commands,
sanitized ledger exports, all attempted slots, costs, exclusions, negative
findings, and stopped studies. Exclude secrets and protected held-out content.
Report planned versus completed counts and every deviation. Label engineering
checks, simulations, exploratory analyses, and preregistered results separately.
An outcome-aware change requires a new study label; it cannot overwrite this
protocol or make earlier observations preregistered.
