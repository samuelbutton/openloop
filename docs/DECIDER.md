# Decision policies

Implemented on October 10, 2026. Supports [N2](../NEEDS.md#n1n10).
The code is in [`openloop.decider`](../src/openloop/decider/).
The decider uses the Python standard library.
It does not run training jobs or read data files.

## Frozen protocol

Create a `Protocol` before calibration and comparison.
Declare the baseline, candidate hashes, metric, minimum effect, alpha, and seed schedules.
The protocol also declares these rule fields:

| Field | Meaning | Default |
| --- | --- | --- |
| `policy` | The screen policy. | `noise_aware` |
| `screen_margin_sigmas` | The `noise_aware` screen needs a mean gain above this multiple of per-run noise. Finite and at least zero. The `autoscientists` policy rejects any other value. | 0.5 |
| `confirmation_scope` | `family` or `screened`. See [false discovery control](#false-discovery-control-and-effect-size). | `family` |
| `confirm_top_k` | Limit the screened confirmation set to the best k candidates. Valid only with `screened`. | none |

Each policy owns the fixed counts that the protocol checks.
`POLICY_RULES` maps each `Policy` to a `PolicyRule` with `screen_seeds` and `calibration_pairs`.
Both current policies use 2 screen seeds and 5 calibration pairs (10 seeds).
A new policy adds one entry to that mapping.
Every field above enters `Protocol.hash`.
Each `StagePlan` contains the baseline input template, metric metadata, and ordered seeds.
The template fixes the data, evaluator, environment, fidelity, and work budget.
`Protocol.hash` identifies the complete rule with decider version 1.
This version does not change ledger input identity or SQLite schema.

The family is finite and fixed.
It must not grow after confirmation scores become available.
New proposals need a new protocol and fresh confirmation evidence.
Repeated searches do not share this protocol's error bound.

Calibration, screen, confirmation, and held-out seeds are disjoint.
Seeds must select independent draws from the declared experiment distribution.
Do not select seeds from observed scores.
The held-out data hash must differ from both selection data hashes.
Different hashes alone do not prove that data memberships are disjoint.
The trusted host must check that condition.

## Noise floor

`Calibration` accepts three to five same-code measurement pairs.
Its per-run estimate is `sqrt(sum((score1 - score2)**2) / (2 * pair_count))`.
This is the estimator in [AutoScientists Appendix A.6][autoscientists].
Five pairs lock the estimate.
Campaigns require five baseline pairs at each validation fidelity before screening.
The protocol declares all ten seeds for each calibration.
Noise estimates and their evidence references enter each decision.

`distinct_seed` measures seed variation.
`fixed_seed` measures execution variation at identical inputs.
The latter cannot replace distinct-seed calibration in a campaign.
A deterministic T0 repeat has zero fixed-seed variation.
It still has nonzero distinct-seed variation.
The baseline estimate is a screening reference, not a bound on every candidate's variance.

## Selectable policies

| Policy | Screen rule | Confirmation |
| --- | --- | --- |
| `noise_aware` | The mean of two paired gains must exceed both the minimum effect and `screen_margin_sigmas` times the per-run noise estimate. | Fresh fixed-count paired sign test. |
| `autoscientists` | Apply the published 2-sigma gate below. | The same fresh fixed-count paired sign test. |

The `noise_aware` screen only saves compute.
Confirmation carries the error bound.
A strict margin discards real gains, because the mean of two gains has noise near one sigma.
The default of 0.5 sigma lets a null candidate pass the screen with probability about `1 - Phi(margin)` when the noise is Gaussian.
A paired gain has variance twice the per-run variance, so a mean of two gains has standard deviation near sigma.
A lower margin raises power, and it raises the number of null candidates that need confirmation.
A margin of zero leaves only the minimum effect.
The cost is confirmation compute, never the error bound.

Orient each metric so that higher scores are better.
The published gate compares a candidate with one frozen champion score.
For gain `delta` and per-run noise `sigma`:

1. If `delta > 2 * sigma`, promote the screen directly.
2. If `0 < delta <= 2 * sigma`, run the candidate at a second seed. Both scores must strictly beat the original champion score.
3. If `delta <= 0`, discard the screen.

`autoscientists_gate` exposes this rule without the confirmation wrapper.
Equality at `2 * sigma` enters the second-seed branch.
A missing second score returns `inconclusive` in that branch.
This follows the published gate in [Appendix A.6][autoscientists].
The paper learns noise during search from at least three pairs and locks it at five pairs.
Openloop instead requires five baseline pairs before search.
This meets the project's rule to measure noise first.
Openloop also adds fresh-seed confirmation and held-out evaluation.
These additions are not claims about the paper's method.

## False discovery control and effect size

Confirmation compares the candidate and baseline at each declared seed.
The data, metric, evaluator, environment, and work budget must match within each pair.
Code and configuration may differ.
Each seed pair counts once.
Inner observations in `Metric.sample_count` do not add independent seeds.

The tested effect is the median paired gain.
The null is `P(gain > minimum_effect) <= 1/2`.
The one-sided exact [sign test][sign-test] uses a binomial upper tail.
Ties count as non-wins and remain in the sample count.
This is a conservative choice.
It is not a test of arbitrary mean gain.

The threshold is `alpha / m`, where `m` is the number of tests in the confirmation set.
This [Bonferroni rule][bonferroni] bounds the chance of any false confirmation by alpha.
It therefore also bounds the false discovery rate by alpha.
Dependence between different candidates is permitted.
Each candidate's seed pairs must be independent under its declared null.
The set and all confirmation seed counts must be fixed before their scores are seen.
Screening and confirmation must use independent evidence.
Do not stop early or reuse confirmation scores to tune a candidate.

`confirmation_scope` defines `m`:

| Scope | Confirmation set | Threshold |
| --- | --- | --- |
| `family` | The whole declared family. Candidates confirm one at a time, in any order. | `alpha / family size` |
| `screened` | The candidates that the screen promoted, fixed once. | `alpha / set size`, frozen with the set |

In `screened` scope the campaign refuses every confirmation until each family member has a resolved screen.
It then fixes the set once, from screen decisions only.
If `confirm_top_k` is set, the set is the k promoted candidates with the highest mean screen gain.
Ties break by ascending candidate hash.
Each promoted candidate outside the set receives a `discard` decision in the confirmation stage.
Its reason states that it was outside the top k and not tested.
That decision has no p-value, no seeds, and no threshold.
Exactly the fixed set is tested, so the set size `m` does not depend on any confirmation score.
This is valid because the set depends only on screen evidence.
Confirmation evidence is fresh and does not exist when the set is fixed.
The protocol check for enough confirmation seeds uses the largest possible set: the family size, or `confirm_top_k` if it is smaller.
An empty set is valid.
Then `close_selection()` with no candidate closes selection with no winner, and no held-out evaluation follows.
That call fails when any candidate is confirmed.

Incomplete confirmation returns `inconclusive`, with no p-value.
A full batch returns `promote` or `discard`.
The decision records median gain, an exact two-sided order-statistic interval, p-value, seed-pair count, and evidence references.
Intervals use the same threshold as the test.
Each decision records that threshold as `test_alpha`; a screen decision has none.
An absent interval endpoint is unbounded.
The sign test is discrete, so its decision and the two-sided interval need not cross their thresholds together.

The raw AutoScientists screen has no repeated-search error bound.
Only the common confirmation layer supplies the bound above.

## Ledger and campaign use

Convert a scored execution with `Sample.from_run(run, metric_name)`.
Unfinished attempts, failures, missing metrics, and cache aliases are rejected.
For a cache alias, resolve its original scored run through the ledger.
An original baseline sample may be shared across candidate comparisons.
It cannot count twice within one comparison or calibration.

```python
from openloop.decider import Campaign, DecisionStage

# protocol and calibrations are frozen records; pairs come from trusted scores.
campaign = Campaign(protocol, screen_noise, confirm_noise)
screen = campaign.evaluate(DecisionStage.SCREEN, candidate_hash, screen_pairs)
# If the screen promotes, provide the declared fresh confirmation batch.
confirm = campaign.evaluate(DecisionStage.CONFIRM, candidate_hash, confirm_pairs)
# Resolve all family members before closing selection.
campaign.close_selection(candidate_hash)
final = await campaign.finalize(trusted_final_evaluator)
```

Evidence can extend an incomplete ordered prefix.
It cannot replace that prefix or revise a completed stage.
The baseline evidence remains fixed across the family.
Failures stay in the execution ledger; they are not negative statistical samples.

`finalize` calls the trusted evaluator only after selection closes.
It passes the selected candidate and held-out plan.
The evaluator must return the complete fixed-count batch.
The final sign test uses plain alpha, not `alpha / m`.
It tests one candidate that was fixed before any held-out score exists.
Selection used validation evidence that is independent of held-out data.
A single pre-specified test needs no multiplicity correction.
With 16 held-out seeds at alpha 0.05, 12 wins pass and 11 do not.
The protocol requires enough held-out seeds to reject a null at alpha.
The decision records the threshold used.
Only a passing final test returns `verified`.
The selected candidate cannot change after a failing final test.
No intermediate held-out scores enter validation decisions.
Permission is consumed before the evaluator starts.
Failure, cancellation, and concurrent calls cannot reopen it in that campaign instance.

The coordinator is in memory.
It is not a file access boundary or a durable campaign store.
The trusted host must isolate held-out files from candidate workers and proposers.
It must prevent creation of a new campaign over the same used held-out set.
Restricted campaign records and recovery across coordinator crashes remain planned.
Do not claim durable one-use access from this API alone.
Neither these pure functions nor Python object privacy protect against a hostile host.

## T0 validation

A decider needs two numbers.
The first is the false discovery rate under nulls.
The second is the detection rate across effect sizes.
A study that plants a gain of ten sigma shows neither: any rule detects it.
The T0 study in [`openloop.studies`](../src/openloop/studies/) measures both.
It lives outside the decider package and uses only the public decider API.

Run the CPU-only study (about 4 minutes on the development Mac):

```sh
uv run python -m openloop.studies.t0_decider --trials 384 --output docs/decider-t0.json
```

Flags: `--trials` (per arm and effect size), `--effects` (comma-separated, default `0,0.25,0.5,1,2,3`), `--family-size` (3 to 23, default 8), and `--output`.
The same flags give byte-identical JSON.
The [recorded report](decider-t0.json) is marked `"simulated": true`.
It records the flags, the decider version, the T0 loop hash, and one protocol hash per arm and effect size.
It uses 384 trials per arm and effect size, so about 2,300 simulated families per arm.
No paid API, training job, or production ledger is used.

### Method

Each trial is an independent family of eight candidates and a frozen baseline at loss 1.
An effect size `delta` is the planted gain in units of the per-run noise sigma at screen fidelity (0.1).
Confirmation and held-out fidelity have half that noise (0.05).
Effect size 0 is the all-null family: eight candidates with loss exactly 1.
A larger `delta` plants one true gain of `delta` sigma, one worsening of 1 sigma, and six nulls.
Null coordinates are unit vectors and sign patterns of 0.5, so their losses are exactly 1 with no rounding.
The planted loss is the square of a square root, which carries one rounding step near 1e-16.
That step is far below the gains studied, and it does not touch any null.

Selection scores use the real T0 `run` and `evaluate` methods.
Held-out scores use a separate trusted synthetic oracle with a separate data hash.
The oracle draws independent noise from the complete final input identity.
The study selects the confirmed candidate with the largest median confirmation gain, then runs held-out once.
Every arm sees the same trials and the same scores, so arm comparisons are paired.
Calibration uses five seed pairs per fidelity.
The screen, confirmation, and held-out stages use 2, 16, and 16 seeds.
Alpha is 0.05 and the minimum effect is zero.

| Arm | Screen | Confirmation threshold |
| --- | --- | --- |
| `noise_aware_margin_0.5_family` | Mean gain above 0.5 sigma | alpha / 8 |
| `noise_aware_margin_2.0_family` | Mean gain above 2 sigma (the earlier default) | alpha / 8 |
| `noise_aware_margin_0.5_screened` | Mean gain above 0.5 sigma | alpha / size of the screened set, no top-k |
| `autoscientists_family` | Published 2-sigma gate | alpha / 8 |
| `raw_autoscientists_screen` | Published 2-sigma gate | None. The report counts promotions only. |

### Detection

Detection is the fraction of trials in which the planted gain was selected and verified on held-out.
Cells show the rate and its 95% Wilson interval, over 384 trials.

| Arm | 0.25 sigma | 0.5 sigma | 1 sigma | 2 sigma | 3 sigma |
| --- | --- | --- | --- | --- | --- |
| `noise_aware`, margin 0.5, family | 0.5% (0.1-1.9) | 8.1% (5.7-11.2) | 58.3% (53.3-63.2) | 92.2% (89.1-94.5) | 99.2% (97.7-99.7) |
| `noise_aware`, margin 0.5, screened | 0.8% (0.3-2.3) | 13.0% (10.0-16.8) | 62.8% (57.8-67.4) | 92.2% (89.1-94.5) | 99.2% (97.7-99.7) |
| `noise_aware`, margin 2.0, family | 0.3% (0.0-1.5) | 2.1% (1.1-4.1) | 20.1% (16.4-24.3) | 53.4% (48.4-58.3) | 84.6% (80.7-87.9) |
| `autoscientists` plus confirmation | 0.8% (0.3-2.3) | 8.6% (6.2-11.8) | 54.4% (49.4-59.3) | 87.8% (84.1-90.7) | 97.9% (95.9-98.9) |
| Raw AutoScientists screen promotes the gain | 48.2% (43.2-53.2) | 55.5% (50.5-60.4) | 64.1% (59.1-68.7) | 87.8% (84.1-90.7) | 97.9% (95.9-98.9) |

The last row is a screen promotion, not a verified result.
The report also records the confirmation-stage rate (`planted_confirmed`).
Held-out loses power at small effects because it is a second 16-seed sign test.
At 0.5 sigma, `noise_aware` margin 0.5 family confirms the gain in 11.5% of trials and verifies it in 8.1%.

### Error rates

The null columns come from the all-null family (effect size 0).
The report repeats the same counts at every other effect size, for the six nulls and the worsening.
Those repeats share the same noise draws across effect sizes, so they are not independent evidence.

| Arm | Any false confirmation, all-null family | Wrong candidate verified on held-out, all-null family | Worsening confirmed, all effect sizes | Raw screen: families with any false promotion |
| --- | --- | --- | --- | --- |
| `noise_aware`, margin 0.5, family | 1.0% (0.4-2.6) | 0.0% (0.0-1.0) | 0 / 1920 | |
| `noise_aware`, margin 0.5, screened | 2.1% (1.1-4.1) | 0.0% (0.0-1.0) | 0 / 1920 | |
| `noise_aware`, margin 2.0, family | 0.0% (0.0-1.0) | 0.0% (0.0-1.0) | 0 / 1920 | |
| `autoscientists` plus confirmation | 1.3% (0.6-3.0) | 0.0% (0.0-1.0) | 0 / 1920 | |
| Raw AutoScientists screen | | | | 74.0% (69.3-78.1) |

No arm verified a wrong candidate in any of the 2,304 families per arm.
Every false-confirmation rate has an upper interval limit below alpha of 5%.
The Bonferroni bound is conservative here, because the discrete sign test cannot use all of its threshold.
The raw screen alone has no error bound: it promotes at least one null in 70% to 74% of families.

### Compute cost

The cost proxy is the mean number of confirmation batches run per family.
Each batch is 16 seed pairs, so each one costs 32 confirmation-fidelity runs.

| Arm | Null family | 1 sigma | 3 sigma |
| --- | --- | --- | --- |
| `noise_aware`, margin 0.5 (family or screened) | 2.67 | 2.77 | 3.09 |
| `noise_aware`, margin 2.0 | 0.50 | 0.62 | 1.23 |
| `autoscientists` plus confirmation | 3.02 | 3.04 | 3.38 |

The screened scope runs the same confirmations as the family scope.
It tests them at a larger threshold, which gains power at no extra compute.
The 2-sigma margin saves about four fifths of the confirmations, and loses most of the power at 1 sigma.
Mean noise estimates were 0.0933 at screen fidelity and 0.0489 at confirmation fidelity.
The true values are 0.1 and 0.05.
The estimate is a sample standard deviation from five pairs, which is biased low on average.
This lowers the screen threshold slightly.

### What the numbers imply

The error side holds in this simulation: no arm confirmed more than 2.1% of all-null families, and none verified a wrong candidate.
The power side is the limit.
T1 noise is near 0.03 BPB, and realistic gains are near 0.5 sigma.
With this schedule (2 screen seeds, 16 confirmation seeds, 16 held-out seeds, family of eight), a true 0.5 sigma gain is verified in about 8% to 13% of campaigns.
A 1 sigma gain is verified in 54% to 63% of campaigns, and a 2 sigma gain in about 90%.
A 2-sigma screen margin cuts 0.5 sigma detection to 2%.
The margin of 0.5 is better, and the screened scope is better still, but no arm reliably detects 0.5 sigma.
A campaign that finds nothing at 0.5 sigma is therefore weak evidence of no effect.
Detecting such gains needs more confirmation and held-out seeds or a lower-noise fidelity.
This study does not say how many are enough; rerun it with the new schedule to find out.

An independent coordinator simulation with known sigma and the earlier held-out threshold gave similar numbers.
For the 2.0 margin it gave 0.2%, 1.5%, 14%, 50%, and 84% at 0.25 to 3 sigma, against 0.3%, 2.1%, 20%, 53%, and 85% here.
For `autoscientists` plus confirmation it gave 1.7%, 10.9%, 57%, 88%, and 97%, against 2.1%, 12.8%, 55%, 88%, and 98% at the confirmation stage here.
The differences are within or near the intervals, except the 2.0 margin at 1 sigma.
The likely cause is that this study estimates sigma from five pairs, and the estimate is low on average.
That lowers the screen threshold and raises early passes.
This explanation was not tested separately.

### Limits

- Noise is Gaussian, homoscedastic, and independent across seeds. Real training noise may be heavier-tailed and may depend on the candidate.
- Null candidates have exactly zero effect. Real near-null candidates may have small true gains or losses.
- Screen, confirmation, and held-out fidelity share one truth. The study has no fidelity bias, rank change, or data shift between them.
- The held-out oracle is synthetic. It does not test a real held-out corpus or T1 access controls.
- Trials differ only by seed and environment hash. The family has one planted gain, so it does not cover several real gains, or a mix of effect sizes.
- A finite simulation does not prove the error bound. The bound follows from the tests and assumptions above. The simulation only checks that the implementation and the assumptions agree.
- The 95% intervals cover sampling error across trials, not error from the model assumptions.
- The study fixes the selection rule (largest median confirmation gain) and the 1 sigma worsening size. Other choices were not tested.

[autoscientists]: https://arxiv.org/html/2605.28655#A6
[sign-test]: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm
[bonferroni]: https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm
