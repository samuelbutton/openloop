"""Frozen decision inputs. Only a trusted evaluator may supply measurements.

No function here reads data or storage. Run references are evidence, not access
controls. A cache alias and an unfinished attempt cannot become a new sample.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType

from openloop.ledger import (
    Direction,
    ExperimentInputs,
    InvalidInputError,
    Metric,
    Run,
    Status,
    content_hash,
)
from openloop.ledger.validation import (
    check_digest,
    check_text,
    coerce_enum,
    finite_number,
)


class Policy(StrEnum):
    NOISE_AWARE = "noise_aware"
    AUTOSCIENTISTS = "autoscientists"


class ConfirmationScope(StrEnum):
    """How the Bonferroni family for confirmation is defined."""

    FAMILY = "family"
    SCREENED = "screened"


class Verdict(StrEnum):
    PROMOTE = "promote"
    DISCARD = "discard"
    INCONCLUSIVE = "inconclusive"
    VERIFIED = "verified"


class DecisionStage(StrEnum):
    SCREEN = "screen"
    CONFIRM = "confirm"
    HELD_OUT = "held_out"


class NoiseKind(StrEnum):
    DISTINCT_SEED = "distinct_seed"
    FIXED_SEED = "fixed_seed"


@dataclass(frozen=True)
class PolicyRule:
    """Fixed-count requirements that a screen policy places on a protocol."""

    screen_seeds: int
    calibration_pairs: int

    def __post_init__(self) -> None:
        if type(self.screen_seeds) is not int or self.screen_seeds < 1:
            raise InvalidInputError("A policy needs at least one screen seed")
        if type(self.calibration_pairs) is not int or not (
            3 <= self.calibration_pairs <= 5
        ):
            raise InvalidInputError("A policy needs three to five calibration pairs")


POLICY_RULES: Mapping[Policy, PolicyRule] = MappingProxyType(
    {
        Policy.NOISE_AWARE: PolicyRule(screen_seeds=2, calibration_pairs=5),
        Policy.AUTOSCIENTISTS: PolicyRule(screen_seeds=2, calibration_pairs=5),
    }
)

DEFAULT_SCREEN_MARGIN_SIGMAS = 0.5


@dataclass(frozen=True)
class Sample:
    """One trusted score per attempt, regardless of metric sample_count."""

    reference: str
    attempt_id: str
    inputs: ExperimentInputs
    metric: Metric
    metric_name: str = field(kw_only=True)

    def __post_init__(self) -> None:
        check_text(self.reference)
        check_text(self.attempt_id)
        check_text(self.metric_name)
        if not isinstance(self.inputs, ExperimentInputs) or not isinstance(
            self.metric, Metric
        ):
            raise InvalidInputError("Samples require typed inputs and metrics")

    @classmethod
    def from_run(cls, run: Run, metric_name: str) -> "Sample":
        """Accept a completed execution, never a cache alias or failure."""
        if run.status is not Status.SUCCEEDED or run.result is None:
            raise InvalidInputError("Only successful scored attempts are evidence")
        if run.reused_from is not None:
            raise InvalidInputError("A cache alias is not an independent sample")
        if metric_name not in run.result.metrics:
            raise InvalidInputError("Required decision metric is missing")
        return cls(
            run.id,
            run.attempt_id,
            run.inputs,
            run.result.metrics[metric_name],
            metric_name=metric_name,
        )

    @property
    def context_hash(self) -> str:
        """Identify comparison conditions, excluding candidate code and seed."""
        return content_hash(
            (
                replace(
                    self.inputs,
                    source_hash="0" * 64,
                    dependencies_hash="0" * 64,
                    config={},
                    seed=0,
                ),
                self.metric.unit,
                self.metric_name,
                self.metric.direction,
                self.metric.split,
                self.metric.sample_count,
            )
        )

    @property
    def score(self) -> float:
        """Orient the metric so that a higher score is better."""
        return (
            -self.metric.value
            if self.metric.direction is Direction.MINIMIZE
            else self.metric.value
        )


@dataclass(frozen=True)
class Pair:
    """Matched baseline and candidate runs at one declared seed."""

    baseline: Sample
    candidate: Sample

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, Sample) or not isinstance(
            self.candidate, Sample
        ):
            raise InvalidInputError("Comparisons require typed samples")
        if self.baseline.context_hash != self.candidate.context_hash:
            raise InvalidInputError("Comparison data, evaluator or work budget differs")
        if self.baseline.inputs.seed != self.candidate.inputs.seed:
            raise InvalidInputError("Comparison seeds differ")
        if self.baseline.attempt_id == self.candidate.attempt_id:
            raise InvalidInputError("A comparison requires two distinct attempts")

    @property
    def gain(self) -> float:
        """Orient improvement upwards for either metric direction."""
        return finite_number(self.candidate.score - self.baseline.score, "Gain")


def unique_samples(samples: tuple[Sample, ...]) -> None:
    """Reject repeated executions, aliases, or scores for one input."""
    for identities in (
        [item.attempt_id for item in samples],
        [item.reference for item in samples],
        [item.inputs.hash for item in samples],
    ):
        if len(set(identities)) != len(identities):
            raise InvalidInputError("Repeated evidence is not independent")


@dataclass(frozen=True)
class StagePlan:
    """Frozen baseline template, split, metric and ordered seed schedule."""

    template: ExperimentInputs
    metric: Metric
    seeds: tuple[int, ...]
    metric_name: str = field(kw_only=True)

    def __post_init__(self) -> None:
        check_text(self.metric_name)
        if not isinstance(self.template, ExperimentInputs) or not isinstance(
            self.metric, Metric
        ):
            raise InvalidInputError("Stage plans require typed inputs and metrics")
        seeds = tuple(self.seeds)
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise InvalidInputError("A stage requires nonnegative integer seeds")
        if len(set(seeds)) != len(seeds):
            raise InvalidInputError("A stage requires distinct seeds")
        object.__setattr__(self, "seeds", seeds)

    @property
    def context_hash(self) -> str:
        return Sample(
            "template",
            "template",
            self.template,
            self.metric,
            metric_name=self.metric_name,
        ).context_hash

    def validate(self, pairs: tuple[Pair, ...], candidate: str) -> None:
        """Accept only an ordered prefix, with the frozen baseline and context."""
        if len(pairs) > len(self.seeds):
            raise InvalidInputError("The declared sample count was exceeded")
        unique_samples(
            tuple(item for pair in pairs for item in (pair.baseline, pair.candidate))
        )
        for pair, seed in zip(pairs, self.seeds, strict=False):
            if (
                pair.baseline.inputs.candidate_hash != self.template.candidate_hash
                or pair.candidate.inputs.candidate_hash != candidate
                or pair.baseline.context_hash != self.context_hash
                or pair.baseline.inputs.seed != seed
            ):
                raise InvalidInputError("Evidence does not match the frozen stage")


@dataclass(frozen=True)
class Protocol:
    """A finite family fixed before calibration or candidate comparisons.

    Confirmation tests median paired gain at a Bonferroni threshold: alpha divided
    by the family size (FAMILY) or by the fixed screened set size (SCREENED). The
    held-out test checks one selected candidate at plain alpha.
    Seeds must represent independent draws and must not be tuned after scores.
    """

    candidates: tuple[str, ...]
    screen: StagePlan
    confirm: StagePlan
    held_out: StagePlan
    screen_noise_seeds: tuple[int, ...]
    confirm_noise_seeds: tuple[int, ...]
    policy: Policy = Policy.NOISE_AWARE
    alpha: float = 0.05
    minimum_effect: float = 0.0
    screen_margin_sigmas: float = DEFAULT_SCREEN_MARGIN_SIGMAS
    confirmation_scope: ConfirmationScope = ConfirmationScope.FAMILY
    confirm_top_k: int | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(plan, StagePlan)
            for plan in (self.screen, self.confirm, self.held_out)
        ):
            raise InvalidInputError("Protocols require typed stage plans")
        candidates = tuple(self.candidates)
        if not candidates or len(set(candidates)) != len(candidates):
            raise InvalidInputError("Declare a nonempty, unique comparison family")
        for candidate in candidates:
            check_digest(candidate)
            if candidate == self.screen.template.candidate_hash:
                raise InvalidInputError("The baseline is not a candidate improvement")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "policy", coerce_enum(Policy, self.policy, "policy"))
        alpha = finite_number(self.alpha, "Alpha")
        effect = finite_number(self.minimum_effect, "Minimum effect")
        if not 0 < alpha < 1 or effect < 0:
            raise InvalidInputError("Require 0 < alpha < 1 and minimum effect >= 0")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "minimum_effect", effect)
        margin = finite_number(self.screen_margin_sigmas, "Screen margin")
        if margin < 0:
            raise InvalidInputError("Screen margin must be nonnegative")
        if (
            self.policy is Policy.AUTOSCIENTISTS
            and margin != DEFAULT_SCREEN_MARGIN_SIGMAS
        ):
            raise InvalidInputError(
                "The AutoScientists policy keeps its published 2-sigma gate"
            )
        object.__setattr__(self, "screen_margin_sigmas", margin)
        scope = coerce_enum(
            ConfirmationScope, self.confirmation_scope, "confirmation scope"
        )
        object.__setattr__(self, "confirmation_scope", scope)
        if self.confirm_top_k is not None:
            if scope is not ConfirmationScope.SCREENED:
                raise InvalidInputError("A top-k limit requires the screened scope")
            if type(self.confirm_top_k) is not int or self.confirm_top_k < 1:
                raise InvalidInputError("A top-k limit must be a positive integer")
        rule = POLICY_RULES[self.policy]
        all_seeds: list[int] = []
        for stage, plan in (
            (DecisionStage.SCREEN, self.screen),
            (DecisionStage.CONFIRM, self.confirm),
            (DecisionStage.HELD_OUT, self.held_out),
        ):
            if (
                plan.template.phase != stage.value
                or plan.template.candidate_hash != self.screen.template.candidate_hash
            ):
                raise InvalidInputError(
                    "Stages require the same baseline and declared phases"
                )
            if plan.metric.split != (
                "held_out" if stage is DecisionStage.HELD_OUT else "validation"
            ):
                raise InvalidInputError(
                    "Selection must use validation; final scores use held_out"
                )
            if (plan.metric_name, plan.metric.unit, plan.metric.direction) != (
                self.screen.metric_name,
                self.screen.metric.unit,
                self.screen.metric.direction,
            ):
                raise InvalidInputError("Stages must measure the same metric")
            all_seeds.extend(plan.seeds)
        for name in ("screen_noise_seeds", "confirm_noise_seeds"):
            seeds = tuple(getattr(self, name))
            if len(seeds) != 2 * rule.calibration_pairs or any(
                type(seed) is not int or seed < 0 for seed in seeds
            ):
                raise InvalidInputError(
                    "Noise calibration seeds must match the policy's pair count"
                )
            object.__setattr__(self, name, seeds)
            all_seeds.extend(seeds)
        if len(set(all_seeds)) != len(all_seeds):
            raise InvalidInputError("Calibration and stage seeds must be disjoint")
        if len(self.screen.seeds) != rule.screen_seeds:
            raise InvalidInputError("Screen seed count must match the policy")
        if 2 ** -len(self.confirm.seeds) > self.test_alpha(self.max_confirmations):
            raise InvalidInputError("Confirmation needs enough seeds to reject a null")
        if 2 ** -len(self.held_out.seeds) > self.alpha:
            raise InvalidInputError("Held-out needs enough seeds to reject a null")
        if (
            self.held_out.template.data_hash == self.confirm.template.data_hash
            or self.held_out.template.data_hash == self.screen.template.data_hash
        ):
            raise InvalidInputError("Held-out data must differ from selection data")
        if (self.held_out.template.fidelity, self.held_out.template.budget_amount) != (
            self.confirm.template.fidelity,
            self.confirm.template.budget_amount,
        ):
            raise InvalidInputError(
                "Held-out and confirmation must use the same fidelity"
            )
        if self.confirm.template.budget_amount < self.screen.template.budget_amount:
            raise InvalidInputError(
                "Confirmation work must not be less than screen work"
            )

    @property
    def max_confirmations(self) -> int:
        """Largest possible confirmation set, which gives the strictest threshold."""
        size = len(self.candidates)
        if self.confirm_top_k is None:
            return size
        return min(size, self.confirm_top_k)

    def test_alpha(self, tests: int) -> float:
        """Bonferroni threshold for a confirmation set of the given size."""
        if type(tests) is not int or tests < 1:
            raise InvalidInputError("A multiple-test threshold needs one or more tests")
        return self.alpha / tests

    @property
    def hash(self) -> str:
        return content_hash({"decider_version": 1, "protocol": self})


@dataclass(frozen=True)
class Decision:
    """Immutable verdict and evidence, suitable for an external campaign record."""

    protocol_hash: str
    candidate_hash: str
    stage: DecisionStage
    verdict: Verdict
    reason: str
    evidence: tuple[str, ...]
    sample_count: int
    median_gain: float | None = None
    interval: tuple[float | None, float | None] = (None, None)
    p_value: float | None = None
    noise_sigma: float | None = None
    test_alpha: float | None = None

    def __post_init__(self) -> None:
        check_digest(self.protocol_hash)
        check_digest(self.candidate_hash)
        check_text(self.reason)
        object.__setattr__(
            self, "stage", coerce_enum(DecisionStage, self.stage, "stage")
        )
        object.__setattr__(
            self, "verdict", coerce_enum(Verdict, self.verdict, "verdict")
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "interval", tuple(self.interval))
        if len(self.interval) != 2:
            raise InvalidInputError("An interval requires two endpoints")
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise InvalidInputError("Sample count must be nonnegative")
        for reference in self.evidence:
            check_text(reference)
        for value in (
            self.median_gain,
            self.p_value,
            self.noise_sigma,
            self.test_alpha,
            *self.interval,
        ):
            if value is not None:
                finite_number(value, "Decision statistic")
        if self.p_value is not None and not 0 <= self.p_value <= 1:
            raise InvalidInputError("P-value must be in [0, 1]")
        if self.test_alpha is not None and not 0 < self.test_alpha <= 1:
            raise InvalidInputError("Test threshold must be in (0, 1]")
        if self.noise_sigma is not None and self.noise_sigma < 0:
            raise InvalidInputError("Noise must be nonnegative")
        lower, upper = self.interval
        if lower is not None and upper is not None and lower > upper:
            raise InvalidInputError("Interval endpoints must be ordered")
