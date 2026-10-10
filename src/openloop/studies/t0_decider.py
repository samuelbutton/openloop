"""CPU-only T0 power and error study for the decider. All evidence is simulated.

Each trial plants one true gain of `delta` screen-noise sigmas, one worsening,
and exact nulls. Every arm sees the same scores (a paired comparison). Selection
scores use the real T0 `run` and `evaluate`. Held-out scores come from a
separate trusted synthetic oracle. Nothing here proves a bound: the bound
follows from the tests in `openloop.decider`, and this study measures it.
"""

import argparse
import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from openloop.decider import (
    Campaign,
    ConfirmationScope,
    DecisionStage,
    Policy,
    Verdict,
    autoscientists_gate,
)
from openloop.ledger import InvalidInputError, JSONValue
from openloop.ledger.validation import check_positive

from .t0_family import (
    BASELINE_LOSS,
    MAX_FAMILY_SIZE,
    SCREEN_NOISE_SIGMA,
    Arm,
    T0Study,
)

DEFAULT_EFFECTS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
DEFAULT_TRIALS = 128
WORSENING_SIGMAS = 1.0
WILSON_Z = 1.959964  # Two-sided 95% normal quantile.
ARMS = (
    Arm("noise_aware_margin_0.5_family"),
    Arm("noise_aware_margin_2.0_family", screen_margin_sigmas=2.0),
    Arm(
        "noise_aware_margin_0.5_screened",
        confirmation_scope=ConfirmationScope.SCREENED,
    ),
    Arm("autoscientists_family", policy=Policy.AUTOSCIENTISTS),
)
RAW_ARM = "raw_autoscientists_screen"


def wilson_interval(count: int, trials: int) -> tuple[float, float]:
    """Return the 95% Wilson score interval for a binomial proportion."""
    if trials < 1 or not 0 <= count <= trials:
        raise InvalidInputError("A proportion needs 0 <= count <= trials, trials >= 1")
    rate = count / trials
    z2 = WILSON_Z**2
    centre = (rate + z2 / (2 * trials)) / (1 + z2 / trials)
    half = (
        WILSON_Z
        * math.sqrt(rate * (1 - rate) / trials + z2 / (4 * trials**2))
        / (1 + z2 / trials)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(count: int, trials: int) -> Mapping[str, JSONValue]:
    """Report a count as a rate with its Wilson interval and trial count."""
    low, high = wilson_interval(count, trials)
    return {
        "count": count,
        "trials": trials,
        "rate": count / trials,
        "wilson95": [low, high],
    }


@dataclass(frozen=True)
class Outcome:
    """What one arm did on one simulated family."""

    confirmed: tuple[str, ...]
    confirmations_run: int
    selected: str | None
    verified: bool
    screen_sigma: float
    confirm_sigma: float


async def run_arm(study: T0Study, arm: Arm) -> Outcome:
    """Run one arm through screen, confirmation, selection, and held-out."""
    protocol = study.protocol_for(arm)
    low = await study.calibration(DecisionStage.SCREEN)
    high = await study.calibration(DecisionStage.CONFIRM)
    campaign = Campaign(protocol, low, high)
    promoted = [
        candidate
        for candidate in study.configs
        if campaign.evaluate(
            DecisionStage.SCREEN,
            candidate,
            await study.pairs(candidate, protocol.screen),
        ).verdict
        is Verdict.PROMOTE
    ]
    confirmed: dict[str, float] = {}
    for candidate in promoted:
        decision = campaign.evaluate(
            DecisionStage.CONFIRM,
            candidate,
            await study.pairs(candidate, protocol.confirm),
        )
        if decision.verdict is Verdict.PROMOTE and decision.median_gain is not None:
            confirmed[candidate] = decision.median_gain
    # Select the largest confirmed median gain: a rule of validation evidence only.
    selected = min(confirmed, key=lambda item: (-confirmed[item], item), default=None)
    campaign.close_selection(selected)
    verified = False
    if selected is not None:
        final = await campaign.finalize(study.pairs)
        verified = final.verdict is Verdict.VERIFIED
    return Outcome(
        tuple(confirmed), len(promoted), selected, verified, low.sigma, high.sigma
    )


async def raw_screen_promotions(study: T0Study) -> tuple[str, ...]:
    """Candidates the unmodified AutoScientists screen promotes, with no confirmation"""
    sigma = (await study.calibration(DecisionStage.SCREEN)).sigma
    promoted: list[str] = []
    for candidate in study.configs:
        first, second = await study.pairs(candidate, study.screen_plan)
        verdict = autoscientists_gate(
            first.gain,
            sigma,
            second_gain=second.candidate.score - first.baseline.score,
        )
        if verdict is Verdict.PROMOTE:
            promoted.append(candidate)
    return tuple(promoted)


def summarize_arm(
    studies: Sequence[T0Study], outcomes: Sequence[Outcome]
) -> Mapping[str, JSONValue]:
    """Aggregate one arm over trials at one effect size."""
    trials = len(outcomes)
    planted = studies[0].improvement is not None
    wrong_confirmed = sum(
        any(item != study.improvement for item in outcome.confirmed)
        for study, outcome in zip(studies, outcomes, strict=True)
    )
    wrong_verified = sum(
        outcome.verified and outcome.selected != study.improvement
        for study, outcome in zip(studies, outcomes, strict=True)
    )
    summary: dict[str, JSONValue] = {
        "family_wise_false_confirmation": rate(wrong_confirmed, trials),
        "false_verification": rate(wrong_verified, trials),
        "mean_confirmations_run": fmean(o.confirmations_run for o in outcomes),
        "mean_screen_noise_sigma": fmean(o.screen_sigma for o in outcomes),
        "mean_confirm_noise_sigma": fmean(o.confirm_sigma for o in outcomes),
    }
    if planted:
        summary["detection"] = rate(
            sum(
                outcome.verified and outcome.selected == study.improvement
                for study, outcome in zip(studies, outcomes, strict=True)
            ),
            trials,
        )
        summary["planted_confirmed"] = rate(
            sum(
                study.improvement in outcome.confirmed
                for study, outcome in zip(studies, outcomes, strict=True)
            ),
            trials,
        )
        summary["worsening_promotions"] = rate(
            sum(
                study.worsening in outcome.confirmed
                for study, outcome in zip(studies, outcomes, strict=True)
            ),
            trials,
        )
    return summary


def summarize_raw(
    studies: Sequence[T0Study], promotions: Sequence[tuple[str, ...]]
) -> Mapping[str, JSONValue]:
    """Count raw screen promotions only; this is not a formal decision."""
    trials = len(promotions)
    summary: dict[str, JSONValue] = {
        "trials_with_false_promotion": rate(
            sum(
                any(item != study.improvement for item in promoted)
                for study, promoted in zip(studies, promotions, strict=True)
            ),
            trials,
        ),
        "mean_promotions": fmean(len(promoted) for promoted in promotions),
    }
    if studies[0].improvement is not None:
        summary["planted_promoted"] = rate(
            sum(
                study.improvement in promoted
                for study, promoted in zip(studies, promotions, strict=True)
            ),
            trials,
        )
    return summary


def delta_label(delta: float) -> str:
    return format(delta, "g")


async def validate(
    *,
    trials: int = DEFAULT_TRIALS,
    effects: Sequence[float] = DEFAULT_EFFECTS,
    family_size: int = 8,
) -> Mapping[str, JSONValue]:
    """Measure detection and false-discovery rates for every arm and effect size."""
    check_positive(trials)
    effects = tuple(effects)
    labels = [delta_label(delta) for delta in effects]
    if not effects or len(set(labels)) != len(labels):
        raise InvalidInputError("Declare distinct effect sizes")
    if not 3 <= family_size <= MAX_FAMILY_SIZE:
        raise InvalidInputError(f"Family size must be in [3, {MAX_FAMILY_SIZE}]")
    for delta in effects:
        # A gain of the whole baseline loss is the largest the objective allows.
        if not 0 <= delta * SCREEN_NOISE_SIGMA <= BASELINE_LOSS:
            raise InvalidInputError("Effect sizes must keep the planted loss >= 0")
    results: dict[str, dict[str, JSONValue]] = {arm.name: {} for arm in ARMS}
    results[RAW_ARM] = {}
    hashes: dict[str, dict[str, JSONValue]] = {arm.name: {} for arm in ARMS}
    for delta, label in zip(effects, labels, strict=True):
        studies = [
            T0Study(
                trial,
                gain=delta * SCREEN_NOISE_SIGMA if delta > 0 else None,
                worsening=WORSENING_SIGMAS * SCREEN_NOISE_SIGMA if delta > 0 else None,
                family_size=family_size,
            )
            for trial in range(trials)
        ]
        for arm in ARMS:
            outcomes = [await run_arm(study, arm) for study in studies]
            results[arm.name][label] = summarize_arm(studies, outcomes)
            hashes[arm.name][label] = studies[0].protocol_for(arm).hash
        results[RAW_ARM][label] = summarize_raw(
            studies, [await raw_screen_promotions(study) for study in studies]
        )
    reference = T0Study(0)
    return {
        "simulated": True,
        "adapter": "T0",
        "decider_version": 1,
        "t0_loop_hash": reference.loop.spec.hash,
        "flags": {
            "trials": trials,
            "effects": list(effects),
            "family_size": family_size,
        },
        "alpha": reference.protocol.alpha,
        "stage_seed_counts": [2, 16, 16],
        "calibration_seed_pairs_per_fidelity": 5,
        "screen_noise_sigma": SCREEN_NOISE_SIGMA,
        "confirm_noise_sigma": SCREEN_NOISE_SIGMA / 2,
        "worsening_sigmas": WORSENING_SIGMAS,
        "effect_unit": "screen-fidelity per-run noise sigma",
        "arms": {
            arm.name: {
                "policy": arm.policy.value,
                "screen_margin_sigmas": arm.screen_margin_sigmas,
                "confirmation_scope": arm.confirmation_scope.value,
                "trial_0_protocol_hash_by_effect": hashes[arm.name],
            }
            for arm in ARMS
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Print reproducible CPU-only simulated measurements, or write them to a file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--effects",
        default=",".join(delta_label(delta) for delta in DEFAULT_EFFECTS),
        help="Comma-separated gains in screen-noise sigmas; 0 is the all-null family",
    )
    parser.add_argument("--family-size", type=int, default=8)
    parser.add_argument("--output", help="Write the JSON report here, not to stdout")
    args = parser.parse_args(argv)
    report = asyncio.run(
        validate(
            trials=args.trials,
            effects=[float(item) for item in args.effects.split(",")],
            family_size=args.family_size,
        )
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
