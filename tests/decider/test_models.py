"""Frozen protocol invariants and ledger evidence admission."""

import asyncio
from dataclasses import replace

import pytest

from openloop import decider
from openloop.decider import ConfirmationScope, Policy, PolicyRule, Sample
from openloop.ledger import InvalidInputError, Ledger, Status
from openloop.loops import ExperimentEnv, LoopAction
from openloop.studies.t0_family import T0Study


@pytest.mark.parametrize(
    "change",
    [
        "overlap",
        "too_few",
        "baseline",
        "same_data",
        "wrong_split",
        "wrong_fidelity",
        "empty",
        "alpha",
        "minimum_effect",
        "negative_margin",
        "nan_margin",
        "autoscientists_margin",
        "top_k_in_family_scope",
        "zero_top_k",
        "bool_top_k",
        "calibration_count",
        "screen_count",
        "held_out_cannot_reject_alpha",
        "top_k_cannot_reject",
    ],
)
def test_invalid_protocol(study: T0Study, change: str) -> None:
    protocol = study.protocol

    def construct() -> None:
        match change:
            case "overlap":
                replace(protocol, screen_noise_seeds=tuple(range(10)))
            case "too_few":
                replace(protocol, confirm=replace(protocol.confirm, seeds=(2,)))
            case "baseline":
                replace(protocol, candidates=(protocol.screen.template.candidate_hash,))
            case "same_data":
                replace(
                    protocol,
                    held_out=replace(
                        protocol.held_out,
                        template=replace(
                            protocol.held_out.template,
                            data_hash=protocol.confirm.template.data_hash,
                        ),
                    ),
                )
            case "wrong_split":
                replace(
                    protocol,
                    screen=replace(
                        protocol.screen,
                        metric=replace(protocol.screen.metric, split="held_out"),
                    ),
                )
            case "wrong_fidelity":
                replace(
                    protocol,
                    held_out=replace(
                        protocol.held_out,
                        template=replace(protocol.held_out.template, budget_amount=1),
                    ),
                )
            case "empty":
                replace(protocol, candidates=())
            case "alpha":
                replace(protocol, alpha=1)
            case "minimum_effect":
                replace(protocol, minimum_effect=-1)
            case "negative_margin":
                replace(protocol, screen_margin_sigmas=-0.1)
            case "nan_margin":
                replace(protocol, screen_margin_sigmas=float("nan"))
            case "autoscientists_margin":
                replace(
                    protocol,
                    policy=Policy.AUTOSCIENTISTS,
                    screen_margin_sigmas=1.0,
                )
            case "top_k_in_family_scope":
                replace(protocol, confirm_top_k=2)
            case "zero_top_k":
                replace(protocol, confirmation_scope="screened", confirm_top_k=0)
            case "bool_top_k":
                replace(protocol, confirmation_scope="screened", confirm_top_k=True)
            case "calibration_count":
                replace(protocol, confirm_noise_seeds=tuple(range(44, 52)))
            case "screen_count":
                replace(protocol, screen=replace(protocol.screen, seeds=(0, 1, 60)))
            case "held_out_cannot_reject_alpha":
                replace(
                    protocol,
                    alpha=0.5,
                    held_out=replace(protocol.held_out, seeds=(18,)),
                    confirm=replace(protocol.confirm, seeds=(2, 3, 4)),
                )
            case "top_k_cannot_reject":
                replace(
                    protocol,
                    alpha=0.001,
                    confirmation_scope="screened",
                    confirm_top_k=8,
                    confirm=replace(protocol.confirm, seeds=tuple(range(2, 12))),
                )

    with pytest.raises(InvalidInputError):
        construct()


def test_cache_alias_and_failure_are_not_samples(
    study: T0Study, ledger: Ledger
) -> None:
    env = ExperimentEnv(
        study.loop,
        ledger,
        runner=study.loop.run,
        config=study.baseline,
        environment_hash=study.environment,
        budget=2,
    )
    first = asyncio.run(env.step(LoopAction(0)))
    cached = asyncio.run(env.step(LoopAction(0)))
    measured = ledger.get(first.run_id)
    sample = Sample.from_run(measured, "loss")
    assert sample.attempt_id == first.attempt_id
    assert not cached.executed
    with pytest.raises(InvalidInputError, match="alias"):
        Sample.from_run(ledger.get(cached.run_id), "loss")
    with pytest.raises(InvalidInputError, match="successful"):
        Sample.from_run(replace(measured, status=Status.FAILED, result=None), "loss")
    with pytest.raises(InvalidInputError, match="missing"):
        Sample.from_run(measured, "other")


def test_protocol_hash_changes_with_policy_and_seed_plan(study: T0Study) -> None:
    protocol = study.protocol
    assert protocol.hash != replace(protocol, minimum_effect=0.01).hash
    assert protocol.hash != replace(protocol, alpha=0.01).hash
    assert (
        protocol.hash
        != replace(
            protocol, confirm=replace(protocol.confirm, seeds=tuple(range(100, 116)))
        ).hash
    )


def test_protocol_hash_covers_screen_and_scope_fields(study: T0Study) -> None:
    protocol = study.protocol
    screened = replace(protocol, confirmation_scope=ConfirmationScope.SCREENED)
    hashes = {
        protocol.hash,
        replace(protocol, screen_margin_sigmas=1.0).hash,
        screened.hash,
        replace(screened, confirm_top_k=2).hash,
        replace(screened, confirm_top_k=3).hash,
    }
    assert len(hashes) == 5
    assert replace(protocol, confirmation_scope="screened") == screened


def test_threshold_helpers(study: T0Study) -> None:
    protocol = study.protocol
    assert protocol.max_confirmations == 8
    assert protocol.test_alpha(8) == 0.05 / 8
    limited = replace(protocol, confirmation_scope="screened", confirm_top_k=3)
    assert limited.max_confirmations == 3
    assert replace(limited, confirm_top_k=99).max_confirmations == 8
    with pytest.raises(InvalidInputError):
        protocol.test_alpha(0)


def test_policy_registry_drives_seed_counts(
    study: T0Study, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert set(decider.POLICY_RULES) == set(Policy)
    protocol = study.protocol
    three_screen_seeds = replace(protocol.screen, seeds=(0, 1, 60))
    with pytest.raises(InvalidInputError, match="Screen seed count"):
        replace(protocol, screen=three_screen_seeds)
    rules = {
        **decider.POLICY_RULES,
        Policy.NOISE_AWARE: PolicyRule(screen_seeds=3, calibration_pairs=4),
    }
    monkeypatch.setattr("openloop.decider.models.POLICY_RULES", rules)
    with pytest.raises(InvalidInputError, match="pair count"):
        replace(protocol)
    with pytest.raises(InvalidInputError, match="pair count"):
        replace(protocol, screen=three_screen_seeds)
    replace(
        protocol,
        screen=three_screen_seeds,
        screen_noise_seeds=tuple(range(34, 42)),
        confirm_noise_seeds=tuple(range(44, 52)),
    )


def test_policy_rule_is_validated() -> None:
    for kwargs in (
        {"screen_seeds": 0, "calibration_pairs": 5},
        {"screen_seeds": 2, "calibration_pairs": 2},
        {"screen_seeds": 2, "calibration_pairs": 6},
    ):
        with pytest.raises(InvalidInputError):
            PolicyRule(**kwargs)
