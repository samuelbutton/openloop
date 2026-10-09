"""Frozen contract identity and validated input boundaries."""

from dataclasses import replace

import pytest

from openloop.ledger import InvalidInputError
from openloop.loops import T0, ContractError, Fidelity, LoopAction, Phase


def test_nested_execution_settings_are_snapshots(t0: T0) -> None:
    execution = {"limits": {"memory": 1024}}
    spec = replace(t0.spec, execution=execution)
    original = spec.hash
    execution["limits"]["memory"] = 2048
    assert spec.hash == original
    assert spec.hash != t0.spec.hash


def test_fidelity_cannot_reverse_work_order(t0: T0) -> None:
    with pytest.raises(ContractError, match="strictly increase"):
        replace(
            t0.spec,
            fidelities=(
                Fidelity("big", 4),
                Fidelity("small", 1),
            ),
        )


def test_fidelity_names_are_unique_and_amounts_strictly_increase(t0: T0) -> None:
    with pytest.raises(ContractError, match="unique"):
        replace(t0.spec, fidelities=(Fidelity("a", 1), Fidelity("a", 2)))
    with pytest.raises(ContractError, match="strictly increase"):
        replace(t0.spec, fidelities=(Fidelity("a", 2), Fidelity("b", 2)))
    with pytest.raises(ContractError, match="at least one"):
        replace(t0.spec, fidelities=())


def test_phase_and_budget_are_frozen(t0: T0) -> None:
    config = t0.normalize_config({"coordinates": [0, 0]})
    inputs = t0.spec.inputs(config, LoopAction(42), "a" * 64)
    with pytest.raises(ContractError, match="frozen"):
        t0.spec.validate_inputs(replace(inputs, phase="held_out"))
    with pytest.raises(ContractError, match="frozen"):
        t0.spec.validate_inputs(replace(inputs, budget_amount=2))


def test_invalid_action_and_duplicate_mutable_keys(t0: T0) -> None:
    with pytest.raises(InvalidInputError, match="Seed"):
        LoopAction(seed=-1)
    with pytest.raises(ContractError, match="Only retries"):
        LoopAction(seed=0, purpose="retry")
    with pytest.raises(ContractError, match="unique"):
        replace(t0.spec, mutable_keys=("coordinates", "coordinates"))


def three_levels(t0: T0):
    return replace(
        t0.spec,
        fidelities=(Fidelity("1x", 1), Fidelity("4x", 4), Fidelity("16x", 16)),
    )


def test_three_level_spec_resolves_levels_and_defaults_to_lowest(t0: T0) -> None:
    spec = three_levels(t0)
    assert [item.name for item in spec.fidelities] == ["1x", "4x", "16x"]
    assert spec.fidelity().name == "1x"
    assert spec.fidelity("16x").amount == 16
    with pytest.raises(ContractError, match="Unknown fidelity"):
        spec.fidelity("2x")
    config = t0.normalize_config({"coordinates": [0, 0]})
    default = spec.inputs(config, LoopAction(1), "a" * 64)
    assert (default.fidelity, default.phase, default.budget_amount) == (
        "1x",
        "screen",
        1,
    )
    with pytest.raises(ContractError, match="Unknown fidelity"):
        spec.inputs(config, LoopAction(1, "2x"), "a" * 64)


def test_phase_and_fidelity_vary_independently_in_the_hash(t0: T0) -> None:
    spec = three_levels(t0)
    config = t0.normalize_config({"coordinates": [0, 0]})
    hashes = {
        (phase, fidelity): spec.inputs(
            config, LoopAction(1, fidelity, phase), "a" * 64
        ).hash
        for phase in Phase
        for fidelity in ("1x", "4x", "16x")
    }
    assert len(set(hashes.values())) == 6
    for item in hashes:
        spec.validate_inputs(
            spec.inputs(config, LoopAction(1, item[1], item[0]), "a" * 64)
        )
