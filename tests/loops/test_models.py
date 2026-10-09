"""Frozen contract identity and validated input boundaries."""

from dataclasses import replace

import pytest

from openloop.ledger import InvalidInputError
from openloop.loops import T0, ContractError, Fidelity, FidelityName, LoopAction


def test_nested_execution_settings_are_snapshots(t0: T0) -> None:
    execution = {"limits": {"memory": 1024}}
    spec = replace(t0.spec, execution=execution)
    original = spec.hash
    execution["limits"]["memory"] = 2048
    assert spec.hash == original
    assert spec.hash != t0.spec.hash


def test_fidelity_cannot_reverse_work_order(t0: T0) -> None:
    with pytest.raises(ContractError, match="cannot cost less"):
        replace(
            t0.spec,
            fidelities=(
                Fidelity(FidelityName.SCREEN, 4),
                Fidelity(FidelityName.CONFIRM, 1),
            ),
        )


def test_duplicate_fidelity_is_not_a_complete_contract(t0: T0) -> None:
    with pytest.raises(ContractError, match="exactly one"):
        replace(t0.spec, fidelities=(Fidelity(FidelityName.SCREEN, 1),) * 2)


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
