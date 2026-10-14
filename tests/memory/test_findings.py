"""Findings state ledger facts about the change and its operator links only."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openloop.decider import Pair, Sample
from openloop.ledger import (
    Derivation,
    ExperimentInputs,
    InvalidInputError,
    Ledger,
    Metric,
    canonical_json,
)
from openloop.memory import ResearchMemory
from openloop.memory.findings import describe_change
from openloop.operators import record_outcome
from tests.memory.test_memory import closed_campaign
from tests.operators.conftest import Bound, SandboxStandIn, World, bind_card


def facts(finding: Derivation) -> dict[str, Any]:
    return json.loads(canonical_json(finding.payload))


@pytest.fixture
def sandbox(tmp_path: Path) -> SandboxStandIn:
    return SandboxStandIn(tmp_path / "sandbox")


def test_unlinked_finding_states_the_change_and_has_no_links(ledger: Ledger) -> None:
    campaign = closed_campaign(ledger)
    finding = asyncio.run(ResearchMemory(ledger).write(campaign))
    assert facts(finding)["links"] == []
    change = facts(finding)["change"]
    template = campaign.protocol.held_out.template
    assert change["code_changed"] is False
    assert change["config"]["added"] == {}
    assert change["config"]["removed"] == {}
    assert set(change["config"]["changed"]) == {"coordinates"}
    assert change["config"]["changed"]["coordinates"]["from"] == [0.5] * 4
    assert change["candidate_hash"] == campaign.selected
    assert change["source_hash"]["baseline"] == template.source_hash
    registration = facts(finding)["registration"]
    assert registration["registered_at"] < registration["first_run_at"]


def test_linked_finding_pairs_prediction_with_observation(
    ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    bound: list[Bound] = []

    def bind(world: World) -> None:
        assert world.study.improvement is not None
        bound.append(bind_card(world, sandbox, world.study.improvement))

    campaign = closed_campaign(ledger, bind=bind)
    assert campaign.final is not None
    outcome = record_outcome(ledger, bound[0].card, campaign.final)
    finding = asyncio.run(ResearchMemory(ledger).write(campaign))
    [link] = facts(finding)["links"]
    assert link["card_hash"] == bound[0].card.hash
    assert (
        link["implementation_success_sequence"] == bound[0].candidate.success_sequence
    )
    assert link["snapshot_hash"] == bound[0].candidate.snapshot_hash
    assert link["outcome_sequence"] == outcome.sequence
    assert link["prediction"]["delta"] == -0.1
    assert link["prediction"]["probability_of_improvement"] == 0.7
    observation = link["observation"]
    gain = campaign.final.median_gain
    assert gain is not None
    assert observation["delta"] == -gain
    assert observation["median_gain"] == gain
    assert observation["direction"] == "minimize"
    assert observation["sample_count"] == campaign.final.sample_count
    # Hypothesis text is not a fact and must not enter the report.
    text = repr(finding.payload)
    for story in ("Move closer", "Planted fixture", "Edit candidate.txt"):
        assert story not in text


def test_linked_finding_before_an_outcome_marks_it_absent(
    ledger: Ledger, sandbox: SandboxStandIn
) -> None:
    def bind(world: World) -> None:
        assert world.study.improvement is not None
        bind_card(world, sandbox, world.study.improvement)

    campaign = closed_campaign(ledger, bind=bind)
    finding = asyncio.run(ResearchMemory(ledger).write(campaign))
    [link] = facts(finding)["links"]
    assert link["outcome_sequence"] is None


def _pair(
    inputs: ExperimentInputs,
    seed: int,
    base: dict[str, object],
    cand: dict[str, object],
) -> Pair:
    metric = Metric(1.0, "loss")

    def sample(name: str, config: dict[str, object], source: str) -> Sample:
        return Sample(
            name,
            name,
            replace(inputs, seed=seed, config=config, source_hash=source),  # type: ignore[arg-type]
            metric,
            metric_name="loss",
        )

    return Pair(sample(f"b{seed}", base, "1" * 64), sample(f"c{seed}", cand, "9" * 64))


def test_change_lists_added_removed_changed_keys_and_code(inputs) -> None:
    base = {"lr": 1, "depth": 2, "gone": True, "same": [1]}
    cand = {"lr": 3, "depth": 2, "new": "x", "same": [1]}
    change = describe_change(
        (_pair(inputs, 1, base, cand), _pair(inputs, 2, base, cand))
    )
    assert change["config"] == {
        "added": {"new": "x"},
        "removed": {"gone": True},
        "changed": {"lr": {"from": 1, "to": 3}},
    }
    assert change["code_changed"] is True
    assert change["source_hash"] == {"baseline": "1" * 64, "candidate": "9" * 64}


def test_change_rejects_inputs_that_vary_between_seeds(inputs) -> None:
    pairs = (
        _pair(inputs, 1, {"lr": 1}, {"lr": 2}),
        _pair(inputs, 2, {"lr": 1}, {"lr": 3}),
    )
    with pytest.raises(InvalidInputError, match="differ between seeds"):
        describe_change(pairs)
