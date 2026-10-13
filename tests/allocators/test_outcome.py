"""A finished allocator run, stated as a screen for the decider."""

import pytest

from openloop.allocators import Hyperband, ScreenOutcome, SuccessiveHalving
from openloop.ledger import Ledger
from openloop.loops import T0, ContractError

from .helpers import complete, snapshot
from .test_rungs import RUNGS, pool


def test_halving_outcome_names_survivors_evidence_and_seeds(ledger: Ledger) -> None:
    loop = T0()
    low, high, candidates = pool(loop)
    policy = SuccessiveHalving(candidates, RUNGS, eta=2)
    for offered, value in zip(low, (4.0, 3.0, 2.0, 1.0), strict=True):
        complete(ledger, loop.spec, offered, value)
    unfinished = snapshot(loop, ledger, (*low, *high))
    with pytest.raises(ContractError, match="unfinished"):
        policy.outcome(unfinished)
    runs = {}
    for offered in high[2:]:
        runs[offered.inputs.candidate_hash] = complete(
            ledger, loop.spec, offered, float(len(runs))
        )
    state = snapshot(loop, ledger, (*low, *high))
    outcome = policy.outcome(state)
    assert set(outcome.survivors) == set(candidates[2:])
    assert outcome.survivors[0] == candidates[2]
    assert policy.outcome(state, survivors=1).survivors == (candidates[2],)
    assert outcome.candidates == candidates
    assert outcome.seeds == (0, 7)
    assert len(outcome.evidence[candidates[2]]) == 2
    assert len(outcome.evidence[candidates[0]]) == 1
    assert set(outcome.evidence[candidates[3]]) >= {runs[candidates[3]].id}
    assert outcome.policy_hash == policy.select(state).policy_hash


def test_hyperband_outcome_covers_the_pool_and_ranks_across_brackets(
    ledger: Ledger,
) -> None:
    loop = T0()
    low, high, candidates = pool(loop, 8)
    policy = Hyperband(candidates, RUNGS, eta=2, seed=42)
    offers = (*low, *high)
    scores = {candidate: float(index) for index, candidate in enumerate(candidates)}
    for _ in range(10):
        step = policy.select(snapshot(loop, ledger, offers, slots=10))
        for offered in step.jobs:
            complete(ledger, loop.spec, offered, scores[offered.inputs.candidate_hash])
    state = snapshot(loop, ledger, offers)
    outcome = policy.outcome(state)
    assigned = {c for bracket in policy.brackets(loop.spec) for c in bracket.candidates}
    assert set(outcome.candidates) == set(candidates)
    assert all(bool(outcome.evidence[c]) == (c in assigned) for c in outcome.candidates)
    assert len(outcome.survivors) == 3
    ranked = sorted(outcome.survivors, key=lambda c: scores[c])
    assert outcome.survivors == tuple(ranked)
    assert policy.outcome(state, survivors=1).survivors == (ranked[0],)
    with pytest.raises(ContractError, match="positive"):
        policy.outcome(state, survivors=0)


def test_outcome_requires_pool_evidence_and_unique_survivors() -> None:
    with pytest.raises(ContractError, match="unique members"):
        ScreenOutcome("a" * 64, ("b" * 64,), ("c" * 64,), {"b" * 64: ()}, (0,))
    with pytest.raises(ContractError, match="cover"):
        ScreenOutcome("a" * 64, ("b" * 64,), (), {}, (0,))
