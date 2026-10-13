"""Observed means, bounded UCB exploration, and repeatable random selection."""

from dataclasses import replace

import pytest

from openloop.allocators import Bandit, Greedy, RandomSearch
from openloop.ledger import Direction, InvalidInputError, Ledger
from openloop.loops import T0, ContractError

from .helpers import complete, job, snapshot


@pytest.mark.parametrize("direction", list(Direction))
def test_greedy_explores_then_uses_mean_at_one_fidelity(
    ledger: Ledger, direction: Direction
) -> None:
    loop = T0()
    loop._spec = replace(loop.spec, direction=direction)
    complete(ledger, loop.spec, job(loop, 0), 0.8)
    complete(ledger, loop.spec, job(loop, 1), 0.2)
    # High-fidelity values cannot change a low-fidelity ordering.
    complete(ledger, loop.spec, job(loop, 0, fidelity="4x"), -100)
    offers = tuple(job(loop, index, seed=1) for index in range(3))
    assert Greedy("1x").select(snapshot(loop, ledger, offers)).jobs == (offers[2],)
    complete(ledger, loop.spec, job(loop, 2), 0.5)
    winner = offers[0] if direction is Direction.MAXIMIZE else offers[1]
    assert Greedy("1x").select(snapshot(loop, ledger, offers)).jobs == (winner,)
    complete(ledger, loop.spec, job(loop, 0, seed=2), -0.8)
    assert Greedy("1x").select(snapshot(loop, ledger, offers)).jobs == (
        offers[2] if direction is Direction.MAXIMIZE else offers[0],
    )


def test_ucb_explores_an_under_sampled_candidate(ledger: Ledger) -> None:
    loop = T0()
    for seed in range(20):
        complete(ledger, loop.spec, job(loop, 0, seed=seed), 0.2)
    complete(ledger, loop.spec, job(loop, 1), 0.4)
    offers = (job(loop, 0, seed=30), job(loop, 1, seed=30), job(loop, 2, seed=30))
    policy = Bandit("1x", 0, 1)
    assert policy.select(snapshot(loop, ledger, offers)).jobs == (offers[2],)
    state = snapshot(loop, ledger, offers[:2])
    assert Greedy("1x").select(state).jobs == (offers[0],)
    assert policy.select(state).jobs == (offers[1],)
    for seed in range(1, 21):
        complete(ledger, loop.spec, job(loop, 1, seed=seed), 0.4)
    assert policy.select(snapshot(loop, ledger, offers[:2])).jobs == (offers[0],)


@pytest.mark.parametrize("direction", list(Direction))
def test_ucb_honors_metric_direction(ledger: Ledger, direction: Direction) -> None:
    loop = T0()
    loop._spec = replace(loop.spec, direction=direction)
    complete(ledger, loop.spec, job(loop, 0), 0.2)
    complete(ledger, loop.spec, job(loop, 1), 0.8)
    offers = (job(loop, 0, seed=1), job(loop, 1, seed=1))
    winner = offers[0] if direction is Direction.MINIMIZE else offers[1]
    assert Bandit("1x", 0, 1).select(snapshot(loop, ledger, offers)).jobs == (winner,)


def test_ucb_clips_out_of_range_scores_and_says_so(ledger: Ledger) -> None:
    loop = T0()
    complete(ledger, loop.spec, job(loop, 0), 2)
    complete(ledger, loop.spec, job(loop, 1), 1)
    offers = (job(loop, 0, seed=1), job(loop, 1, seed=1))
    allocation = Bandit("1x", 0, 1).select(snapshot(loop, ledger, offers, slots=2))
    # A score of 2 is clipped to the bound 1, so it ties a score of exactly 1.
    assert {item.inputs.hash for item in allocation.jobs} == {
        item.inputs.hash for item in offers
    }
    assert "Clipped 1 out-of-range" in allocation.reason
    inside = Bandit("1x", 0, 2).select(snapshot(loop, ledger, offers, slots=2))
    assert "Clipped" not in inside.reason


def test_random_seed_changes_selection_and_policy_identity(ledger: Ledger) -> None:
    loop = T0()
    state = snapshot(
        loop, ledger, tuple(job(loop, index) for index in range(12)), slots=12
    )
    first = RandomSearch("1x", seed=1).select(state)
    second = RandomSearch("1x", seed=2).select(state)
    assert first.jobs != second.jobs
    assert first.policy_hash != second.policy_hash
    assert first.policy_hash != Greedy("1x").select(state).policy_hash


@pytest.mark.parametrize(
    "policy", [Greedy("unknown"), RandomSearch("unknown"), Bandit("unknown", 0, 1)]
)
def test_unknown_fidelity_is_rejected(ledger: Ledger, policy) -> None:
    with pytest.raises(ContractError, match="Unknown fidelity"):
        policy.select(snapshot(T0(), ledger, ()))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(1, 1), (2, 1), (float("nan"), 1), (0, float("inf")), (-1e308, 1e308)],
)
def test_invalid_reward_bounds_are_rejected(lower: float, upper: float) -> None:
    with pytest.raises(InvalidInputError):
        Bandit("1x", lower, upper)
