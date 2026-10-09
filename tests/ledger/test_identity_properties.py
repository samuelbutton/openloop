"""Check the pure identity layer against generated inputs."""

import json
from dataclasses import replace
from random import Random

from hypothesis import given
from hypothesis import strategies as st

from openloop.ledger import ExperimentInputs, Reproducibility, canonical_json
from openloop.ledger.identity import JSONValue

digests = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)
text = st.text(min_size=1).filter(lambda value: value.strip() != "")
keys = st.text(max_size=8)

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(keys, children, max_size=4)
    ),
    max_leaves=12,
)
json_objects = st.dictionaries(keys, json_values, max_size=4)


@st.composite
def experiment_inputs(draw: st.DrawFn) -> ExperimentInputs:
    return ExperimentInputs(
        source_hash=draw(digests),
        dependencies_hash=draw(digests),
        data_hash=draw(digests),
        environment_hash=draw(digests),
        evaluator_hash=draw(digests),
        loop_hash=draw(digests),
        seed=draw(st.integers(min_value=0, max_value=2**63 - 1)),
        fidelity=draw(text),
        budget_unit=draw(text),
        budget_amount=draw(st.integers(min_value=1, max_value=2**63 - 1)),
        reproducibility=draw(st.sampled_from(Reproducibility)),
        config=draw(json_objects),
        execution=draw(json_objects),
        tokenizer_hash=draw(st.none() | digests),
        phase=draw(text),
    )


def reordered(value: JSONValue, rng: Random) -> JSONValue:
    """Rebuild every object with its keys in a shuffled insertion order."""
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: reordered(item, rng) for key, item in items}
    if isinstance(value, list):
        return [reordered(item, rng) for item in value]
    return value


@given(json_objects, st.randoms(use_true_random=False))
def test_canonical_json_ignores_key_order(value, rng):
    assert canonical_json(reordered(value, rng)) == canonical_json(value)


@given(json_objects)
def test_canonical_json_round_trips_to_plain_structure(value):
    assert json.loads(canonical_json(value)) == value


@given(experiment_inputs())
def test_rebuilt_inputs_keep_both_hashes(inputs):
    stored = json.loads(canonical_json(inputs.manifest))["inputs"]
    rebuilt = ExperimentInputs(**stored)
    assert rebuilt.hash == inputs.hash
    assert rebuilt.candidate_hash == inputs.candidate_hash


@given(experiment_inputs(), st.integers(min_value=-(2**53), max_value=2**53))
def test_int_and_float_values_have_different_hashes(inputs, number):
    as_int = replace(inputs, config={"value": number})
    as_float = replace(inputs, config={"value": float(number)})
    assert as_int.hash != as_float.hash
    assert as_int.candidate_hash != as_float.candidate_hash
