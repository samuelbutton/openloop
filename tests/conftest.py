import pytest
from hypothesis import settings

from openloop.ledger import ExperimentInputs, Ledger, Metric, Reproducibility, Result

# Derandomized so CI failures reproduce; a failing example is still shrunk.
settings.register_profile("ci", derandomize=True, max_examples=100, deadline=None)
settings.load_profile("ci")


@pytest.fixture
def inputs():
    return ExperimentInputs(
        source_hash="1" * 64,
        dependencies_hash="2" * 64,
        data_hash="3" * 64,
        environment_hash="4" * 64,
        evaluator_hash="5" * 64,
        loop_hash="6" * 64,
        seed=42,
        fidelity="screen",
        budget_unit="tokens",
        budget_amount=1024,
        reproducibility=Reproducibility.DETERMINISTIC,
        config={"depth": 4, "layers": [1, 2]},
        execution={"device": "cpu"},
    )


@pytest.fixture
def result():
    return Result({"val_bpb": Metric(1.75, "BPB")}, {"checkpoint": "7" * 64})


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.sqlite3") as store:
        yield store


@pytest.fixture
def finish(ledger):
    def finish(run, result):
        ledger.start_stage(run.id, "execution")
        ledger.start_stage(run.id, "evaluation")
        return ledger.complete(run.id, result)

    return finish
