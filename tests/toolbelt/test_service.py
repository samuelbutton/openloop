"""Read-only selection queries preserve scope and independent execution counts."""

from dataclasses import replace
from pathlib import Path

import pytest

from openloop.ledger import InvalidInputError, Ledger, Metric, NotFoundError, Result
from openloop.toolbelt import Toolbelt
from openloop.toolbelt.__main__ import main

from .conftest import records


def test_query_filters_before_paging_and_excludes_reuse(
    ledger, toolbelt: Toolbelt, inputs, finish
) -> None:
    metric = Result({"loss": Metric(2, "BPB")})
    first = finish(ledger.submit(inputs), metric)
    ledger.submit(inputs)
    second = finish(ledger.submit(replace(inputs, seed=43)), metric)
    finish(ledger.submit(replace(inputs, seed=44, phase="held_out")), metric)
    finish(
        ledger.submit(replace(inputs, seed=45)),
        Result({"loss": Metric(999, "BPB", split="held_out")}),
    )
    ledger.submit(replace(inputs, seed=46))
    ledger.fail(ledger.submit(replace(inputs, seed=47)).id, "fixture")
    first_page = toolbelt.metric_query("loss", limit=1)
    assert first_page["total"] == 2
    assert first_page["next_offset"] == 1
    assert records(first_page["items"])[0]["run_id"] == first.id
    next_page = toolbelt.metric_query("loss", limit=1, offset=1)
    assert records(next_page["items"])[0]["run_id"] == second.id
    assert next_page["next_offset"] is None
    assert toolbelt.metric_query("loss", offset=99)["items"] == []
    assert toolbelt.metric_query("unknown")["total"] == 0
    assert toolbelt.metric_query("loss", candidate_hash="a" * 64)["total"] == 0
    assert toolbelt.metric_query("loss", loop_hash="a" * 64)["total"] == 0
    assert toolbelt.metric_query("loss", fidelity="different")["total"] == 0
    assert toolbelt.metric_query("loss", loop_hash=inputs.loop_hash)["total"] == 2


def test_query_excludes_replications_but_keeps_retries(
    ledger, toolbelt: Toolbelt, inputs, finish
) -> None:
    metric = Result({"loss": Metric(2, "BPB")})
    original = finish(ledger.submit(inputs), metric)
    finish(ledger.submit(inputs, purpose="replication"), metric)
    failed = ledger.fail(ledger.submit(replace(inputs, seed=43)).id, "fixture")
    retry = finish(
        ledger.submit(replace(inputs, seed=43), purpose="retry", retry_of=failed.id),
        metric,
    )
    report = toolbelt.metric_query("loss")
    assert [row["run_id"] for row in records(report["items"])] == [
        original.id,
        retry.id,
    ]


def test_no_query_or_error_changes_the_ledger(tmp_path: Path, repository: Path, inputs):
    path = tmp_path / "closed.sqlite3"
    with Ledger(path) as ledger:
        run = ledger.submit(inputs)
    before = path.read_bytes()
    belt = Toolbelt(path, repository)
    belt.metric_query("loss")
    belt.run_diff(run.id, run.id)
    with pytest.raises(NotFoundError):
        belt.run_diff("missing", run.id)
    with pytest.raises(InvalidInputError):
        belt.metric_query("loss", limit=0)
    belt.repository_map()
    assert path.read_bytes() == before
    absent = tmp_path / "absent.sqlite3"
    with pytest.raises(FileNotFoundError):
        Toolbelt(absent, repository)
    assert not absent.exists()


def test_main_rejects_non_sqlite_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repository: Path
) -> None:
    not_a_ledger = tmp_path / "notes.txt"
    not_a_ledger.write_text("this is not a SQLite database " * 20)
    monkeypatch.setattr(
        "sys.argv",
        ["toolbelt", "--ledger", str(not_a_ledger), "--repository", str(repository)],
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
