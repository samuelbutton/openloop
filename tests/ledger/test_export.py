"""Row transport retains original evidence and never writes a frozen ledger."""

import json

from openloop.ledger import Ledger


def test_export_retains_cache_events_plugins_and_timestamps(
    ledger, inputs, result, finish, tmp_path
):
    original = ledger.submit(inputs, hypothesis="Original")
    finish(original, result)
    alias = ledger.submit(inputs, hypothesis="Cache alias")
    derivation = ledger.record_derivation("example", inputs.hash, {"run": original.id})
    exported = ledger.export_rows()
    rows = json.loads(exported)
    assert rows["schema_version"] == ledger.schema_version
    assert len(rows["tables"]["attempt"]) == 1
    assert len(rows["tables"]["submission"]) == 2
    assert (
        next(row for row in rows["tables"]["submission"] if row["id"] == alias.id)[
            "reused_from"
        ]
        == original.id
    )
    assert rows["tables"]["derivation"][0]["hash"] == derivation.hash
    assert (
        max(rows["tables"]["event"], key=lambda event: event["sequence"])["timestamp"]
        == ledger.get(original.id).events[-1].timestamp
    )
    path = tmp_path / "ledger.sqlite3"
    before = path.read_bytes()
    with Ledger(path, read_only=True) as frozen:
        assert frozen.export_rows() == exported
    assert path.read_bytes() == before
