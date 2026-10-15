"""Local HTTP integration with temporary evidence only; no external service."""

import json
import sqlite3
import sys
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

from openloop.ledger import Ledger, Metric, Result
from openloop.ui.__main__ import main
from openloop.ui.server import create_server


@pytest.fixture
def served(tmp_path, inputs):
    path = tmp_path / "ui.sqlite3"
    with Ledger(path) as ledger:
        run = ledger.submit(inputs)
        ledger.start_stage(run.id, "execution")
        ledger.start_stage(run.id, "evaluation")
        ledger.complete(run.id, Result({"loss": Metric(1, "BPB")}))
    original = path.read_bytes()
    server = create_server(path, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, run.id
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()
        assert path.read_bytes() == original


def request(server, path, *, headers=None, method="GET"):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_assets_snapshot_and_diff_are_read_only(served):
    server, run_id = served
    assert server.server_address[0] == "127.0.0.1"
    for path, mime in [
        ("/", "text/html"),
        ("/app.js", "text/javascript"),
        ("/dom.js", "text/javascript"),
        ("/curves.js", "text/javascript"),
        ("/style.css", "text/css"),
    ]:
        code, headers, data = request(server, path)
        assert code == 200
        assert headers["Content-Type"].startswith(mime)
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert headers["Cache-Control"] == "no-store"
        assert data
    code, _, body = request(server, "/api/snapshot")
    assert code == 200
    assert json.loads(body)["summary"]["runs"] == 1
    code, _, body = request(server, f"/api/diff?baseline={run_id}&candidate={run_id}")
    assert code == 200
    assert json.loads(body)["independent_attempts"] is False


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example"},
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_foreign_origins_and_dns_rebinding_are_rejected(served, headers):
    server, _ = served
    assert request(server, "/api/snapshot", headers=headers)[0] == 403


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/../.env", 404),
        ("/api/snapshot?ledger=other", 404),
        ("/api/diff?baseline=missing&candidate=missing", 404),
        ("/api/diff?baseline=x&baseline=y&candidate=z", 400),
        ("/api/diff?baseline=&candidate=x", 400),
    ],
)
def test_bad_routes_and_arguments_cannot_select_files(served, path, status):
    server, _ = served
    assert request(server, path)[0] == status


def test_writes_and_missing_ledgers_are_rejected(served, tmp_path: Path):
    server, _ = served
    assert request(server, "/api/snapshot", method="POST")[0] == 405
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        create_server(path, port=0)
    assert not path.exists()


def test_unreadable_ledger_returns_json_error(tmp_path: Path):
    path = tmp_path / "ui.sqlite3"
    with Ledger(path):
        pass
    server = create_server(path, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path.write_bytes(b"not a database" * 100)
        code, headers, body = request(server, "/api/snapshot")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert code == 503
    assert headers["Content-Type"].startswith("application/json")
    assert json.loads(body) == {"error": "Cannot read the ledger"}


def test_cli_reports_non_ledger_files_as_usage_errors(
    tmp_path: Path, monkeypatch, capsys
):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"not a database" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        create_server(path, port=0)
    monkeypatch.setattr(sys, "argv", ["openloop.ui", "--ledger", str(path)])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
