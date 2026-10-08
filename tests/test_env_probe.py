"""Check benchmark accounting, platform boundaries, and ledger preservation."""

import json
import platform

import pytest

from openloop import env_probe


def test_matmul_summary_uses_median_and_flop_count():
    summary = env_probe.summarize_matmul(1000, [0.003, 0.001, 0.002])
    assert summary["samples_ms"] == [3.0, 1.0, 2.0]
    assert summary["median_ms"] == 2.0
    assert summary["gflops"] == 1000.0


@pytest.mark.parametrize(
    "samples", [[], [0.0], [-0.001], [float("nan")], [float("inf")]]
)
def test_matmul_summary_rejects_invalid_timings(samples):
    with pytest.raises(ValueError):
        env_probe.summarize_matmul(32, samples)


@pytest.mark.parametrize("option", ["--size", "--repeats", "--warmup"])
def test_cli_rejects_zero_before_probing(monkeypatch, option):
    monkeypatch.setattr("sys.argv", ["env_probe", option, "0"])
    with pytest.raises(SystemExit) as error:
        env_probe.main()
    assert error.value.code == 2


def test_probe_rejects_unsupported_host(monkeypatch):
    monkeypatch.setattr(env_probe.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Apple silicon"):
        env_probe.probe_environment()


def test_cli_saves_environment_and_preserves_existing_ledger(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.json"
    record = {"environment": {"chip": "test chip"}}
    monkeypatch.setattr(env_probe, "probe_environment", lambda *args: record)
    monkeypatch.setattr("sys.argv", ["env_probe", "--output", str(ledger)])
    env_probe.main()
    assert json.loads(ledger.read_text()) == record
    with pytest.raises(SystemExit) as error:
        env_probe.main()
    assert error.value.code == 1
    assert json.loads(ledger.read_text()) == record


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires Apple silicon Metal",
)
def test_real_metal_matmul():
    result = env_probe.benchmark_matmul(size=32, repeats=2, warmup=1)
    assert len(result["samples_ms"]) == 2
    assert result["median_ms"] > 0
    assert result["gflops"] > 0
