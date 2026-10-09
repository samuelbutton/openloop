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
    with pytest.raises(ValueError, match="finite and positive"):
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


def fake_host(monkeypatch):
    monkeypatch.setattr(env_probe.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(env_probe.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        env_probe.platform, "mac_ver", lambda: ("15.0", ("", "", ""), "")
    )
    monkeypatch.setattr(env_probe.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(env_probe, "version", lambda name: f"{name}-1")
    values = {"machdep.cpu.brand_string": "Test chip\n", "hw.memsize": "17179869184\n"}
    monkeypatch.setattr(
        env_probe.subprocess, "check_output", lambda cmd, text: values[cmd[-1]]
    )


def test_identity_is_deterministic_and_excludes_measurements(monkeypatch):
    fake_host(monkeypatch)
    first = env_probe.environment_identity()
    assert env_probe.environment_identity() == first
    assert set(first) == {
        "chip",
        "ram_bytes",
        "macos",
        "architecture",
        "python",
        "mlx_version",
        "mlx_lm_version",
    }
    assert first["ram_bytes"] == 17179869184


def test_probe_separates_identity_from_calibration(monkeypatch):
    fake_host(monkeypatch)
    monkeypatch.setattr(env_probe, "benchmark_matmul", lambda *args: {"size": 1})
    first = env_probe.probe_environment()
    second = env_probe.probe_environment()
    assert first["environment"] == second["environment"]
    assert first["environment"] == env_probe.environment_identity()
    calibration = first["calibration"]
    assert isinstance(calibration, dict)
    assert set(calibration) == {"captured_at", "matmul"}
    assert calibration["matmul"] == {"size": 1}


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
    samples, median_ms, gflops = (
        result["samples_ms"],
        result["median_ms"],
        result["gflops"],
    )
    assert isinstance(samples, list)
    assert len(samples) == 2
    assert isinstance(median_ms, float)
    assert median_ms > 0
    assert isinstance(gflops, float)
    assert gflops > 0
