"""Capture Apple silicon identity and a separate synchronized MLX matmul calibration."""

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter


def positive_int(value: str) -> int:
    """Parse strictly positive benchmark dimensions and iteration counts."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def summarize_matmul(size: int, samples_s: list[float]) -> dict[str, object]:
    """Report measured latency and conventional 2*N**3 matmul throughput."""
    if (
        size <= 0
        or not samples_s
        or any(not isfinite(sample) or sample <= 0 for sample in samples_s)
    ):
        raise ValueError("size and timing samples must be finite and positive")
    elapsed = median(samples_s)
    return {
        "size": size,
        "samples_ms": [sample * 1_000 for sample in samples_s],
        "median_ms": elapsed * 1_000,
        "gflops": 2 * size**3 / elapsed / 1e9,
    }


def benchmark_matmul(
    size: int = 2048, repeats: int = 10, warmup: int = 3
) -> dict[str, object]:
    """Time float32 GPU matmuls, excluding input creation and warm-up."""
    if min(size, repeats, warmup) <= 0:
        raise ValueError("size, repeats, and warmup must be positive")

    import mlx.core as mx  # pyright: ignore[reportMissingImports] # Apple silicon only

    if not mx.metal.is_available():
        raise RuntimeError("The MLX Metal GPU backend is unavailable")

    samples = []
    with mx.stream(mx.gpu):
        key_a, key_b = mx.random.split(mx.random.key(0))
        a = mx.random.normal((size, size), dtype=mx.float32, key=key_a)
        b = mx.random.normal((size, size), dtype=mx.float32, key=key_b)
        mx.eval(a, b)
        mx.synchronize(mx.gpu)

        for iteration in range(warmup + repeats):
            start = perf_counter()
            result = a @ b
            mx.eval(result)
            mx.synchronize(mx.gpu)
            elapsed = perf_counter() - start
            if iteration >= warmup:
                samples.append(elapsed)
            if (
                iteration == warmup + repeats - 1
                and not mx.all(mx.isfinite(result)).item()
            ):
                raise RuntimeError("Matmul produced non-finite values")

    return {
        "device": "gpu",
        "dtype": "float32",
        "seed": 0,
        "warmup": warmup,
        "repeats": repeats,
        **summarize_matmul(size, samples),
    }


def environment_identity() -> dict[str, object]:
    """Collect stable facts that can affect results, without benchmarking."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("The environment probe requires native Apple silicon macOS")

    def sysctl(name: str) -> str:
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", name], text=True
        ).strip()

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "ram_bytes": int(sysctl("hw.memsize")),
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
    }


def probe_environment(
    size: int = 2048, repeats: int = 10, warmup: int = 3
) -> dict[str, object]:
    """Collect a stable environment identity and a separate timed calibration."""
    environment = environment_identity()
    return {
        "environment": environment,
        "calibration": {
            "captured_at": datetime.now(UTC).isoformat(),
            "matmul": benchmark_matmul(size, repeats, warmup),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=positive_int, default=2048)
    parser.add_argument("--repeats", type=positive_int, default=10)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--output", type=Path, help="save a new ledger JSON file")
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite ledger: {args.output}")
        record = probe_environment(args.size, args.repeats, args.warmup)
        payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as output:
                output.write(payload)
        print(payload, end="")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"env_probe: {error}\n")


if __name__ == "__main__":
    main()
