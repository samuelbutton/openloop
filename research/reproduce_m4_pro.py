"""Run the reviewed public M4 Pro winner and retain separate reproduction evidence.

This opt-in script runs local GPU training. It writes only to a new output
directory. It does not change upstream code, cached data, or any ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import re
import subprocess
from pathlib import Path

from openloop.executors import ExecutorError, Limits, LocalExecutor, ProcessJob, State

REVISION = "1d8377683d9a5b1af14f5933e6f5130c54e94460"
REFERENCE_BPB = 1.429396
SOURCE_HASHES = {
    "train_mlx.py": "f53ec5cfee3b768627b02387645f5dca3ba2fec51ae24bf5521854bcba821d52",
    "prepare.py": "f74ae808d992db1ea86a11c374ae7bdf074ee422cca5510e69ccb2628146f9ce",
    "backends/muon_mlx.py": (
        "a941fdb060dac1d9a76f5c19bd72b4996e67e3f387a8ee35fc29b173e8d54d38"
    ),
    "backends/__init__.py": (
        "8da786fe82e8ece027fbf4a49dae761bb98fe20079f511f3a38c5ba8e9ed28ba"
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_summary(log: str) -> dict[str, float | int | str]:
    """Parse only the final summary. Reject missing or nonfinite BPB."""
    summary: dict[str, float | int | str] = {}
    section = log.rsplit("\n---\n", 1)
    if len(section) != 2:
        raise ExecutorError("Training omitted its final summary")
    integers = {"num_steps", "depth"}
    text = {"backend", "chip"}
    for line in section[1].splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.strip()
        if key in integers:
            if not value.isdecimal() or int(value) <= 0:
                raise ExecutorError(f"Invalid integer summary: {key}")
            summary[key] = int(value)
        elif key in text:
            summary[key] = value
        else:
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                raise ExecutorError(f"Invalid numeric summary: {key}")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ExecutorError(f"Nonfinite summary: {key}")
            summary[key] = numeric
    if "val_bpb" not in summary or "num_steps" not in summary:
        raise ExecutorError("Training omitted BPB or step count")
    return summary


def save(path: Path, value: object) -> None:
    """Write a new evidence file without overwriting prior evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


async def reproduce(source: Path, python: Path, output: Path) -> None:
    """Run exact pinned source on the pre-existing ClimbMix cache."""
    for name, expected in SOURCE_HASHES.items():
        if digest(source / name) != expected:
            raise ExecutorError(f"Reference source changed: {name}")
    cache = Path.home() / ".cache/autoresearch"
    manifest = json.loads(
        Path("research/artifacts/autoresearch-mlx/data_manifest.json").read_text()
    )
    expected_shards = {
        Path(item["path"]).name
        for item in manifest
        if item["path"].endswith(".parquet")
    }
    if {path.name for path in (cache / "data").glob("*.parquet")} != expected_shards:
        raise ExecutorError(
            "Cached shard membership differs from the recorded snapshot"
        )
    for item in manifest:
        if digest(cache / item["path"]) != item["sha256"]:
            raise ExecutorError(f"Cached data changed: {item['path']}")
    runtime = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import json,platform; from importlib.metadata import version; "
                "print(json.dumps({'python':platform.python_version(),'packages':"
                "{n:version(n) for n in ('mlx','mlx-metal','numpy','pyarrow','regex',"
                "'requests','rustbpe','tiktoken')}}))",
            ],
            text=True,
        )
    )
    hardware = json.loads(
        subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "provenance.json",
        {
            "repository": "https://github.com/elementalcollision/autoresearch",
            "revision": REVISION,
            "source_sha256": SOURCE_HASHES,
            "runtime": runtime,
            "macos": platform.mac_ver()[0],
            "chip": hardware["SPDisplaysDataType"][0]["sppci_model"],
            "gpu_cores": int(hardware["SPDisplaysDataType"][0]["sppci_cores"]),
            "ram_bytes": int(
                subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"])
            ),
            "cached_data": manifest,
            "reference_bpb": REFERENCE_BPB,
            "reference_gpu_cores": 16,
            "evaluation_tokens": 5_242_880,
            "dataset": "karpathy/climbmix-400b-shuffle",
            "seed": 42,
            "runtime_note": (
                "Reference uv.lock lacks MLX; "
                "observed versions are recorded explicitly."
            ),
        },
    )
    executor = LocalExecutor(output / "jobs", memory_sample_seconds=5)
    try:
        job_id = await executor.submit(
            ProcessJob(
                (str(python), "-u", "train_mlx.py"),
                cwd=source,
                environment={
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "HOME": str(Path.home()),
                },
                reviewed=True,
                gpu=True,
                limits=Limits(
                    wall_seconds=900,
                    memory_bytes=16 * 1024**3,
                    output_bytes=16 * 1024**2,
                ),
            ),
            key="public-m4-pro-reference",
        )
        print(f"job_id: {job_id}", flush=True)
        receipt = await executor.collect(job_id)
        save(
            output / "execution.json",
            {
                "job_id": receipt.job_id,
                "input_hash": receipt.input_hash,
                "state": receipt.state,
                "exit_code": receipt.exit_code,
                "submitted_at": receipt.submitted_at.isoformat(),
                "started_at": receipt.started_at.isoformat()
                if receipt.started_at
                else None,
                "finished_at": receipt.finished_at.isoformat()
                if receipt.finished_at
                else None,
                "elapsed_seconds": receipt.elapsed_seconds,
                "peak_rss_bytes": receipt.peak_rss_bytes,
                "error": receipt.error,
                "artifacts": [
                    {
                        "path": str(item.path.relative_to(output)),
                        "sha256": item.sha256,
                        "size_bytes": item.size_bytes,
                    }
                    for item in receipt.artifacts
                ],
            },
        )
        if receipt.state != State.SUCCEEDED:
            raise ExecutorError(f"Reference failed: {receipt.state}: {receipt.error}")
        stdout = next(
            item.path for item in receipt.artifacts if item.path.name == "stdout"
        )
        summary = parse_summary(stdout.read_text())
        save(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        await executor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-local-training", action="store_true", required=True)
    args = parser.parse_args()
    # Dereferencing a venv's Python symlink selects the base interpreter and
    # loses that environment's installed packages.
    asyncio.run(
        reproduce(args.source.resolve(), args.python.absolute(), args.output.resolve())
    )


if __name__ == "__main__":
    main()
