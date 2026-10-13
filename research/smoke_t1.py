"""Run ONE approved native T1 job on the shared recipe and keep its evidence.

This starts local MLX GPU training. It writes only to a new output directory,
with its own ledger, receipt and executor logs. No existing evidence changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from openloop.executors import LocalExecutor
from openloop.ledger import Ledger, Purpose, content_hash
from openloop.ledger.validation import plain
from openloop.loops import T1, ExperimentEnv, LoopAction, Phase, TrainingSource
from openloop.loops.corpus import Corpus


def save(path: Path, value: object) -> None:
    """Write one new evidence record, refusing all overwrites."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


async def smoke(args: argparse.Namespace) -> None:
    source = TrainingSource.load(args.upstream)
    corpus = Corpus.load(args.corpus)
    hardware = {
        "chip": subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "ram_bytes": int(
            subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True)
        ),
        "macos": platform.mac_ver()[0],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    executor = LocalExecutor(args.output / "executor", memory_sample_seconds=5)
    loop = T1(source, corpus, args.tokenizer, args.worker_python, executor=executor)
    save(
        args.output / "started.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "loop_hash": loop.spec.hash,
            "hardware": hardware,
            "fidelity": args.fidelity,
            "seed": args.seed,
        },
    )
    try:
        with Ledger(args.output / "ledger.sqlite3") as ledger:
            env = ExperimentEnv(
                loop,
                ledger,
                runner=loop.run,
                config={},
                environment_hash=content_hash(hardware),
                budget=loop.spec.fidelity(args.fidelity).amount,
                hypothesis="Native smoke run of the shared starting recipe",
            )
            step = await env.step(
                LoopAction(
                    args.seed, args.fidelity, Phase.SCREEN, purpose=Purpose.REPLICATION
                )
            )
            save(args.output / "run.json", ledger.get(step.run_id))
    finally:
        await executor.close()
    save(args.output / "finished.json", {"finished_at": datetime.now(UTC).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fidelity", default="8m")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--approve-local-training", action="store_true", required=True)
    asyncio.run(smoke(parser.parse_args()))


if __name__ == "__main__":
    main()
