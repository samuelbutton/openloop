"""Run read-only T4 smoke lookups through the Env and an isolated temporary ledger.

Loads only existing verified benchmark files. Writes one new receipt file;
never changes benchmark manifests, previous evidence, or an existing ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from openloop.ledger import Ledger, content_hash
from openloop.ledger.validation import plain
from openloop.loops import (
    HPOB,
    ExperimentEnv,
    LoopAction,
    NASBench201,
    Phase,
    TabularLoop,
)
from openloop.loops.tabular import Benchmark, MetaSplit


async def probe(loop: TabularLoop) -> dict[str, object]:
    """Query each fidelity, confirm cache reuse, and retain actual ledger receipts."""
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    setting = loop.spec.mutable_keys[0]
    results = []
    with (
        TemporaryDirectory(prefix="openloop-t4-") as directory,
        Ledger(Path(directory) / "smoke.sqlite3") as ledger,
    ):
        for fidelity in loop.spec.fidelities:
            seed = loop.seeds(0, fidelity=fidelity.name)[0]
            env = ExperimentEnv(
                loop,
                ledger,
                runner=loop.run,
                config={setting: 0},
                environment_hash=content_hash(environment),
                budget=2 * fidelity.amount,
            )
            action = LoopAction(seed, fidelity.name, Phase.CONFIRM)
            measured = await env.step(action)
            cached = await env.step(action)
            assert measured.executed and not cached.executed
            assert measured.attempt_id == cached.attempt_id
            results.append(
                {
                    "fidelity": fidelity.name,
                    "seed": seed,
                    "run": plain(ledger.get(measured.run_id)),
                    "cache_hit": not cached.executed,
                }
            )
    return {
        "spec": plain(loop.spec),
        "environment": environment,
        "candidate_count": len(loop.candidates),
        "queries": results,
        "training_executed": False,
        "test_scores_requested": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=list(Benchmark))
    parser.add_argument("--data", type=Path, default=Path("data/tabular"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-space", default="4796")
    parser.add_argument("--dataset", default="31")
    parser.add_argument(
        "--split", choices=list(MetaSplit), default=MetaSplit.VALIDATION
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output evidence already exists; choose a new path")
    start = time.perf_counter()
    loop = (
        NASBench201.load(args.data)
        if args.benchmark == Benchmark.NAS201
        else HPOB.load(args.data, args.search_space, args.dataset, split=args.split)
    )
    report = asyncio.run(probe(loop))
    report["elapsed_seconds"] = time.perf_counter() - start
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Verified {args.benchmark}: {len(loop.candidates)} candidates; "
        f"receipt {args.output}"
    )


if __name__ == "__main__":
    main()
