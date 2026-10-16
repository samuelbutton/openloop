"""CPU-only batch wiring check with the existing simulated T0 loop, not a study."""

import asyncio
import os
import platform
import sys
from pathlib import Path

from openloop.ledger import Ledger, content_hash
from openloop.loops import T0, ExperimentEnv, LoopAction


async def run() -> None:
    """Write one T0 result at the coordinator-supplied ledger path."""
    loop = T0()
    with Ledger(Path(os.environ["OPENLOOP_LEDGER"])) as ledger:
        env = ExperimentEnv(
            loop,
            ledger,
            runner=loop.run,
            config={"coordinates": [0.0, 0.0]},
            environment_hash=content_hash(
                {"platform": platform.platform(), "python": sys.version}
            ),
            budget=1,
            hypothesis="Simulated T0 batch wiring check; no research claim",
        )
        await env.step(LoopAction(seed=42, fidelity="1x"))


if __name__ == "__main__":
    asyncio.run(run())
