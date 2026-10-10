"""Kill a live CPU coordinator. Only temporary test ledgers are created or read.

The parent waits for a committed stage before SIGKILL. There is no mocked
exception, guessed delay, GPU job, or production-ledger access.
"""

import asyncio
import os
import selectors
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from openloop.ledger import Ledger, Purpose, Stage, Status
from openloop.loops import T0, ExperimentEnv, LoopAction

COORDINATOR = """
import asyncio, signal, sys
from openloop.ledger import Ledger
from openloop.loops import T0, ExperimentEnv, LoopAction

def pause():
    (run,) = ledger.query(status='running')
    print(run.id, flush=True)
    signal.pause()

class PausedEvaluator(T0):
    def evaluate(self, job, output):
        if sys.argv[2] == 'evaluation':
            pause()
        return super().evaluate(job, output)

async def runner(job, context):
    if sys.argv[2] == 'execution':
        pause()
    return await workload.run(job, context)

with Ledger(sys.argv[1]) as ledger:
    workload = PausedEvaluator()
    env = ExperimentEnv(workload, ledger, runner=runner,
        config={'coordinates': [0, 0]}, environment_hash='a' * 64, budget=1)
    asyncio.run(env.step(LoopAction(42, request_key='interrupted')))
"""


def episode(workload: T0, ledger: Ledger) -> ExperimentEnv:
    return ExperimentEnv(
        workload,
        ledger,
        runner=workload.run,
        config={"coordinates": [0, 0]},
        environment_hash="a" * 64,
        budget=1,
    )


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and pipe selectors need POSIX")
@pytest.mark.parametrize("stage", [Stage.EXECUTION, Stage.EVALUATION])
def test_sigkill_preserves_committed_evidence_and_allows_retry(
    tmp_path: Path, stage: Stage
) -> None:
    path = tmp_path / "crash.sqlite3"
    workload = T0()
    with Ledger(path) as ledger:
        baseline_step = asyncio.run(episode(workload, ledger).step(LoopAction(41)))
        baseline = ledger.get(baseline_step.run_id)

    with subprocess.Popen(
        [sys.executable, "-c", COORDINATOR, str(path), stage],
        env={"PATH": os.defpath, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout is not None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=10), (
                    "Coordinator did not reach its stage"
                )
                run_id = child.stdout.readline().strip()
            assert run_id, "Coordinator exited before committing its stage"
            with Ledger(path) as ledger:
                interrupted = ledger.get(run_id)
                before = ledger.query(include_reused=True)
                assert interrupted.status is Status.RUNNING
                assert interrupted.result is None
                assert interrupted.events[-1].stage is stage
            child.send_signal(signal.SIGKILL)
            _, stderr = child.communicate(timeout=10)
            assert child.returncode == -signal.SIGKILL, stderr
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)

    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    with Ledger(path) as ledger:
        assert ledger.query(include_reused=True) == before
        assert ledger.get(baseline.id) == baseline
        assert ledger.get(run_id) == interrupted
        # Incomplete work can be referenced, but it is not a successful cache hit.
        pending = ledger.submit(interrupted.inputs, request_key="pending-reference")
        assert pending.attempt_id == interrupted.attempt_id
        assert pending.status is Status.RUNNING
        assert pending.result is None
        assert ledger.query(status="succeeded") == (baseline,)

        failed = ledger.fail(run_id, "Coordinator terminated by SIGKILL")
        assert failed.status is Status.FAILED
        assert failed.result is None
        assert failed.events[: len(interrupted.events)] == interrupted.events
        assert failed.stages[stage][-1].payload["duration_clock"] == "wall"
        retry = asyncio.run(
            episode(workload, ledger).step(
                LoopAction(
                    42,
                    request_key="retry",
                    purpose=Purpose.RETRY,
                    retry_of=run_id,
                )
            )
        )
        assert retry.executed
        assert retry.status is Status.SUCCEEDED
        assert retry.attempt_id != interrupted.attempt_id
        recovered = ledger.get(retry.run_id)
        assert recovered.inputs == interrupted.inputs
        assert recovered.retry_of == interrupted.attempt_id
        assert run_id in recovered.parents
        assert ledger.get(run_id) == failed

        cached = asyncio.run(episode(workload, ledger).step(LoopAction(42)))
        assert not cached.executed
        assert cached.work_units == 0
        assert cached.attempt_id == retry.attempt_id
        assert cached.result == retry.result
        assert len(ledger.query(status="succeeded")) == 2
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
