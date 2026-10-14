"""SIGKILL a coordinator after the final is journaled; a new process publishes."""

import asyncio
import os
import selectors
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from openloop.decider import PROTOCOL_NAMESPACE, CampaignJournal
from openloop.ledger import InvalidTransitionError, Ledger
from openloop.memory import ResearchMemory

ROOT = Path(__file__).parents[2]
BEFORE_WRITE = """
import asyncio, signal, sys
from openloop.ledger import Ledger, content_hash
from openloop.memory import ResearchMemory
from openloop.memory.store import NAMESPACE
from tests.memory.test_memory import closed_campaign
with Ledger(sys.argv[1]) as ledger:
    campaign = closed_campaign(ledger)
    subject, payload = asyncio.run(ResearchMemory(ledger).report(campaign))
    print(content_hash((NAMESPACE, subject, payload)), flush=True)
    signal.pause()
"""
DURING_FINAL = """
import signal, sys
from openloop.ledger import Ledger
from tests.memory.test_memory import closed_campaign

def stop():
    print('ready', flush=True)
    signal.pause()

with Ledger(sys.argv[1]) as ledger:
    closed_campaign(ledger, during_final=stop)
"""


def killed_after_first_line(script: str, path: Path) -> str:
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        env={"PATH": os.defpath, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"},
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=60), "Child did not reach the crash point"
            line = process.stdout.readline().decode().strip()
        process.send_signal(signal.SIGKILL)
        process.communicate(timeout=10)
        return line
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_new_process_publishes_the_identical_finding_after_a_crash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crash.sqlite3"
    expected = killed_after_first_line(BEFORE_WRITE, path)
    with Ledger(path) as ledger:
        memory = ResearchMemory(ledger)
        assert memory.query() == ()
        [registration] = ledger.derivations(PROTOCOL_NAMESPACE)
        finding = asyncio.run(memory.write_from_journal(registration.subject_hash))
        assert finding.hash == expected
        again = asyncio.run(memory.write_from_journal(registration.subject_hash))
        assert again == finding
        assert memory.query() == (finding,)


def test_restore_after_a_kill_during_the_final_never_grants_a_second_test(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crash.sqlite3"
    assert killed_after_first_line(DURING_FINAL, path) == "ready"
    with Ledger(path) as ledger:
        [registration] = ledger.derivations(PROTOCOL_NAMESPACE)
        journal = CampaignJournal(ledger)
        restored = asyncio.run(journal.restore(registration.subject_hash))
        assert restored.closed
        assert restored.final_consumed
        assert restored.final is None
        runs = ledger.query_all(include_reused=True)
        with pytest.raises(InvalidTransitionError, match="verified final"):
            asyncio.run(ResearchMemory(ledger).write(restored))
        with pytest.raises(InvalidTransitionError, match="verified final"):
            asyncio.run(
                ResearchMemory(ledger).write_from_journal(registration.subject_hash)
            )
        assert ResearchMemory(ledger).query() == ()
        assert ledger.query_all(include_reused=True) == runs
