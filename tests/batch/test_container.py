"""Opt-in test of the built batch image through the restricted host launcher."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from openloop.batch.receive import receive


@pytest.mark.skipif(
    not os.environ.get("OPENLOOP_BATCH_IMAGE"), reason="Set a built batch image ID"
)
def test_restricted_container_drains_and_returns_real_t0(tmp_path):
    image = os.environ["OPENLOOP_BATCH_IMAGE"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image)
    state, returned = tmp_path / "state", tmp_path / "return"
    state.mkdir()
    returned.mkdir()
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "version": 1,
                "image": image,
                "jobs": [
                    {
                        "name": "smoke",
                        "argv": ["python", "-m", "openloop.batch.smoke"],
                        "wall_seconds": 30,
                        "memory_bytes": 536870912,
                    }
                ],
            }
        )
    )
    repository = Path(__file__).parents[2]
    command = [
        str(repository / "scripts" / "run-batch.sh"),
        image,
        str(queue),
        str(state),
        str(returned),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (batch_root,) = returned.iterdir()
    accepted = receive(batch_root, tmp_path / "accepted")
    receipt = json.loads((accepted / "smoke" / "receipt.json").read_text())
    assert receipt["succeeded"]
    assert receipt["execution"]["exit_code"] == 0
    assert receipt["execution"]["elapsed_seconds"] > 0
    before = (accepted / "smoke" / "ledger-rows.json").read_bytes()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receive(batch_root, tmp_path / "accepted")
    assert (accepted / "smoke" / "ledger-rows.json").read_bytes() == before
