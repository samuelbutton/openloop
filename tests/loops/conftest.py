"""Tiny frozen data and CPU-only source stand-ins; never use native MLX training."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openloop.fineweb import POLICY, file_hash, json_hash, write_splits
from openloop.loops import T0, T1, TrainingSource
from openloop.loops.corpus import Corpus

TOY_TRAIN = """
from types import SimpleNamespace
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = 2**16
DEVICE_BATCH_SIZE = 16
DEPTH = 4
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
STARTUP_EXCLUDE_STEPS = 1
TIME_BUDGET = 300
seed_values = []
seed_api = SimpleNamespace(seed=lambda value: seed_values.append(value))
mx = SimpleNamespace(random=seed_api)
mx.random.seed(42)
tokenizer = Tokenizer.from_directory()
model = "CPU stand-in"
print(f"Time budget: {TIME_BUDGET}s")
step = 0
total_training_time = 0.0
dt = globals().get("dt", 1.0)
progresses = []
while True:
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    progresses.append(progress)
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt
    remaining = max(0, TIME_BUDGET - total_training_time)
    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break
total_tokens = step * TOTAL_BATCH_SIZE
raise RuntimeError("Candidate final evaluation must not execute")
"""

TOY_PREPARE = """
MAX_SEQ_LEN = 2048
EVAL_TOKENS = 1572864
TIME_BUDGET = 300
dt = 1.0
class Tokenizer:
    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        return cls()
def make_dataloader(*args):
    raise RuntimeError("Not needed by the CPU stand-in")
def evaluate_bpb(model, tokenizer, batch_size):
    assert model == "CPU stand-in"
    assert batch_size == 256
    batch, epoch = next(_document_batches("val", 2))
    assert batch and epoch == 1
    return 1.75
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    root = tmp_path / "fineweb"
    (root / "raw").mkdir(parents=True)
    splits = root / "splits"
    splits.mkdir()
    shard = root / "raw" / "tiny.parquet"
    pq.write_table(
        pa.table({"text": [f"document {index}" for index in range(500)]}), shard
    )
    manifest = {
        "policy": POLICY,
        "shards": [
            {
                "local_path": "raw/tiny.parquet",
                "sha256": file_hash(shard),
                "size_bytes": shard.stat().st_size,
            }
        ],
        **write_splits([shard], splits),
    }
    manifest["snapshot_sha256"] = json_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return Corpus.load(root)


@pytest.fixture
def t0() -> T0:
    return T0()


@pytest.fixture
def t1(corpus: Corpus, tmp_path: Path) -> T1:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.pkl", "token_bytes.npy"):
        (tokenizer / name).write_bytes(b"CPU fixture; no pickle or MLX is loaded")
    return T1(
        TrainingSource(TOY_TRAIN, TOY_PREPARE, "version = 1\npackage = []"),
        corpus,
        tokenizer,
        Path(sys.executable),
    )
