"""Read-only local model snapshots. Paths never identify a model; bytes do."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from openloop.ledger import content_hash
from openloop.ledger.validation import check_digest

from .checkpoints import digest
from .contracts import TrainingError

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MODEL_REVISION = "53a32aee5e9447773fd2b85988395066aef3700a"
MODEL_FILES = (
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class ModelSnapshot:
    root: Path
    files: Mapping[str, str]
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION

    def __post_init__(self) -> None:
        if not self.files or any(
            Path(name).name != name or name.startswith(".") for name in self.files
        ):
            raise TrainingError("Model snapshot needs plain asset filenames")
        for value in self.files.values():
            check_digest(value)
        if len(self.revision) != 40 or any(
            c not in "0123456789abcdef" for c in self.revision
        ):
            raise TrainingError("Model revision must be an immutable commit SHA")
        object.__setattr__(self, "root", self.root.resolve())
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    @classmethod
    def capture(cls, root: Path) -> "ModelSnapshot":
        """Hash the pinned recipe's complete, already-downloaded model asset set."""
        return cls(root, {name: digest(root / name) for name in MODEL_FILES})

    @property
    def hash(self) -> str:
        return content_hash(
            {"model_id": self.model_id, "revision": self.revision, "files": self.files}
        )

    def verify(self) -> None:
        for name, expected in self.files.items():
            if digest(self.root / name) != expected:
                raise TrainingError(f"Frozen model asset changed: {name}")
