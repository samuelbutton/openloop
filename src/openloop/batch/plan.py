"""Pure batch contracts. Queue files are trusted coordinator input, never candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from openloop.executors.models import InvalidJobError, Limits, ProcessJob
from openloop.ledger import content_hash


@dataclass(frozen=True)
class BatchJob:
    """One reviewed command. It must write its ledger at OPENLOOP_LEDGER."""

    name: str
    argv: tuple[str, ...]
    wall_seconds: float
    memory_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.name
        ):
            raise InvalidJobError("Job names need 1-64 letters, digits, '-' or '_'")
        validated = ProcessJob(self.argv, limits=self.limits, reviewed=True)
        object.__setattr__(self, "argv", validated.argv)

    @property
    def limits(self) -> Limits:
        return Limits(wall_seconds=self.wall_seconds, memory_bytes=self.memory_bytes)


@dataclass(frozen=True)
class Batch:
    """A finite ordered queue. Changed content gets a separate state directory."""

    jobs: tuple[BatchJob, ...]
    image: str

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.image
        ):
            raise InvalidJobError("Batch image must be an immutable local image ID")
        object.__setattr__(self, "jobs", tuple(self.jobs))
        if not all(isinstance(job, BatchJob) for job in self.jobs):
            raise InvalidJobError("Expected typed batch jobs")
        names = [job.name for job in self.jobs]
        if len(set(names)) != len(names):
            raise InvalidJobError("Duplicate job names")

    @property
    def hash(self) -> str:
        return content_hash({"version": 1, "image": self.image, "jobs": self.jobs})

    @classmethod
    def parse(cls, value: object) -> Batch:
        """Reject unknown fields, invalid limits, and malformed commands up front."""
        if (
            not isinstance(value, Mapping)
            or set(value) != {"version", "image", "jobs"}
            or type(value["version"]) is not int
            or value["version"] != 1
            or not isinstance(value["jobs"], list)
        ):
            raise InvalidJobError("Expected a version-1 batch with a jobs array")
        jobs: list[BatchJob] = []
        for item in value["jobs"]:
            if not isinstance(item, Mapping) or set(item) != {
                "name",
                "argv",
                "wall_seconds",
                "memory_bytes",
            }:
                raise InvalidJobError("Invalid batch job fields")
            argv = item["argv"]
            if not isinstance(argv, list) or not all(
                isinstance(arg, str) for arg in argv
            ):
                raise InvalidJobError("argv must be a string array")
            jobs.append(
                BatchJob(
                    item["name"],
                    tuple(argv),
                    item["wall_seconds"],
                    item["memory_bytes"],
                )
            )
        return cls(tuple(jobs), value["image"])
