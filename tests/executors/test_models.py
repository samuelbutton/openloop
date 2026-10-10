"""Executor value validation and input identity; no processes are started."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openloop.executors import CommitSource, InvalidJobError, Limits, ProcessJob


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), "1"])
def test_invalid_limits(value) -> None:
    with pytest.raises(InvalidJobError):
        Limits(wall_seconds=value)


@pytest.mark.parametrize("name", ["queue_seconds", "cpu_seconds", "file_bytes"])
def test_optional_limits_default_to_none_and_reject_bad_values(name: str) -> None:
    assert getattr(Limits(), name) is None
    for value in (0, -1, True, float("nan")):
        with pytest.raises(InvalidJobError):
            Limits(**{name: value})  # type: ignore[arg-type]
    with pytest.raises(InvalidJobError):
        Limits(file_bytes=1.5)  # type: ignore[arg-type]


def test_limits_enter_identity() -> None:
    job = ProcessJob(("true",))
    for limits in (
        Limits(queue_seconds=1),
        Limits(cpu_seconds=1),
        Limits(file_bytes=1),
        Limits(outputs_bytes=1),
    ):
        assert replace(job, limits=limits).hash != job.hash


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../b", "a//b", ".", "a\0b"]
)
def test_invalid_input_paths(name: str) -> None:
    with pytest.raises(InvalidJobError):
        ProcessJob(("true",), files={name: b"data"})


def test_inputs_are_frozen_snapshots() -> None:
    environment = {"PATH": "/bin"}
    files = {"input": b"before"}
    job = ProcessJob(("true",), environment=environment, files=files)
    original = job.hash
    environment["PATH"] = "/other"
    files["input"] = b"after"
    assert job.hash == original
    with pytest.raises(TypeError):
        job.files["input"] = b"mutation"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        job.gpu = True  # type: ignore[misc]


@given(st.binary(max_size=100), st.binary(max_size=100))
def test_byte_inputs_enter_identity(left: bytes, right: bytes) -> None:
    job = ProcessJob(("true",), stdin=left)
    other = replace(job, stdin=right)
    assert (job.hash == other.hash) == (left == right)


def test_policy_enters_identity() -> None:
    job = ProcessJob(("true",))
    assert replace(job, gpu=True).hash != job.hash
    assert replace(job, limits=Limits(memory_bytes=4096)).hash != job.hash


def test_moving_source_refs_and_file_directory_collisions_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(InvalidJobError):
        CommitSource(tmp_path, "main")
    with pytest.raises(InvalidJobError):
        ProcessJob(("true",), files={"a": b"", "a/b": b""})
