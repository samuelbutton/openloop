"""Content-addressed inputs, append-only run evidence, and explicit replication.

Only trusted callers may complete runs. This module records evidence; it does
not execute candidate code, validate an evaluator, or make statistical decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

type JSONValue = (
    None
    | bool
    | int
    | float
    | str
    | Mapping[str, JSONValue]
    | tuple[JSONValue, ...]
    | list[JSONValue]
)
type Purpose = Literal["run", "retry", "replication"]
type Status = Literal["queued", "running", "succeeded", "failed"]
type Stage = Literal[
    "proposal", "implementation", "queue", "execution", "evaluation", "decision"
]

_STAGES = ("proposal", "implementation", "queue", "execution", "evaluation", "decision")
_SCHEMA_VERSION = 1
_SCHEMA = (
    """CREATE TABLE experiment (
        hash TEXT PRIMARY KEY, manifest TEXT NOT NULL
    )""",
    """CREATE TABLE attempt (
        id TEXT PRIMARY KEY, experiment_hash TEXT NOT NULL REFERENCES experiment,
        purpose TEXT NOT NULL CHECK (purpose IN ('run', 'retry', 'replication')),
        retry_of TEXT REFERENCES attempt
    )""",
    """CREATE TABLE submission (
        id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES attempt,
        hypothesis TEXT NOT NULL, created_at TEXT NOT NULL,
        reused_from TEXT REFERENCES submission, request_key TEXT UNIQUE,
        purpose TEXT NOT NULL CHECK (purpose IN ('run', 'retry', 'replication'))
    )""",
    """CREATE TABLE parent (
        child_id TEXT NOT NULL REFERENCES submission,
        parent_id TEXT NOT NULL REFERENCES submission,
        PRIMARY KEY (child_id, parent_id), CHECK (child_id != parent_id)
    )""",
    """CREATE TABLE event (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id TEXT NOT NULL REFERENCES attempt,
        kind TEXT NOT NULL, timestamp TEXT NOT NULL, payload TEXT NOT NULL
    )""",
    """CREATE TABLE probe (
        id TEXT PRIMARY KEY,
        baseline_id TEXT NOT NULL REFERENCES submission,
        repeat_id TEXT NOT NULL REFERENCES submission,
        timestamp TEXT NOT NULL, atol REAL NOT NULL, rtol REAL NOT NULL,
        matches INTEGER NOT NULL CHECK (matches IN (0, 1)), differences TEXT NOT NULL
    )""",
    "CREATE INDEX attempt_inputs ON attempt(experiment_hash)",
    "CREATE INDEX event_attempt ON event(attempt_id, sequence)",
    "CREATE INDEX parent_ancestor ON parent(parent_id)",
    """CREATE VIEW attempt_state AS
        SELECT a.*, CASE
            WHEN last.kind = 'succeeded' THEN 'succeeded'
            WHEN last.kind = 'failed' THEN 'failed'
            WHEN (SELECT payload FROM event WHERE attempt_id = a.id
                  AND kind = 'stage_started' ORDER BY sequence DESC LIMIT 1)
                = '{"stage":"queue"}' THEN 'queued'
            ELSE 'running' END AS status
        FROM attempt a JOIN event last ON last.attempt_id = a.id
            AND last.sequence = (
                SELECT MAX(sequence) FROM event WHERE attempt_id = a.id
            )
    """,
)


def _plain(value: object) -> JSONValue:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Encode declared inputs without key-order or whitespace differences."""
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: JSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Content references must be lowercase SHA-256 digests")


def _text(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty string")


def _positive_integer(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("Expected a positive integer")


@dataclass(frozen=True)
class ExperimentInputs:
    """All declared result-affecting inputs. Hashes refer to complete snapshots."""

    source_hash: str
    dependencies_hash: str
    data_hash: str
    environment_hash: str
    evaluator_hash: str
    loop_hash: str
    seed: int
    fidelity: str
    budget_unit: str
    budget_amount: int
    config: Mapping[str, JSONValue] = field(default_factory=dict)
    execution: Mapping[str, JSONValue] = field(default_factory=dict)
    tokenizer_hash: str | None = None
    stage: str = "screen"

    def __post_init__(self) -> None:
        for name in (
            "source_hash",
            "dependencies_hash",
            "data_hash",
            "environment_hash",
            "evaluator_hash",
            "loop_hash",
        ):
            _digest(getattr(self, name))
        if self.tokenizer_hash is not None:
            _digest(self.tokenizer_hash)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        _positive_integer(self.budget_amount)
        for value in (self.fidelity, self.budget_unit, self.stage):
            _text(value)
        for name in ("config", "execution"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a JSON object")
            canonical_json(value)
            object.__setattr__(self, name, _freeze(value))

    @property
    def manifest(self) -> Mapping[str, JSONValue]:
        return {"schema_version": _SCHEMA_VERSION, "inputs": _plain(self)}

    @property
    def hash(self) -> str:
        return content_hash(self.manifest)


@dataclass(frozen=True)
class Metric:
    value: float
    unit: str
    direction: Literal["minimize", "maximize"] = "minimize"
    split: str = "validation"
    sample_count: int = 1

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("Metric values must be finite numbers")
        object.__setattr__(self, "value", float(self.value))
        _text(self.unit)
        _text(self.split)
        _positive_integer(self.sample_count)
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("Invalid metric direction")


@dataclass(frozen=True)
class Result:
    metrics: Mapping[str, Metric]
    artifacts: Mapping[str, str] = field(default_factory=dict)
    verdict: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping) or not isinstance(
            self.artifacts, Mapping
        ):
            raise ValueError("Metrics and artifacts must be mappings")
        if not self.metrics:
            raise ValueError("A completed result requires at least one metric")
        for name, metric in self.metrics.items():
            _text(name)
            if not isinstance(metric, Metric):
                raise ValueError("Expected typed metrics")
        for role, digest in self.artifacts.items():
            _text(role)
            _digest(digest)
        if self.verdict is not None:
            _text(self.verdict)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


@dataclass(frozen=True)
class Event:
    sequence: int
    kind: str
    timestamp: str
    payload: Mapping[str, JSONValue]


@dataclass(frozen=True)
class Run:
    """One submission; reused submissions reference the same execution attempt."""

    id: str
    attempt_id: str
    inputs: ExperimentInputs
    hypothesis: str
    parents: tuple[str, ...]
    purpose: Purpose
    retry_of: str | None
    created_at: str
    reused_from: str | None
    status: Status
    events: tuple[Event, ...]
    result: Result | None

    @property
    def stages(self) -> Mapping[str, tuple[Event, ...]]:
        return MappingProxyType(
            {
                stage: tuple(
                    event
                    for event in self.events
                    if event.payload.get("stage") == stage
                )
                for stage in _STAGES
            }
        )


@dataclass(frozen=True)
class ProbeResult:
    id: str
    baseline_id: str
    repeat_id: str
    matches: bool
    differences: tuple[str, ...]
    atol: float
    rtol: float


def compare_results(
    baseline: Result,
    repeat: Result,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> tuple[str, ...]:
    """Compare metrics and artifact hashes, excluding timing, costs, and verdicts."""
    for tolerance in (atol, rtol):
        if type(tolerance) not in (int, float):
            raise ValueError("Tolerances must be finite nonnegative numbers")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Tolerances must be finite nonnegative numbers")
    differences = []
    for name in sorted(baseline.metrics.keys() | repeat.metrics.keys()):
        left, right = baseline.metrics.get(name), repeat.metrics.get(name)
        if left is None or right is None:
            differences.append(f"metric {name}: missing or extra")
        elif (left.unit, left.direction, left.split, left.sample_count) != (
            right.unit,
            right.direction,
            right.split,
            right.sample_count,
        ):
            differences.append(f"metric {name}: metadata differs")
        elif abs(left.value - right.value) > atol + rtol * abs(left.value):
            differences.append(f"metric {name}: {left.value!r} -> {right.value!r}")
    for role in sorted(baseline.artifacts.keys() | repeat.artifacts.keys()):
        if baseline.artifacts.get(role) != repeat.artifacts.get(role):
            differences.append(f"artifact {role}: hash differs")
    return tuple(differences)


class Ledger:
    """A SQLite ledger. Use one connection per thread and close it after use."""

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(path, timeout=10, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._ticks: dict[tuple[str, str], float] = {}
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            with self._transaction():
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    if self._db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table'"
                    ).fetchone():
                        raise ValueError("Refusing to initialize an unrelated database")
                    for statement in _SCHEMA:
                        self._db.execute(statement)
                    for table in (
                        "experiment",
                        "attempt",
                        "submission",
                        "parent",
                        "event",
                        "probe",
                    ):
                        for operation in ("UPDATE", "DELETE"):
                            self._db.execute(f"""
                                CREATE TRIGGER {table}_{operation.lower()}
                                BEFORE {operation} ON {table}
                                BEGIN SELECT RAISE(ABORT, 'append-only ledger'); END
                            """)
                    self._db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                elif version != _SCHEMA_VERSION:
                    raise ValueError(f"Unsupported ledger schema version: {version}")
            self._enable_wal()
        except BaseException:
            self._db.close()
            raise

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _enable_wal(self) -> None:
        # Journal-mode changes can return BUSY without honoring connect(timeout).
        deadline = time.monotonic() + 10
        while True:
            try:
                self._db.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as error:
                if (
                    error.sqlite_errorcode != sqlite3.SQLITE_BUSY
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(0.01)

    @contextmanager
    def _transaction(self, *, write: bool = True) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds")

    def _event(self, attempt_id: str, kind: str, payload: object) -> None:
        self._db.execute(
            "INSERT INTO event(attempt_id, kind, timestamp, payload) "
            "VALUES (?, ?, ?, ?)",
            (attempt_id, kind, self._now(), canonical_json(payload)),
        )

    def _get(self, run_id: str) -> Run:
        row = self._db.execute(
            """
            SELECT s.*, a.experiment_hash, a.retry_of, a.status, e.manifest
            FROM submission s JOIN attempt_state a ON a.id = s.attempt_id
            JOIN experiment e ON e.hash = a.experiment_hash WHERE s.id = ?
        """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        events = tuple(
            Event(
                item["sequence"],
                item["kind"],
                item["timestamp"],
                _freeze(json.loads(item["payload"])),
            )
            for item in self._db.execute(
                "SELECT * FROM event WHERE attempt_id = ? ORDER BY sequence",
                (row["attempt_id"],),
            )
        )
        result = None
        if events[-1].kind == "succeeded":
            payload = events[-1].payload
            result = Result(
                metrics={
                    name: Metric(**metric)
                    for name, metric in payload["metrics"].items()
                },
                artifacts=payload["artifacts"],
                verdict=payload["verdict"],
            )
        return Run(
            id=row["id"],
            attempt_id=row["attempt_id"],
            inputs=ExperimentInputs(**json.loads(row["manifest"])["inputs"]),
            hypothesis=row["hypothesis"],
            parents=tuple(
                item[0]
                for item in self._db.execute(
                    "SELECT parent_id FROM parent WHERE child_id = ? "
                    "ORDER BY parent_id",
                    (run_id,),
                )
            ),
            purpose=row["purpose"],
            retry_of=row["retry_of"] if row["purpose"] == "retry" else None,
            created_at=row["created_at"],
            reused_from=row["reused_from"],
            status=row["status"],
            events=events,
            result=result,
        )

    def get(self, run_id: str) -> Run:
        with self._transaction(write=False):
            return self._get(run_id)

    def submit(
        self,
        inputs: ExperimentInputs,
        *,
        hypothesis: str = "",
        parents: tuple[str, ...] = (),
        request_key: str | None = None,
        purpose: Purpose = "run",
        retry_of: str | None = None,
    ) -> Run:
        """Record provenance; reuse successful results or pending normal work.

        Parents are submission IDs. A request key makes submission idempotent.
        Replication always executes again; retries must reference a failed run.
        """
        if purpose not in ("run", "retry", "replication"):
            raise ValueError("Invalid execution purpose")
        if (purpose == "retry") != (retry_of is not None):
            raise ValueError("Only retries require retry_of")
        if not isinstance(hypothesis, str):
            raise ValueError("Hypothesis must be text")
        if request_key is not None:
            _text(request_key)
        parent_ids = tuple(sorted(set(parents) | ({retry_of} if retry_of else set())))
        with self._transaction():
            if request_key is not None:
                existing = self._db.execute(
                    "SELECT id FROM submission WHERE request_key = ?",
                    (request_key,),
                ).fetchone()
                if existing:
                    run = self._get(existing[0])
                    retry_attempt = self._get(retry_of).attempt_id if retry_of else None
                    if (
                        run.inputs.hash,
                        run.hypothesis,
                        run.parents,
                        run.purpose,
                        run.retry_of,
                    ) != (
                        inputs.hash,
                        hypothesis,
                        parent_ids,
                        purpose,
                        retry_attempt,
                    ):
                        raise ValueError("Request key already has different inputs")
                    return run
            for parent in parent_ids:
                self._get(parent)
            retry_attempt = None
            if retry_of is not None:
                previous = self._get(retry_of)
                if previous.status != "failed" or previous.inputs.hash != inputs.hash:
                    raise ValueError(
                        "Retry requires a failed run with identical inputs"
                    )
                retry_attempt = previous.attempt_id
            reuse = None
            if purpose == "run":
                reuse = self._db.execute(
                    """
                    SELECT s.id, a.id AS attempt_id FROM attempt_state a
                    JOIN submission s ON s.attempt_id = a.id AND s.reused_from IS NULL
                    WHERE a.experiment_hash = ? AND a.purpose IN ('run', 'retry') AND (
                        a.status IN ('queued', 'running') OR
                        (a.status = 'succeeded' AND NOT EXISTS (
                            SELECT 1 FROM probe p
                            JOIN submission b ON b.id = p.baseline_id
                            JOIN attempt ba ON ba.id = b.attempt_id
                            WHERE ba.experiment_hash = a.experiment_hash
                                AND p.matches = 0
                        ))
                    ) ORDER BY s.rowid LIMIT 1
                """,
                    (inputs.hash,),
                ).fetchone()
            run_id = uuid4().hex
            attempt_id = reuse["attempt_id"] if reuse else uuid4().hex
            if reuse is None:
                self._db.execute(
                    "INSERT OR IGNORE INTO experiment VALUES (?, ?)",
                    (inputs.hash, canonical_json(inputs.manifest)),
                )
                self._db.execute(
                    "INSERT INTO attempt VALUES (?, ?, ?, ?)",
                    (attempt_id, inputs.hash, purpose, retry_attempt),
                )
                self._event(attempt_id, "stage_started", {"stage": "queue"})
                self._ticks[attempt_id, "queue"] = time.monotonic()
            self._db.execute(
                "INSERT INTO submission VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    attempt_id,
                    hypothesis,
                    self._now(),
                    reuse["id"] if reuse else None,
                    request_key,
                    purpose,
                ),
            )
            self._db.executemany(
                "INSERT INTO parent VALUES (?, ?)",
                [(run_id, parent) for parent in parent_ids],
            )
            return self._get(run_id)

    @staticmethod
    def _writable(run: Run) -> None:
        if run.reused_from is not None:
            raise ValueError("Reused submissions cannot modify an execution attempt")
        if run.status in ("succeeded", "failed"):
            raise ValueError("Run is already terminal")

    def _finish_stage(self, run: Run) -> None:
        starts = [event for event in run.events if event.kind == "stage_started"]
        if not starts:
            return
        start = starts[-1]
        stage = start.payload["stage"]
        tick = self._ticks.get((run.attempt_id, stage))
        duration = (
            time.monotonic() - tick
            if tick is not None
            else (
                datetime.now(UTC) - datetime.fromisoformat(start.timestamp)
            ).total_seconds()
        )
        self._event(
            run.attempt_id,
            "stage_finished",
            {
                "stage": stage,
                "duration_seconds": max(0.0, duration),
                "duration_clock": "monotonic" if tick is not None else "wall",
            },
        )

    def start_stage(self, run_id: str, stage: Stage) -> Run:
        """Close the current stage and start a later stage atomically."""
        if stage not in _STAGES:
            raise ValueError("Unknown stage")
        with self._transaction():
            run = self._get(run_id)
            self._writable(run)
            current = next(
                event.payload["stage"]
                for event in reversed(run.events)
                if event.kind == "stage_started"
            )
            if _STAGES.index(stage) <= _STAGES.index(current):
                raise ValueError("Stages must advance in order")
            self._finish_stage(run)
            self._event(run.attempt_id, "stage_started", {"stage": stage})
            self._ticks[run.attempt_id, stage] = time.monotonic()
            return self._get(run_id)

    def record_preparation(
        self,
        run_id: str,
        stage: Literal["proposal", "implementation"],
        *,
        started_at: datetime,
        finished_at: datetime,
    ) -> Run:
        """Attach actual pre-submission UTC times; never invent earlier stages."""
        if stage not in ("proposal", "implementation"):
            raise ValueError("Only preparation stages accept historical timestamps")
        if started_at.utcoffset() is None or finished_at.utcoffset() is None:
            raise ValueError("Preparation timestamps must have a timezone")
        with self._transaction():
            run = self._get(run_id)
            self._writable(run)
            if started_at > finished_at or finished_at > datetime.fromisoformat(
                run.created_at
            ):
                raise ValueError("Preparation must finish before submission")
            if run.stages[stage]:
                raise ValueError("Preparation stage is already recorded")
            other = run.stages["implementation" if stage == "proposal" else "proposal"]
            if other:
                boundary = datetime.fromisoformat(
                    other[0].payload[
                        "started_at" if stage == "proposal" else "finished_at"
                    ]
                )
                if (stage == "proposal" and finished_at > boundary) or (
                    stage == "implementation" and started_at < boundary
                ):
                    raise ValueError("Preparation stages must not overlap")
            self._event(
                run.attempt_id,
                "preparation",
                {
                    "stage": stage,
                    "started_at": started_at.astimezone(UTC).isoformat(),
                    "finished_at": finished_at.astimezone(UTC).isoformat(),
                    "duration_seconds": (finished_at - started_at).total_seconds(),
                    "duration_clock": "wall",
                },
            )
            return self._get(run_id)

    def complete(self, run_id: str, result: Result) -> Run:
        """Publish a trusted evaluator result and close the active stage."""
        with self._transaction():
            return self._complete(run_id, result)

    def _complete(self, run_id: str, result: Result) -> Run:
        if not isinstance(result, Result):
            raise ValueError("Expected a typed evaluator result")
        run = self._get(run_id)
        self._writable(run)
        if not run.stages["execution"] or not run.stages["evaluation"]:
            raise ValueError("Execution and evaluation stages are required")
        self._finish_stage(run)
        self._event(run.attempt_id, "succeeded", result)
        return self._get(run_id)

    def fail(self, run_id: str, reason: str) -> Run:
        _text(reason)
        with self._transaction():
            run = self._get(run_id)
            self._writable(run)
            self._finish_stage(run)
            self._event(run.attempt_id, "failed", {"reason": reason})
            return self._get(run_id)

    def lineage(self, run_id: str) -> tuple[Run, ...]:
        """Return ancestors before descendants, including cache and retry links."""
        with self._transaction(write=False):
            ordered: dict[str, Run] = {}
            stack = [(run_id, False)]
            while stack:
                current, expanded = stack.pop()
                if current in ordered:
                    continue
                run = self._get(current)
                if expanded:
                    ordered[current] = run
                else:
                    stack.append((current, True))
                    ancestors = run.parents + (
                        (run.reused_from,) if run.reused_from else ()
                    )
                    stack.extend((parent, False) for parent in reversed(ancestors))
            return tuple(ordered.values())

    def query(
        self,
        *,
        status: Status | None = None,
        experiment_hash: str | None = None,
        purpose: Purpose | None = None,
        include_reused: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Run, ...]:
        """Filter runs with stable pagination; aliases are excluded by default."""
        _positive_integer(limit)
        if type(offset) is not int or offset < 0:
            raise ValueError("Offset must be a nonnegative integer")
        if status is not None and status not in (
            "queued",
            "running",
            "succeeded",
            "failed",
        ):
            raise ValueError("Invalid status")
        if purpose is not None and purpose not in ("run", "retry", "replication"):
            raise ValueError("Invalid purpose")
        if experiment_hash is not None:
            _digest(experiment_hash)
        with self._transaction(write=False):
            rows = self._db.execute(
                """
                SELECT s.id FROM submission s
                JOIN attempt_state a ON a.id = s.attempt_id
                WHERE (? OR s.reused_from IS NULL)
                    AND (? IS NULL OR a.experiment_hash = ?)
                    AND (? IS NULL OR s.purpose = ?)
                    AND (? IS NULL OR a.status = ?)
                ORDER BY s.rowid LIMIT ? OFFSET ?
            """,
                (
                    include_reused,
                    experiment_hash,
                    experiment_hash,
                    purpose,
                    purpose,
                    status,
                    status,
                    limit,
                    offset,
                ),
            )
            return tuple(self._get(row[0]) for row in rows.fetchall())

    def probe(
        self,
        run_id: str,
        runner: Callable[[ExperimentInputs], Result],
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ProbeResult:
        """Run one explicit replication and persist measured drift.

        The callback must use these exact inputs and return trusted measurements.
        A mismatch disables completed-result reuse for the experiment hash.
        """
        baseline = self.get(run_id)
        if baseline.status != "succeeded" or baseline.result is None:
            raise ValueError("A probe requires a completed baseline")
        compare_results(baseline.result, baseline.result, atol=atol, rtol=rtol)
        repeat = self.submit(
            baseline.inputs,
            hypothesis="Determinism probe",
            parents=(run_id,),
            purpose="replication",
        )
        self.start_stage(repeat.id, "execution")
        try:
            result = runner(baseline.inputs)
            self.start_stage(repeat.id, "evaluation")
            if not isinstance(result, Result):
                raise ValueError("Expected a typed evaluator result")
            differences = compare_results(baseline.result, result, atol=atol, rtol=rtol)
            report = ProbeResult(
                uuid4().hex,
                baseline.id,
                repeat.id,
                not differences,
                differences,
                float(atol),
                float(rtol),
            )
            with self._transaction():
                self._complete(repeat.id, result)
                self._db.execute(
                    "INSERT INTO probe VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        report.id,
                        report.baseline_id,
                        report.repeat_id,
                        self._now(),
                        report.atol,
                        report.rtol,
                        report.matches,
                        canonical_json(differences),
                    ),
                )
        except Exception as error:
            self.fail(repeat.id, f"{type(error).__name__}: {error}")
            raise
        return report

    def probes(self, run_id: str) -> tuple[ProbeResult, ...]:
        """Read persisted comparisons for a baseline submission."""
        with self._transaction(write=False):
            self._get(run_id)
            return tuple(
                ProbeResult(
                    row["id"],
                    row["baseline_id"],
                    row["repeat_id"],
                    bool(row["matches"]),
                    tuple(json.loads(row["differences"])),
                    row["atol"],
                    row["rtol"],
                )
                for row in self._db.execute(
                    "SELECT * FROM probe WHERE baseline_id = ? ORDER BY timestamp, id",
                    (run_id,),
                )
            )
