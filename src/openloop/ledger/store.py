"""SQLite adapter: atomic admission, stage events, results, and probes."""

# Runtime isinstance checks validate untyped callers at the trust boundary.
# pyright: reportUnnecessaryIsInstance=false

import json
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Self
from uuid import uuid4

from .errors import (
    ConflictError,
    InvalidInputError,
    InvalidTransitionError,
    NotFoundError,
)
from .identity import (
    ExperimentInputs,
    Reproducibility,
    canonical_json,
    check_digest,
    check_positive,
    check_text,
    coerce_enum,
    freeze_object,
)
from .records import (
    Event,
    EventKind,
    Metric,
    Preparation,
    PreparationStage,
    ProbeResult,
    Purpose,
    Result,
    Run,
    Stage,
    Status,
    compare_results,
)
from .schema import ensure_schema

_STAGE_ORDER = tuple(Stage)


class Ledger:
    """A SQLite ledger. Use one connection per thread and close it after use."""

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(path, timeout=10, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        # Monotonic stage starts for this connection; dropped when a stage ends.
        self._ticks: dict[tuple[str, Stage], float] = {}
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            with self._transaction():
                ensure_schema(self._db)
            self._enable_wal()
        except BaseException:
            self._db.close()
            raise

    def __enter__(self) -> Self:
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
    def _transaction(self, *, write: bool = True) -> Generator[None]:
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

    def _event(
        self,
        attempt_id: str,
        kind: EventKind,
        payload: object,
        *,
        stage: Stage | None = None,
    ) -> None:
        self._db.execute(
            "INSERT INTO event(attempt_id, kind, stage, timestamp, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (attempt_id, kind, stage, self._now(), canonical_json(payload)),
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
            raise NotFoundError(f"Unknown run: {run_id}")
        event_rows = self._db.execute(
            "SELECT * FROM event WHERE attempt_id = ? ORDER BY sequence",
            (row["attempt_id"],),
        ).fetchall()
        events = tuple(
            Event(
                sequence=item["sequence"],
                kind=EventKind(item["kind"]),
                timestamp=item["timestamp"],
                stage=Stage(item["stage"]) if item["stage"] else None,
                payload=freeze_object(json.loads(item["payload"])),
            )
            for item in event_rows
        )
        result = None
        if events[-1].kind is EventKind.SUCCEEDED:
            payload = json.loads(event_rows[-1]["payload"])
            result = Result(
                metrics={
                    name: Metric(**metric)
                    for name, metric in payload["metrics"].items()
                },
                artifacts=payload["artifacts"],
            )
        purpose = Purpose(row["purpose"])
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
            purpose=purpose,
            retry_of=row["retry_of"] if purpose is Purpose.RETRY else None,
            created_at=row["created_at"],
            reused_from=row["reused_from"],
            status=Status(row["status"]),
            preparations=tuple(
                Preparation(
                    stage=PreparationStage(item["stage"]),
                    started_at=item["started_at"],
                    finished_at=item["finished_at"],
                    duration_seconds=item["duration_seconds"],
                )
                for item in self._db.execute(
                    "SELECT * FROM preparation WHERE submission_id = ? "
                    "ORDER BY started_at, stage",
                    (run_id,),
                )
            ),
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
        purpose: Purpose | str = Purpose.RUN,
        retry_of: str | None = None,
    ) -> Run:
        """Record provenance; reuse successful results or pending normal work.

        Parents are submission IDs. A request key makes submission idempotent.
        Replication always executes again; retries must reference a failed run.
        """
        purpose = coerce_enum(Purpose, purpose, "purpose")
        if (purpose is Purpose.RETRY) != (retry_of is not None):
            raise InvalidInputError("Only retries require retry_of")
        if not isinstance(hypothesis, str):
            raise InvalidInputError("Hypothesis must be text")
        if request_key is not None:
            check_text(request_key)
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
                        raise ConflictError(
                            "Request key already has different inputs or provenance"
                        )
                    return run
            for parent in parent_ids:
                self._get(parent)
            retry_attempt = None
            if retry_of is not None:
                previous = self._get(retry_of)
                if previous.status is not Status.FAILED:
                    raise InvalidTransitionError("Retry requires a failed run")
                if previous.inputs.hash != inputs.hash:
                    raise InvalidInputError("Retry requires identical inputs")
                retry_attempt = previous.attempt_id
            reuse = None
            if purpose is Purpose.RUN:
                reuse = self._db.execute(
                    """
                    SELECT s.id, a.id AS attempt_id FROM attempt_state a
                    JOIN submission s ON s.attempt_id = a.id AND s.reused_from IS NULL
                    WHERE a.experiment_hash = :hash
                        AND a.purpose IN (:run, :retry) AND (
                        a.status IN (:queued, :running) OR
                        (a.status = :succeeded AND NOT EXISTS (
                            SELECT 1 FROM probe p
                            JOIN submission b ON b.id = p.baseline_id
                            JOIN attempt ba ON ba.id = b.attempt_id
                            WHERE ba.experiment_hash = a.experiment_hash
                                AND p.invalidates_cache = 1
                        ))
                    ) ORDER BY s.rowid LIMIT 1
                """,
                    {
                        "hash": inputs.hash,
                        "run": Purpose.RUN,
                        "retry": Purpose.RETRY,
                        "queued": Status.QUEUED,
                        "running": Status.RUNNING,
                        "succeeded": Status.SUCCEEDED,
                    },
                ).fetchone()
            run_id = uuid4().hex
            attempt_id = reuse["attempt_id"] if reuse else uuid4().hex
            if reuse is None:
                self._db.execute(
                    "INSERT OR IGNORE INTO experiment(hash, candidate_hash, manifest) "
                    "VALUES (?, ?, ?)",
                    (
                        inputs.hash,
                        inputs.candidate_hash,
                        canonical_json(inputs.manifest),
                    ),
                )
                self._db.execute(
                    "INSERT INTO attempt(id, experiment_hash, purpose, retry_of) "
                    "VALUES (?, ?, ?, ?)",
                    (attempt_id, inputs.hash, purpose, retry_attempt),
                )
                self._event(attempt_id, EventKind.STAGE_STARTED, {}, stage=Stage.QUEUE)
            self._db.execute(
                "INSERT INTO submission(id, attempt_id, hypothesis, created_at, "
                "reused_from, request_key, purpose) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
                "INSERT INTO parent(child_id, parent_id) VALUES (?, ?)",
                [(run_id, parent) for parent in parent_ids],
            )
            run = self._get(run_id)
        if reuse is None:
            self._ticks[attempt_id, Stage.QUEUE] = time.monotonic()
        return run

    @staticmethod
    def _writable(run: Run) -> None:
        if run.reused_from is not None:
            raise InvalidTransitionError(
                "Reused submissions cannot modify an execution attempt"
            )
        if run.status in (Status.SUCCEEDED, Status.FAILED):
            raise InvalidTransitionError("Run is already terminal")

    @staticmethod
    def _open_stage(run: Run) -> Stage:
        return next(
            event.stage
            for event in reversed(run.events)
            if event.kind is EventKind.STAGE_STARTED and event.stage is not None
        )

    def _finish_stage(self, run: Run) -> None:
        stage = self._open_stage(run)
        start = next(
            event
            for event in reversed(run.events)
            if event.kind is EventKind.STAGE_STARTED
        )
        tick = self._ticks.pop((run.attempt_id, stage), None)
        duration = (
            time.monotonic() - tick
            if tick is not None
            else (
                datetime.now(UTC) - datetime.fromisoformat(start.timestamp)
            ).total_seconds()
        )
        self._event(
            run.attempt_id,
            EventKind.STAGE_FINISHED,
            {
                "duration_seconds": max(0.0, duration),
                "duration_clock": "monotonic" if tick is not None else "wall",
            },
            stage=stage,
        )

    def start_stage(self, run_id: str, stage: Stage | str) -> Run:
        """Close the current stage and start a later stage atomically."""
        stage = coerce_enum(Stage, stage, "stage")
        with self._transaction():
            run = self._get(run_id)
            self._writable(run)
            if _STAGE_ORDER.index(stage) <= _STAGE_ORDER.index(self._open_stage(run)):
                raise InvalidTransitionError("Stages must advance in order")
            self._finish_stage(run)
            self._event(run.attempt_id, EventKind.STAGE_STARTED, {}, stage=stage)
            updated = self._get(run_id)
        self._ticks[run.attempt_id, stage] = time.monotonic()
        return updated

    def record_preparation(
        self,
        run_id: str,
        stage: PreparationStage | str,
        *,
        started_at: datetime,
        finished_at: datetime,
    ) -> Run:
        """Attach actual pre-submission UTC times to a submission.

        Cache-hit submissions record their own preparation. The ledger never
        invents earlier stages.
        """
        stage = coerce_enum(PreparationStage, stage, "preparation stage")
        if started_at.utcoffset() is None or finished_at.utcoffset() is None:
            raise InvalidInputError("Preparation timestamps must have a timezone")
        with self._transaction():
            run = self._get(run_id)
            if started_at > finished_at or finished_at > datetime.fromisoformat(
                run.created_at
            ):
                raise InvalidInputError("Preparation must finish before submission")
            recorded = {item.stage: item for item in run.preparations}
            if stage in recorded:
                raise ConflictError("Preparation stage is already recorded")
            proposal = recorded.get(PreparationStage.PROPOSAL)
            implementation = recorded.get(PreparationStage.IMPLEMENTATION)
            if (
                stage is PreparationStage.PROPOSAL
                and implementation
                and finished_at > datetime.fromisoformat(implementation.started_at)
            ) or (
                stage is PreparationStage.IMPLEMENTATION
                and proposal
                and started_at < datetime.fromisoformat(proposal.finished_at)
            ):
                raise InvalidInputError("Preparation stages must not overlap")
            self._db.execute(
                "INSERT INTO preparation(submission_id, stage, started_at, "
                "finished_at, duration_seconds) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    stage,
                    started_at.astimezone(UTC).isoformat(timespec="microseconds"),
                    finished_at.astimezone(UTC).isoformat(timespec="microseconds"),
                    (finished_at - started_at).total_seconds(),
                ),
            )
            return self._get(run_id)

    def complete(self, run_id: str, result: Result) -> Run:
        """Publish a trusted evaluator result and close the active stage."""
        with self._transaction():
            return self._complete(run_id, result)

    def _complete(self, run_id: str, result: Result) -> Run:
        if not isinstance(result, Result):
            raise InvalidInputError("Expected a typed evaluator result")
        run = self._get(run_id)
        self._writable(run)
        if not run.stages[Stage.EXECUTION] or not run.stages[Stage.EVALUATION]:
            raise InvalidTransitionError("Execution and evaluation stages are required")
        self._finish_stage(run)
        self._event(run.attempt_id, EventKind.SUCCEEDED, result)
        return self._get(run_id)

    def fail(self, run_id: str, reason: str) -> Run:
        check_text(reason)
        with self._transaction():
            run = self._get(run_id)
            self._writable(run)
            self._finish_stage(run)
            self._event(run.attempt_id, EventKind.FAILED, {"reason": reason})
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
        status: Status | str | None = None,
        experiment_hash: str | None = None,
        candidate_hash: str | None = None,
        purpose: Purpose | str | None = None,
        include_reused: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Run, ...]:
        """Filter runs with stable pagination; aliases are excluded by default."""
        check_positive(limit)
        if type(offset) is not int or offset < 0:
            raise InvalidInputError("Offset must be a nonnegative integer")
        if status is not None:
            status = coerce_enum(Status, status, "status")
        if purpose is not None:
            purpose = coerce_enum(Purpose, purpose, "purpose")
        for digest in (experiment_hash, candidate_hash):
            if digest is not None:
                check_digest(digest)
        with self._transaction(write=False):
            rows = self._db.execute(
                """
                SELECT s.id FROM submission s
                JOIN attempt_state a ON a.id = s.attempt_id
                JOIN experiment e ON e.hash = a.experiment_hash
                WHERE (:reused OR s.reused_from IS NULL)
                    AND (:hash IS NULL OR a.experiment_hash = :hash)
                    AND (:candidate IS NULL OR e.candidate_hash = :candidate)
                    AND (:purpose IS NULL OR s.purpose = :purpose)
                    AND (:status IS NULL OR a.status = :status)
                ORDER BY s.rowid LIMIT :limit OFFSET :offset
            """,
                {
                    "reused": include_reused,
                    "hash": experiment_hash,
                    "candidate": candidate_hash,
                    "purpose": purpose,
                    "status": status,
                    "limit": limit,
                    "offset": offset,
                },
            )
            return tuple(self._get(row[0]) for row in rows.fetchall())

    def record_probe(
        self,
        baseline_id: str,
        repeat_id: str,
        result: Result,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ProbeResult:
        """Complete a replication and persist its comparison in one transaction.

        A mismatch for deterministic inputs disables completed-result reuse.
        For noisy inputs it is recorded as variation and reuse continues.
        """
        if not isinstance(result, Result):
            raise InvalidInputError("Expected a typed evaluator result")
        with self._transaction():
            baseline = self._get(baseline_id)
            if baseline.status is not Status.SUCCEEDED or baseline.result is None:
                raise InvalidTransitionError("A probe requires a completed baseline")
            repeat = self._get(repeat_id)
            if repeat.purpose is not Purpose.REPLICATION:
                raise InvalidInputError("A probe repeat must be a replication")
            if baseline_id not in repeat.parents:
                raise InvalidInputError(
                    "A probe repeat must have the baseline as parent"
                )
            if repeat.inputs.hash != baseline.inputs.hash:
                raise InvalidInputError("A probe repeat must use the baseline inputs")
            differences = compare_results(baseline.result, result, atol=atol, rtol=rtol)
            self._complete(repeat_id, result)
            report = ProbeResult(
                id=uuid4().hex,
                baseline_id=baseline_id,
                repeat_id=repeat_id,
                matches=not differences,
                differences=differences,
                atol=float(atol),
                rtol=float(rtol),
                invalidates_cache=bool(differences)
                and baseline.inputs.reproducibility is Reproducibility.DETERMINISTIC,
            )
            self._db.execute(
                "INSERT INTO probe(id, baseline_id, repeat_id, timestamp, atol, rtol, "
                "matches, invalidates_cache, differences) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.id,
                    baseline_id,
                    repeat_id,
                    self._now(),
                    report.atol,
                    report.rtol,
                    report.matches,
                    report.invalidates_cache,
                    canonical_json(differences),
                ),
            )
            return report

    def probes(self, run_id: str) -> tuple[ProbeResult, ...]:
        """Read persisted comparisons for a baseline submission."""
        with self._transaction(write=False):
            self._get(run_id)
            return tuple(
                ProbeResult(
                    id=row["id"],
                    baseline_id=row["baseline_id"],
                    repeat_id=row["repeat_id"],
                    matches=bool(row["matches"]),
                    differences=tuple(json.loads(row["differences"])),
                    atol=row["atol"],
                    rtol=row["rtol"],
                    invalidates_cache=bool(row["invalidates_cache"]),
                )
                for row in self._db.execute(
                    "SELECT * FROM probe WHERE baseline_id = ? ORDER BY timestamp, id",
                    (run_id,),
                )
            )
