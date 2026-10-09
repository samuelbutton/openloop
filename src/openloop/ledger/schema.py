"""SQLite schema, append-only triggers, and version checks.

The storage version is independent of the input identity version.
"""

import sqlite3
from enum import StrEnum

from .errors import SchemaError
from .records import EventKind, PreparationStage, Purpose, Stage, Status

SCHEMA_VERSION = 2
TABLES = (
    "experiment",
    "attempt",
    "submission",
    "parent",
    "event",
    "probe",
    "preparation",
)


def _in(column: str, members: type[StrEnum]) -> str:
    values = ", ".join(f"'{member}'" for member in members)
    return f"{column} IN ({values})"


_DDL = (
    """CREATE TABLE experiment (
        hash TEXT PRIMARY KEY, candidate_hash TEXT NOT NULL, manifest TEXT NOT NULL
    )""",
    f"""CREATE TABLE attempt (
        id TEXT PRIMARY KEY, experiment_hash TEXT NOT NULL REFERENCES experiment,
        purpose TEXT NOT NULL CHECK ({_in("purpose", Purpose)}),
        retry_of TEXT REFERENCES attempt
    )""",
    f"""CREATE TABLE submission (
        id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES attempt,
        hypothesis TEXT NOT NULL, created_at TEXT NOT NULL,
        reused_from TEXT REFERENCES submission, request_key TEXT UNIQUE,
        purpose TEXT NOT NULL CHECK ({_in("purpose", Purpose)})
    )""",
    """CREATE TABLE parent (
        child_id TEXT NOT NULL REFERENCES submission,
        parent_id TEXT NOT NULL REFERENCES submission,
        PRIMARY KEY (child_id, parent_id), CHECK (child_id != parent_id)
    )""",
    f"""CREATE TABLE event (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id TEXT NOT NULL REFERENCES attempt,
        kind TEXT NOT NULL CHECK ({_in("kind", EventKind)}),
        stage TEXT CHECK ({_in("stage", Stage)}),
        timestamp TEXT NOT NULL, payload TEXT NOT NULL
    )""",
    f"""CREATE TABLE preparation (
        submission_id TEXT NOT NULL REFERENCES submission,
        stage TEXT NOT NULL CHECK ({_in("stage", PreparationStage)}),
        started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
        duration_seconds REAL NOT NULL,
        PRIMARY KEY (submission_id, stage)
    )""",
    """CREATE TABLE probe (
        id TEXT PRIMARY KEY,
        baseline_id TEXT NOT NULL REFERENCES submission,
        repeat_id TEXT NOT NULL REFERENCES submission,
        timestamp TEXT NOT NULL, atol REAL NOT NULL, rtol REAL NOT NULL,
        matches INTEGER NOT NULL CHECK (matches IN (0, 1)),
        invalidates_cache INTEGER NOT NULL CHECK (invalidates_cache IN (0, 1)),
        differences TEXT NOT NULL
    )""",
    "CREATE INDEX experiment_candidate ON experiment(candidate_hash)",
    "CREATE INDEX attempt_inputs ON attempt(experiment_hash)",
    "CREATE INDEX event_attempt ON event(attempt_id, sequence)",
    "CREATE INDEX parent_ancestor ON parent(parent_id)",
    f"""CREATE VIEW attempt_state AS
        SELECT a.*, CASE
            WHEN last.kind = '{EventKind.SUCCEEDED}' THEN '{Status.SUCCEEDED}'
            WHEN last.kind = '{EventKind.FAILED}' THEN '{Status.FAILED}'
            WHEN (SELECT stage FROM event WHERE attempt_id = a.id
                  AND kind = '{EventKind.STAGE_STARTED}'
                  ORDER BY sequence DESC LIMIT 1) = '{Stage.QUEUE}'
                THEN '{Status.QUEUED}'
            ELSE '{Status.RUNNING}' END AS status
        FROM attempt a JOIN event last ON last.attempt_id = a.id
            AND last.sequence = (
                SELECT MAX(sequence) FROM event WHERE attempt_id = a.id
            )
    """,
)


def _create(db: sqlite3.Connection) -> None:
    for statement in _DDL:
        db.execute(statement)
    for table in TABLES:
        for operation in ("UPDATE", "DELETE"):
            db.execute(f"""
                CREATE TRIGGER {table}_{operation.lower()}
                BEFORE {operation} ON {table}
                BEGIN SELECT RAISE(ABORT, 'append-only ledger'); END
            """)
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def ensure_schema(db: sqlite3.Connection) -> None:
    """Create an empty database or validate an existing one.

    Call inside a write transaction.
    """
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'").fetchone():
            raise SchemaError("Refusing to initialize an unrelated database")
        _create(db)
    elif version != SCHEMA_VERSION:
        # A future migration would upgrade `version` step by step here.
        raise SchemaError(
            f"Unsupported ledger schema version {version}; "
            f"this release reads version {SCHEMA_VERSION} and has no migration"
        )
