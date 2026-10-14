"""SQLite plugin report adapter. Ledger owns transactions and observed times."""

import json
import sqlite3

from .derivations import Derivation
from .identity import canonical_json
from .validation import check_digest, check_text


def _read(row: sqlite3.Row) -> Derivation:
    return Derivation(
        row["namespace"],
        row["subject_hash"],
        json.loads(row["payload"]),
        row["timestamp"],
    )


def append_derivation(db: sqlite3.Connection, record: Derivation) -> Derivation:
    db.execute(
        "INSERT INTO derivation "
        "(hash, namespace, subject_hash, payload, timestamp) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(hash) DO NOTHING",
        (
            record.hash,
            record.namespace,
            record.subject_hash,
            canonical_json(record.payload),
            record.timestamp,
        ),
    )
    return _read(
        db.execute("SELECT * FROM derivation WHERE hash = ?", (record.hash,)).fetchone()
    )


def read_derivations(
    db: sqlite3.Connection, namespace: str, subject_hash: str | None
) -> tuple[Derivation, ...]:
    check_text(namespace)
    if subject_hash is not None:
        check_digest(subject_hash)
    rows = db.execute(
        "SELECT * FROM derivation WHERE namespace = ? "
        "AND (? IS NULL OR subject_hash = ?) ORDER BY rowid",
        (namespace, subject_hash, subject_hash),
    )
    return tuple(_read(row) for row in rows.fetchall())
