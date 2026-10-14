"""SQLite activity operations. Ledger owns their transactions and observed times."""

import json
import sqlite3
from collections.abc import Mapping

from .activities import (
    MAX_IMPLEMENTATION_ATTEMPTS,
    Activity,
    ActivityKind,
    HypothesisOutcome,
)
from .errors import ConflictError, InvalidInputError, InvalidTransitionError
from .identity import canonical_json
from .validation import check_digest, check_text, coerce_enum, plain


def _activity(row: sqlite3.Row) -> Activity:
    payload = plain(json.loads(row["payload"]))
    if not isinstance(payload, dict):
        # plain returns mappings as dicts; the schema stores objects only.
        raise InvalidInputError("Activity payload must be an object")
    return Activity(
        row["sequence"],
        ActivityKind(row["kind"]),
        row["subject_hash"],
        row["timestamp"],
        payload,
        parent_sequence=row["parent_sequence"],
    )


def read_activities(
    db: sqlite3.Connection, subject_hash: str | None
) -> tuple[Activity, ...]:
    if subject_hash is not None:
        check_digest(subject_hash)
        rows = db.execute(
            "SELECT * FROM activity WHERE subject_hash = ? ORDER BY sequence",
            (subject_hash,),
        )
    else:
        rows = db.execute("SELECT * FROM activity ORDER BY sequence")
    return tuple(_activity(row) for row in rows.fetchall())


def append_activity(
    db: sqlite3.Connection,
    kind: ActivityKind | str,
    subject_hash: str,
    payload: object,
    *,
    timestamp: str,
    key: str | None,
    parent_sequence: int | None,
) -> Activity:
    kind = coerce_enum(ActivityKind, kind, "activity kind")
    check_digest(subject_hash)
    body = plain(payload)
    if not isinstance(body, dict):
        raise InvalidInputError("Activity payload must be an object")
    if key is not None:
        check_text(key)
        prior = db.execute(
            "SELECT * FROM activity WHERE request_key = ?", (key,)
        ).fetchone()
        if prior is not None:
            old = _activity(prior)
            comparable = dict(old.payload)
            if kind is ActivityKind.IMPLEMENTATION_STARTED:
                comparable.pop("attempt", None)
            if (
                old.kind,
                old.subject_hash,
                canonical_json(comparable),
                old.parent_sequence,
            ) != (
                kind,
                subject_hash,
                canonical_json(body),
                parent_sequence,
            ):
                raise ConflictError(
                    "Activity key already identifies a different record"
                )
            return old
    history = read_activities(db, subject_hash)
    starts = tuple(
        item for item in history if item.kind is ActivityKind.IMPLEMENTATION_STARTED
    )
    endings = tuple(
        item
        for item in history
        if item.kind
        in (ActivityKind.IMPLEMENTATION_FAILURE, ActivityKind.IMPLEMENTATION_SUCCESS)
    )
    if kind is ActivityKind.IMPLEMENTATION_STARTED:
        if "attempt" in body:
            raise InvalidInputError(
                "Implementation attempt numbers are ledger-assigned"
            )
        if not any(item.kind is ActivityKind.HYPOTHESIS for item in history):
            raise InvalidTransitionError(
                "Implementation requires a recorded hypothesis"
            )
        if any(item.kind is ActivityKind.IMPLEMENTATION_SUCCESS for item in endings):
            raise InvalidTransitionError("Hypothesis is already implemented")
        if starts and (
            not endings or endings[-1].parent_sequence != starts[-1].sequence
        ):
            raise InvalidTransitionError(
                "Previous implementation attempt is unresolved"
            )
        if len(starts) >= MAX_IMPLEMENTATION_ATTEMPTS:
            raise InvalidTransitionError("Implementation retry limit reached")
        if starts:
            prior_plan = dict(starts[0].payload)
            prior_plan.pop("attempt", None)
            if canonical_json(prior_plan) != canonical_json(body):
                raise ConflictError("Implementation source and smoke policy are frozen")
        body["attempt"] = len(starts) + 1
    if kind in (
        ActivityKind.IMPLEMENTATION_FAILURE,
        ActivityKind.IMPLEMENTATION_SUCCESS,
    ) and (
        not starts
        or parent_sequence != starts[-1].sequence
        or any(item.parent_sequence == parent_sequence for item in endings)
    ):
        raise InvalidTransitionError("An implementation ends its current attempt once")
    parent = next((item for item in history if item.sequence == parent_sequence), None)
    if parent_sequence is not None and parent is None:
        raise InvalidInputError("Activity parent must belong to the same subject")
    _check_structure(kind, body, parent, history)
    cursor = db.execute(
        "INSERT INTO activity(kind, subject_hash, timestamp, payload, "
        "request_key, parent_sequence) VALUES (?, ?, ?, ?, ?, ?)",
        (kind, subject_hash, timestamp, canonical_json(body), key, parent_sequence),
    )
    if cursor.lastrowid is None:
        raise InvalidInputError("Activity sequence was not assigned")
    return Activity(
        cursor.lastrowid,
        kind,
        subject_hash,
        timestamp,
        body,
        parent_sequence=parent_sequence,
    )


def _check_structure(
    kind: ActivityKind,
    body: Mapping[str, object],
    parent: Activity | None,
    history: tuple[Activity, ...],
) -> None:
    """Enforce record links that are structural; limits and prices are policy."""
    if kind is ActivityKind.HYPOTHESIS_OUTCOME:
        if parent is None or parent.kind is not ActivityKind.HYPOTHESIS:
            raise InvalidTransitionError(
                "An outcome must reference its hypothesis record"
            )
        outcome = coerce_enum(HypothesisOutcome, body.get("outcome"), "outcome")
        if outcome is not HypothesisOutcome.SCREEN_DROPPED or "decision_hash" in body:
            check_digest(body.get("decision_hash"))
        if any(
            item.kind is ActivityKind.HYPOTHESIS_OUTCOME
            and item.parent_sequence == parent.sequence
            for item in history
        ):
            raise InvalidTransitionError("A hypothesis has one recorded outcome")
    elif kind is ActivityKind.SPEND_RESERVATION:
        if parent is not None:
            raise InvalidTransitionError("A spend reservation has no parent")
    elif kind is ActivityKind.SPEND_SETTLEMENT:
        if parent is None or parent.kind is not ActivityKind.SPEND_RESERVATION:
            raise InvalidTransitionError("A settlement must reference a reservation")
        if any(
            item.kind is ActivityKind.SPEND_SETTLEMENT
            and item.parent_sequence == parent.sequence
            for item in history
        ):
            raise InvalidTransitionError("A reservation is settled once")
