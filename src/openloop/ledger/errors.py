"""Ledger exceptions."""


class LedgerError(Exception):
    """Base class for ledger failures."""


class InvalidInputError(LedgerError, ValueError):
    """A value is malformed or outside its allowed set."""


class NotFoundError(LedgerError, LookupError):
    """A referenced run does not exist."""


class ConflictError(LedgerError):
    """A request conflicts with evidence that is already recorded."""


class InvalidTransitionError(LedgerError):
    """The run's state does not permit the requested write."""


class SchemaError(LedgerError):
    """The database is unrelated or has an unsupported schema version."""
