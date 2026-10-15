"""Serve the read-only agent toolbelt over MCP stdio; stdout is protocol-only."""

import argparse
import sqlite3

from openloop.ledger import LedgerError

from .repository import RepositoryError
from .server import create_server
from .service import Toolbelt


def main() -> None:
    """Start stdio tools for the two paths explicitly supplied by the host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="Existing SQLite ledger")
    parser.add_argument("--repository", required=True, help="Git top-level directory")
    arguments = parser.parse_args()
    try:
        toolbelt = Toolbelt(arguments.ledger, arguments.repository)
    except (LedgerError, RepositoryError, OSError, sqlite3.DatabaseError) as error:
        parser.error(str(error))
    create_server(toolbelt).run(transport="stdio")


if __name__ == "__main__":
    main()
