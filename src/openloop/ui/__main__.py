"""Start the local researcher UI without opening or migrating a writable ledger."""

import argparse
import sqlite3
from contextlib import suppress

from openloop.ledger import LedgerError

from .server import create_server


def main() -> None:
    """Serve one explicitly configured ledger on IPv4 loopback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    try:
        server = create_server(args.ledger, port=args.port)
    except (OSError, LedgerError, sqlite3.Error) as error:
        parser.error(str(error))
    print(f"Openloop UI: http://127.0.0.1:{server.server_port}", flush=True)
    with server, suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
