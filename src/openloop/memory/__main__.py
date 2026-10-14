"""Read research findings as JSON. Never migrate or write the supplied ledger."""

import argparse

from openloop.ledger import SCHEMA_VERSION, Ledger, canonical_json

from .store import ResearchMemory


def main() -> None:
    """Print all findings matching optional candidate, protocol, and loop hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger")
    parser.add_argument("--candidate-hash")
    parser.add_argument("--protocol-hash")
    parser.add_argument("--loop-hash")
    arguments = parser.parse_args()
    with Ledger(arguments.ledger, read_only=True) as ledger:
        if ledger.schema_version != SCHEMA_VERSION:
            raise SystemExit(
                f"{arguments.ledger} uses ledger schema {ledger.schema_version}; "
                f"research memory needs schema {SCHEMA_VERSION}. This command never "
                "migrates a ledger. Copy the file, open the copy with "
                "Ledger(path, migrate=True), and read the copy. Never migrate "
                "frozen evidence."
            )
        findings = ResearchMemory(ledger).query(
            candidate_hash=arguments.candidate_hash,
            protocol_hash=arguments.protocol_hash,
            loop_hash=arguments.loop_hash,
        )
        print(
            canonical_json([{"hash": item.hash, "record": item} for item in findings])
        )


if __name__ == "__main__":
    main()
