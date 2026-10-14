"""Read-only Python toolbelt. Each call opens its own public Ledger connection.

The host fixes the repository and ledger paths. Agent arguments cannot select
other files, mutate evidence, run jobs, or release held-out measurements.
"""

from pathlib import Path

from openloop.ledger import Ledger, Purpose, Status
from openloop.ledger.validation import check_digest, check_text

from .reports import (
    Report,
    compare_runs,
    is_selection,
    metric_row,
    page,
    validation_metrics,
)
from .repository import RepositoryMap


class Toolbelt:
    """Provide run diffs, validation metric queries, and a committed source map."""

    def __init__(self, ledger_path: str | Path, repository: str | Path) -> None:
        self.ledger_path = Path(ledger_path).resolve(strict=True)
        with Ledger(self.ledger_path, read_only=True):
            pass
        self.repository = RepositoryMap(Path(repository))

    def run_diff(self, baseline: str, candidate: str) -> Report:
        """Compare recorded selection inputs and scores without asserting causation."""
        check_text(baseline)
        check_text(candidate)
        with Ledger(self.ledger_path, read_only=True) as ledger:
            return compare_runs(ledger.get(baseline), ledger.get(candidate))

    def metric_query(
        self,
        metric: str,
        *,
        candidate_hash: str | None = None,
        loop_hash: str | None = None,
        fidelity: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Report:
        """Page original successful validation metrics; cache aliases add no rows.

        Each row retains its scope. This interface does not pool different data,
        fidelities, seeds, evaluators, environments, or simulated executions.
        """
        check_text(metric)
        page((), limit=limit, offset=offset)
        if loop_hash is not None:
            check_digest(loop_hash)
        if fidelity is not None:
            check_text(fidelity)
        with Ledger(self.ledger_path, read_only=True) as ledger:
            runs = ledger.query_all(
                status=Status.SUCCEEDED, candidate_hash=candidate_hash
            )
        rows = [
            metric_row(run, metric, scores[metric])
            for run in runs
            if run.purpose is not Purpose.REPLICATION
            and is_selection(run)
            and (loop_hash is None or run.inputs.loop_hash == loop_hash)
            and (fidelity is None or run.inputs.fidelity == fidelity)
            and metric in (scores := validation_metrics(run))
        ]
        return page(rows, limit=limit, offset=offset)

    def repository_map(
        self, *, prefix: str = "", limit: int = 50, offset: int = 0
    ) -> Report:
        """Page committed paths and Python symbols; never execute project code."""
        return self.repository.query(prefix=prefix, limit=limit, offset=offset)
