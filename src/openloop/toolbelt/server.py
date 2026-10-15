"""Official SDK MCP adapter. Stdio only; the host fixes both read-only roots."""

from collections.abc import Callable
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from openloop.ledger import LedgerError

from .reports import MAX_PAGE_SIZE, Report
from .repository import RepositoryError
from .service import Toolbelt

type Limit = Annotated[int, Field(strict=True, ge=1, le=MAX_PAGE_SIZE)]
type Offset = Annotated[int, Field(strict=True, ge=0)]


def _invoke(call: Callable[[], Report]) -> Report:
    try:
        return call()
    except (LedgerError, RepositoryError) as error:
        raise ToolError(str(error)) from error


def create_server(toolbelt: Toolbelt) -> MCPServer:
    """Register exactly three tools with structured output and read-only hints."""
    server = MCPServer(
        "openloop-toolbelt",
        version="0.1.0",
        instructions=(
            "Read-only selection evidence. Deltas are not causal explanations or "
            "verified findings. Held-out results are unavailable. Repository maps "
            "show committed HEAD, not uncommitted changes."
        ),
    )
    annotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @server.tool(annotations=annotations)
    def run_diff(baseline: str, candidate: str) -> dict[str, object]:
        """Compare two selection runs: input changes, scoped deltas and cache links."""
        return _invoke(lambda: toolbelt.run_diff(baseline, candidate))

    @server.tool(annotations=annotations)
    def metric_query(
        metric: str,
        *,
        candidate_hash: str | None = None,
        loop_hash: str | None = None,
        fidelity: str | None = None,
        limit: Limit = 50,
        offset: Offset = 0,
    ) -> dict[str, object]:
        """Query original successful validation scores with scope and pagination."""
        return _invoke(
            lambda: toolbelt.metric_query(
                metric,
                candidate_hash=candidate_hash,
                loop_hash=loop_hash,
                fidelity=fidelity,
                limit=limit,
                offset=offset,
            )
        )

    @server.tool(annotations=annotations)
    def repository_map(
        *, prefix: str = "", limit: Limit = 50, offset: Offset = 0
    ) -> dict[str, object]:
        """Map committed files and Python definitions under a repository prefix."""
        return _invoke(
            lambda: toolbelt.repository_map(prefix=prefix, limit=limit, offset=offset)
        )

    return server
