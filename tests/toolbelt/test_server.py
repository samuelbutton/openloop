"""Use the official MCP client for discovery, calls, errors, and real stdio."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from openloop.ledger import Metric, Result
from openloop.toolbelt import Toolbelt
from openloop.toolbelt.server import create_server

from .conftest import records


def test_mcp_discovery_structured_output_and_errors(toolbelt, ledger, inputs, finish):
    run = finish(ledger.submit(inputs), Result({"loss": Metric(1, "BPB")}))
    hidden = finish(
        ledger.submit(replace(inputs, phase="held_out")),
        Result({"loss": Metric(12345, "BPB", split="held_out")}),
    )

    async def scenario() -> None:
        async with Client(create_server(toolbelt), raise_exceptions=True) as client:
            listing = await client.list_tools()
            assert {tool.name for tool in listing.tools} == {
                "run_diff",
                "metric_query",
                "repository_map",
            }
            for tool in listing.tools:
                assert tool.annotations is not None
                assert tool.annotations.read_only_hint is True
                assert tool.annotations.open_world_hint is False
            query = await client.call_tool("metric_query", {"metric": "loss"})
            assert query.is_error is not True
            assert query.structured_content is not None
            assert query.structured_content["total"] == 1
            diff = await client.call_tool(
                "run_diff", {"baseline": run.id, "candidate": run.id}
            )
            assert diff.is_error is not True
            assert diff.structured_content is not None
            assert diff.structured_content["independent_attempts"] is False
            mapping = await client.call_tool("repository_map", {"prefix": "module.py"})
            assert mapping.is_error is not True
            assert mapping.structured_content is not None
            assert (
                records(mapping.structured_content["items"])[0]["path"] == "module.py"
            )
            for name, arguments in (
                ("run_diff", {"baseline": run.id, "candidate": hidden.id}),
                ("run_diff", {"baseline": run.id, "candidate": "missing"}),
                ("metric_query", {"metric": "loss", "limit": 0}),
                ("metric_query", {"metric": "loss", "limit": True}),
                ("metric_query", {"metric": "loss", "limit": "50"}),
                ("repository_map", {"prefix": "../outside"}),
            ):
                error = await client.call_tool(name, arguments)
                assert error.is_error is True
                assert "12345" not in str(error.content)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_stdio_client_can_initialize_and_call_tools(
    toolbelt: Toolbelt, repository: Path, mode: Literal["auto", "legacy"]
) -> None:
    async def scenario() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "openloop.toolbelt",
                "--ledger",
                str(toolbelt.ledger_path),
                "--repository",
                str(repository),
            ],
        )
        async with Client(parameters, mode=mode) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 3
            result = await client.call_tool("repository_map", {})
            assert result.is_error is not True
            assert result.structured_content is not None
            assert result.structured_content["total"] == 2

    asyncio.run(scenario())
