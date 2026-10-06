"""SDK FunctionTool wrappers for the meta-tool registry.

These are the only tool definitions the LLM sees when meta-tools mode is active.
Together they consume ~300 tokens instead of ~21,000.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agents import RunContextWrapper
from agents.tool import FunctionTool

from strix.tools.meta_tools.registry import MetaToolRegistry


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry accessor — set once at agent build time by the factory.
# ---------------------------------------------------------------------------

_active_registry: MetaToolRegistry | None = None


def set_active_registry(registry: MetaToolRegistry) -> None:
    """Install the meta-tool registry for the current agent lifetime."""
    global _active_registry  # noqa: PLW0603
    _active_registry = registry


def get_active_registry() -> MetaToolRegistry:
    """Return the active registry, raising if none is set."""
    if _active_registry is None:
        msg = "MetaToolRegistry not initialised — call set_active_registry() first"
        raise RuntimeError(msg)
    return _active_registry


# ---------------------------------------------------------------------------
# search_tools
# ---------------------------------------------------------------------------


async def _search_tools_impl(
    ctx: RunContextWrapper[Any],  # noqa: ARG001
    query: str,
) -> str:
    """Search the tool registry and return lightweight name+description matches."""
    registry = get_active_registry()
    results = registry.search(query)
    if not results:
        return json.dumps({"results": [], "message": f"No tools matched '{query}'."})
    return json.dumps({"results": results, "count": len(results)})


search_tools: FunctionTool = FunctionTool(
    name="search_tools",
    description=(
        "Search the available tool registry by keyword. Returns lightweight summaries "
        "(name + description) of matching tools. Use this to discover what tools are "
        "available before calling them. Pass an empty string to list all tools."
    ),
    params_json_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search keyword to match against tool names and descriptions. "
                    "Use empty string to list all available tools."
                ),
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    on_invoke_tool=_search_tools_impl,
    strict_json_schema=True,
)


# ---------------------------------------------------------------------------
# get_tool_schema
# ---------------------------------------------------------------------------


async def _get_tool_schema_impl(
    ctx: RunContextWrapper[Any],  # noqa: ARG001
    name: str,
) -> str:
    """Return the full JSON schema for a single tool."""
    registry = get_active_registry()
    schema = registry.get_schema(name)
    if schema is None:
        return json.dumps({"error": f"Tool '{name}' not found. Use search_tools to discover tools."})
    return json.dumps(schema)


get_tool_schema: FunctionTool = FunctionTool(
    name="get_tool_schema",
    description=(
        "Get the full parameter schema for a specific tool by name. Call this after "
        "search_tools to inspect a tool's arguments before calling it with call_tool."
    ),
    params_json_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact tool name as returned by search_tools.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    on_invoke_tool=_get_tool_schema_impl,
    strict_json_schema=True,
)


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


async def _call_tool_impl(
    ctx: RunContextWrapper[Any],
    name: str,
    arguments: str,
) -> str:
    """Execute a registered tool by name with the given arguments."""
    registry = get_active_registry()
    executor = registry.get_executor(name)
    if executor is None:
        return json.dumps({"error": f"Tool '{name}' not found. Use search_tools to discover tools."})
    try:
        if isinstance(executor, FunctionTool):
            result = await executor.on_invoke_tool(ctx, arguments)
        else:
            result = await executor.on_invoke_tool(ctx, arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.debug("call_tool(%s) failed: %s", name, exc, exc_info=True)
        return json.dumps({"error": f"Tool '{name}' failed: {exc}"})


call_tool: FunctionTool = FunctionTool(
    name="call_tool",
    description=(
        "Execute a tool from the registry by name. First use search_tools to find the "
        "tool, then get_tool_schema to see its parameters, then call this with the "
        "tool name and a JSON string of arguments."
    ),
    params_json_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact tool name to execute.",
            },
            "arguments": {
                "type": "string",
                "description": (
                    "JSON-encoded arguments matching the tool's parameter schema. "
                    "Use get_tool_schema first to see the expected format."
                ),
            },
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    },
    on_invoke_tool=_call_tool_impl,
    strict_json_schema=True,
)


# ---------------------------------------------------------------------------
# Public tuple for easy import in the factory.
# ---------------------------------------------------------------------------

META_TOOLS: tuple[FunctionTool, ...] = (search_tools, get_tool_schema, call_tool)
