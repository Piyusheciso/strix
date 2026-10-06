"""External tool registry that stores full schemas outside the LLM context.

Tools are registered once at agent build time. The LLM never sees their full
schemas — it discovers them through ``search`` and ``get_schema``, then
executes them through ``execute``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agents.tool import FunctionTool


if TYPE_CHECKING:
    from collections.abc import Sequence

    from agents.tool import Tool


logger = logging.getLogger(__name__)


class MetaToolRegistry:
    """Stores full tool schemas outside the LLM context window.

    Provides search, schema lookup, and execution without bloating the prompt.
    Thread-safe for concurrent reads; writes happen once at build time.
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._executors: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool's metadata and executor for JIT access."""
        name = tool.name
        description = getattr(tool, "description", "") or ""
        schema: dict[str, Any] = {}
        if isinstance(tool, FunctionTool):
            schema = tool.params_json_schema or {}
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": schema,
        }
        self._executors[name] = tool

    def register_many(self, tools: Sequence[Tool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    @property
    def tool_names(self) -> list[str]:
        """All registered tool names."""
        return list(self._tools)

    def search(self, query: str) -> list[dict[str, str]]:
        """Return lightweight ``{name, description}`` summaries matching *query*.

        A case-insensitive substring match against name and description.
        Returns all tools when *query* is empty.
        """
        query_lower = query.lower().strip()
        results: list[dict[str, str]] = []
        for name, data in self._tools.items():
            if not query_lower or (
                query_lower in name.lower()
                or query_lower in data["description"].lower()
            ):
                results.append({"name": name, "description": data["description"]})
        return results

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """Return the full JSON schema for a single tool, or ``None``."""
        entry = self._tools.get(name)
        if entry is None:
            return None
        return {
            "name": entry["name"],
            "description": entry["description"],
            "parameters": entry["parameters"],
        }

    def get_executor(self, name: str) -> Tool | None:
        """Return the raw ``Tool`` object for direct invocation."""
        return self._executors.get(name)

    def build_index(self) -> str:
        """Generate a compact bulleted index of all registered tools.

        This conveys discovery capability in < 100 tokens instead of 20,000+.
        """
        if not self._tools:
            return "No tools available."
        lines = ["Available Tools:"]
        for name, data in self._tools.items():
            desc = data["description"]
            # Truncate long descriptions to keep the index compact.
            if len(desc) > 80:
                desc = desc[:77] + "..."
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
