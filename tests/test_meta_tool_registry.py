"""Tests for MetaToolRegistry: search, schema lookup, index generation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents.tool import FunctionTool

from strix.tools.meta_tools.registry import MetaToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str, schema: dict[str, Any] | None = None) -> FunctionTool:
    """Create a minimal FunctionTool for testing."""
    return FunctionTool(
        name=name,
        description=description,
        params_json_schema=schema or {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        on_invoke_tool=AsyncMock(return_value="ok"),
        strict_json_schema=False,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_single_tool(self) -> None:
        registry = MetaToolRegistry()
        tool = _make_tool("test_tool", "A test tool")
        registry.register(tool)
        assert registry.tool_count == 1
        assert "test_tool" in registry.tool_names

    def test_register_many(self) -> None:
        registry = MetaToolRegistry()
        tools = [_make_tool(f"tool_{i}", f"Tool {i}") for i in range(5)]
        registry.register_many(tools)
        assert registry.tool_count == 5

    def test_register_overwrites(self) -> None:
        registry = MetaToolRegistry()
        tool1 = _make_tool("same_name", "Version 1")
        tool2 = _make_tool("same_name", "Version 2")
        registry.register(tool1)
        registry.register(tool2)
        assert registry.tool_count == 1
        schema = registry.get_schema("same_name")
        assert schema is not None
        assert schema["description"] == "Version 2"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_by_name(self) -> None:
        registry = MetaToolRegistry()
        registry.register(_make_tool("web_search", "Search the web"))
        registry.register(_make_tool("list_files", "List files"))
        results = registry.search("web")
        assert len(results) == 1
        assert results[0]["name"] == "web_search"

    def test_search_by_description(self) -> None:
        registry = MetaToolRegistry()
        registry.register(_make_tool("scanner", "HTTP vulnerability scanner"))
        results = registry.search("vulnerability")
        assert len(results) == 1
        assert results[0]["name"] == "scanner"

    def test_search_case_insensitive(self) -> None:
        registry = MetaToolRegistry()
        registry.register(_make_tool("MyTool", "Does Something"))
        assert len(registry.search("mytool")) == 1
        assert len(registry.search("MYTOOL")) == 1
        assert len(registry.search("something")) == 1

    def test_search_empty_query_returns_all(self) -> None:
        registry = MetaToolRegistry()
        for i in range(10):
            registry.register(_make_tool(f"tool_{i}", f"Tool {i}"))
        results = registry.search("")
        assert len(results) == 10

    def test_search_no_matches(self) -> None:
        registry = MetaToolRegistry()
        registry.register(_make_tool("foo", "bar"))
        results = registry.search("nonexistent")
        assert results == []

    def test_search_results_are_lightweight(self) -> None:
        """Search results contain only name+description, never full schemas."""
        registry = MetaToolRegistry()
        registry.register(_make_tool("tool", "desc", {"type": "object", "properties": {"x": {"type": "string"}}}))
        results = registry.search("tool")
        assert len(results) == 1
        assert set(results[0].keys()) == {"name", "description"}
        assert "parameters" not in results[0]


# ---------------------------------------------------------------------------
# Schema lookup
# ---------------------------------------------------------------------------


class TestGetSchema:
    def test_get_existing_schema(self) -> None:
        registry = MetaToolRegistry()
        schema = {"type": "object", "properties": {"url": {"type": "string"}}}
        registry.register(_make_tool("probe", "Probe a URL", schema))
        result = registry.get_schema("probe")
        assert result is not None
        assert result["name"] == "probe"
        assert result["parameters"] == schema

    def test_get_missing_schema(self) -> None:
        registry = MetaToolRegistry()
        assert registry.get_schema("missing") is None


# ---------------------------------------------------------------------------
# Executor lookup
# ---------------------------------------------------------------------------


class TestGetExecutor:
    def test_get_existing_executor(self) -> None:
        registry = MetaToolRegistry()
        tool = _make_tool("exec_me", "Execute me")
        registry.register(tool)
        assert registry.get_executor("exec_me") is tool

    def test_get_missing_executor(self) -> None:
        registry = MetaToolRegistry()
        assert registry.get_executor("missing") is None


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_index_format(self) -> None:
        registry = MetaToolRegistry()
        registry.register(_make_tool("web_search", "Search the web"))
        registry.register(_make_tool("shell", "Run shell commands"))
        index = registry.build_index()
        assert index.startswith("Available Tools:")
        assert "- web_search: Search the web" in index
        assert "- shell: Run shell commands" in index

    def test_index_empty_registry(self) -> None:
        registry = MetaToolRegistry()
        assert registry.build_index() == "No tools available."

    def test_index_truncates_long_descriptions(self) -> None:
        registry = MetaToolRegistry()
        long_desc = "A" * 200
        registry.register(_make_tool("verbose", long_desc))
        index = registry.build_index()
        # 77 chars + "..."
        assert "..." in index
        assert len(index.split("- verbose: ")[1]) <= 80
