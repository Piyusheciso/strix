"""Tests for agent factory meta-tools mode."""

from __future__ import annotations

import pytest

from strix.agents.factory import build_strix_agent
from strix.tools.meta_tools.tools import search_tools, get_tool_schema, call_tool
from strix.tools.thinking.tool import think
from strix.tools.finish.tool import finish_scan


def test_build_agent_without_meta_tools() -> None:
    agent = build_strix_agent(
        name="root",
        is_root=True,
        use_meta_tools=False
    )
    # The normal agent has ~43 tools.
    assert len(agent.tools) > 10
    
    # finish_scan is present for root.
    names = [t.name for t in agent.tools]
    assert "finish_scan" in names
    assert "search_tools" not in names


def test_build_agent_with_meta_tools() -> None:
    agent = build_strix_agent(
        name="root",
        is_root=True,
        use_meta_tools=True
    )
    # Meta-tools agent has exactly 5 tools: think, search, schema, call, finish
    assert len(agent.tools) == 5
    names = [t.name for t in agent.tools]
    
    assert "think" in names
    assert "search_tools" in names
    assert "get_tool_schema" in names
    assert "call_tool" in names
    assert "finish_scan" in names
    
    # Instruction index should be injected
    assert "Available Tools:" in agent.instructions
    assert "- shell: " in agent.instructions or "- exec_command: " in agent.instructions

def test_build_child_agent_with_meta_tools() -> None:
    agent = build_strix_agent(
        name="child",
        is_root=False,
        use_meta_tools=True
    )
    # Meta-tools agent has exactly 5 tools: think, search, schema, call, agent_finish
    assert len(agent.tools) == 5
    names = [t.name for t in agent.tools]
    
    assert "think" in names
    assert "search_tools" in names
    assert "call_tool" in names
    assert "agent_finish" in names
    assert "finish_scan" not in names
