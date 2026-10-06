"""Tests for CompactSessionManager integration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from strix.llm.compact_session import CompactSessionManager, should_compact_session


class TestCompactSessionManager:
    def test_initialization(self) -> None:
        mgr = CompactSessionManager(max_context_tokens=16000, reserve_tokens=4000)
        assert mgr.max_tokens == 16000
        assert mgr.threshold == 12000
        assert mgr.window_resets == 0
        assert len(mgr.messages) == 0

    def test_estimate_tokens_fallback(self) -> None:
        mgr = CompactSessionManager()
        # Fallback uses char length // 4
        text = "a" * 100
        assert mgr.estimate_tokens(text) == 25

    def test_should_reset_window_with_messages(self) -> None:
        mgr = CompactSessionManager(max_context_tokens=1000, reserve_tokens=200)
        mgr.messages.append({"role": "user", "content": "a" * 3200}) # ~800 tokens
        assert mgr.should_reset_window() is True
        
        mgr.messages = [{"role": "user", "content": "a" * 400}] # ~100 tokens
        assert mgr.should_reset_window() is False

    def test_should_reset_window_with_override(self) -> None:
        mgr = CompactSessionManager(max_context_tokens=1000, reserve_tokens=200)
        assert mgr.should_reset_window(800) is True
        assert mgr.should_reset_window(799) is False

    def test_create_new_window(self) -> None:
        mgr = CompactSessionManager()
        mgr.messages.append({"role": "user", "content": "old stuff"})
        
        state = {"foo": "bar"}
        mgr.create_new_window("did some things", state)
        
        assert mgr.window_resets == 1
        assert mgr.checkpoint_state == state
        assert len(mgr.messages) == 2
        assert mgr.messages[0]["role"] == "system"
        assert "PREVIOUS SESSION CHECKPOINT" in mgr.messages[1]["content"]
        assert "did some things" in mgr.messages[1]["content"]
        assert json.dumps(state) in mgr.messages[1]["content"]

    def test_checkpointing(self) -> None:
        mgr = CompactSessionManager(max_context_tokens=16000, model="test-model")
        mgr.create_new_window("summary", {"key": "value"})
        
        ckpt = mgr.create_checkpoint()
        assert ckpt["max_tokens"] == 16000
        assert ckpt["model"] == "test-model"
        assert ckpt["window_resets"] == 1
        assert ckpt["checkpoint_state"] == {"key": "value"}
        
        restored = CompactSessionManager.from_checkpoint(ckpt)
        assert restored.max_tokens == 16000
        assert restored.model == "test-model"
        assert restored.window_resets == 1
        assert restored.checkpoint_state == {"key": "value"}


def test_should_compact_session_integration() -> None:
    mgr = CompactSessionManager(max_context_tokens=1000, reserve_tokens=200)
    
    # 3200 chars total = ~800 tokens, which hits threshold 800
    assert should_compact_session(
        mgr,
        instructions="a" * 1600,
        tools_text="a" * 800,
        history_text="a" * 800
    ) is True
    
    assert should_compact_session(
        mgr,
        instructions="a" * 100,
        tools_text="a" * 100,
        history_text="a" * 100
    ) is False
