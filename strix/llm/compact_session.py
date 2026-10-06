"""Minimal pattern for running LLM agents within a 16k token budget.

Uses on-demand tool schemas and state-based session handoffs.
Integrates with the existing ``maybe_compact()`` infrastructure in
:mod:`strix.llm.compaction` and is wired into the execution loop through
:func:`should_compact_session` and :func:`perform_window_reset`.

Gated behind ``STRIX_META_TOOLS=true`` — when disabled the existing
``maybe_compact()`` path handles compaction alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strix.llm.context_budget import count_tokens


logger = logging.getLogger(__name__)

# 1. Compact System Prompt (< 500 tokens)
BASE_SYSTEM_PROMPT = """You are a focused automation assistant.
Operate strictly within the provided task scope.
Use `search_tools` to find relevant capabilities and `get_tool_schema` to inspect arguments before calling `call_tool`.
When nearing task completion or session limits, output a concise checkpoint state using `save_checkpoint`.
"""


class ToolRegistry:
    """Stores full tool schemas outside the LLM context window.

    .. deprecated::
        Prefer :class:`strix.tools.meta_tools.registry.MetaToolRegistry` which
        integrates with the SDK ``Tool`` type. This class is kept for backward
        compatibility with any external code that instantiates it directly.
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._executors: dict[str, Any] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Any,
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._executors[name] = func

    def search(self, query: str) -> list[dict[str, str]]:
        """Returns brief metadata (name + description) instead of full schemas."""
        query = query.lower()
        return [
            {"name": name, "description": data["description"]}
            for name, data in self._tools.items()
            if query in name.lower() or query in data["description"].lower()
        ]

    def get_schema(self, name: str) -> dict[str, Any] | None:
        return self._tools.get(name)

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        func = self._executors.get(name)
        if not func:
            return f"Error: Tool '{name}' not found."
        return func(**args)


class CompactSessionManager:
    """Manages context budget and performs state handoffs across windows.

    This integrates with Strix's existing compaction system:
    - ``should_reset_window()`` is checked before each LLM turn
    - ``create_new_window()`` performs the window handoff with state preservation
    - ``create_checkpoint()`` serialises progress for disk persistence
    """

    def __init__(
        self,
        *,
        max_context_tokens: int = 16_384,
        reserve_tokens: int = 4_000,
        model: str = "",
    ) -> None:
        self.max_tokens = max_context_tokens
        self.threshold = max_context_tokens - reserve_tokens
        self.reserve_tokens = reserve_tokens
        self.model = model
        self.messages: list[dict[str, str]] = []
        self.checkpoint_state: dict[str, Any] = {}
        self._window_resets = 0

    @property
    def window_resets(self) -> int:
        """Number of window resets performed so far."""
        return self._window_resets

    def estimate_tokens(self, text: str) -> int:
        """Token count for text, using the model tokenizer when available."""
        if self.model:
            return count_tokens(self.model, text)
        # Fallback: ~4 chars per token is a conservative upper bound.
        return len(text) // 4

    def current_usage(self) -> int:
        """Estimate the current token usage across all messages."""
        total = 0
        for msg in self.messages:
            total += self.estimate_tokens(msg.get("content", ""))
        return total

    def should_reset_window(self, estimated_tokens: int | None = None) -> bool:
        """Whether the context window should be reset.

        Args:
            estimated_tokens: Override for the current token count.
                If ``None``, computed from ``self.messages``.
        """
        usage = estimated_tokens if estimated_tokens is not None else self.current_usage()
        return usage >= self.threshold

    def create_new_window(self, summary: str, state_data: dict[str, Any]) -> None:
        """Reset message history while preserving critical progress state.

        The fresh window starts with minimal instructions + a state checkpoint,
        keeping the per-window overhead at ~2,000 tokens.
        """
        self.checkpoint_state = state_data
        self._window_resets += 1

        # Fresh window starts with minimal instructions + state checkpoint
        self.messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PREVIOUS SESSION CHECKPOINT:\n"
                    f"Summary: {summary}\n"
                    f"State: {json.dumps(state_data)}\n"
                    f"Continue execution from this state."
                ),
            },
        ]
        logger.info(
            "Window reset #%d: preserved %d state keys, new history has %d messages",
            self._window_resets,
            len(state_data),
            len(self.messages),
        )

    def create_checkpoint(self) -> dict[str, Any]:
        """Serialise the session manager state for disk persistence."""
        return {
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "model": self.model,
            "window_resets": self._window_resets,
            "checkpoint_state": dict(self.checkpoint_state),
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> CompactSessionManager:
        """Restore from a persisted checkpoint."""
        mgr = cls(
            max_context_tokens=int(data.get("max_tokens", 16_384)),
            reserve_tokens=int(data.get("reserve_tokens", 4_000)),
            model=str(data.get("model", "")),
        )
        mgr._window_resets = int(data.get("window_resets", 0))
        mgr.checkpoint_state = dict(data.get("checkpoint_state", {}))
        return mgr


def should_compact_session(
    manager: CompactSessionManager,
    *,
    instructions: str = "",
    tools_text: str = "",
    history_text: str = "",
) -> bool:
    """Check whether the session needs a window reset.

    This is the integration point called by the execution loop before each
    LLM turn, when meta-tools mode is active.
    """
    total = (
        manager.estimate_tokens(instructions)
        + manager.estimate_tokens(tools_text)
        + manager.estimate_tokens(history_text)
    )
    return manager.should_reset_window(total)
