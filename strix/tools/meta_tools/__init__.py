"""Meta-tool subsystem for JIT tool schema loading.

Instead of attaching all tool definitions to the LLM context, the meta-tools
expose a search-then-call interface:

    search_tools(query)  →  lightweight name+description matches
    get_tool_schema(name) →  full JSON schema for one tool
    call_tool(name, args)  →  execute a registered tool

This keeps the per-turn tool overhead at ~300 tokens (3 small schemas) instead
of ~21,000 tokens (43 full schemas).

Gated behind ``STRIX_META_TOOLS=true``.
"""
