# Code Style Rules

## Python
- Python 3.11+ syntax only. Always use `from __future__ import annotations`.
- Type hints on all function signatures. Use `X | Y` union syntax, not `Optional[X]`.
- `f-string` for string formatting, never `.format()` or `%`.
- No comments unless the WHY is non-obvious (workaround, hidden constraint, subtle invariant).
- No docstrings on methods — well-named identifiers explain themselves.

## Agents
- Every agent inherits `BaseAgent` and implements `analyze(ticker, state) -> dict`.
- Agents must never raise — always `return {"error": str(e), "score": 0.5}` on failure.
- Agents write only to their own state key. Never read or overwrite another agent's key.
- Score range is always 0.0–1.0. Higher = more attractive / less risky.

## Data Layer
- All external API calls go through `src/warren_brain/data/`. Never call APIs directly in agents.
- All API responses must be cached via `get_cache()` with appropriate TTL.
- Use `tenacity` retry decorator on any external HTTP call.
- Never hardcode API keys — always read from `settings`.

## LLM Calls
- Always use `complete()` from `src/warren_brain/llm.py`. Never instantiate OpenAI/Anthropic clients directly in agents.
- Always use `json_mode=True` when expecting structured JSON back.
- Keep prompts focused — don't ask the LLM to both score and summarize raw data if scoring can be done deterministically.

## MCP Tools
- Tool definitions live only in `src/warren_brain/mcp/tools.py`.
- Each tool description must clearly state what it returns and when to use it — Claude reads this to decide whether to call it.
- `execute_tool()` must handle unknown tool names gracefully: `return {"error": f"Unknown tool: {name}"}`.
