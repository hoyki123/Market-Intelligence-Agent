# Guardrails — Warren Brain

All validation, safety checks, and sanity rules live in `src/warren_brain/guardrails.py`.
**Never duplicate these checks in agents, tools, or UI.** Always import from that module.

## Enforcement Points

| Layer | What is checked | Where enforced |
|-------|----------------|----------------|
| UI — before analysis | Batch cap (≤5 tickers), ticker format, market data existence, leveraged ETF warning | `dashboard.py → _run_analysis_ui()` |
| UI — result display | Stale data banner (>2h old), data quality (DEGRADED/INSUFFICIENT), action vs composite mismatch, high-risk flags, "not financial advice" caption | `dashboard.py → _render_ticker()` |
| UI — chat | Input length cap (500 chars), off-topic detection, certainty/allocation language guardrails appended to every system prompt | `dashboard.py → _render_chat()`, `_answer_question()` |
| MCP agent — tool calls | Ticker mismatch detection + auto-correction (Claude calling tools for wrong ticker) | `mcp/agent.py → _run()` |
| Static pipeline — output | LLM output validation (action field, price sanity, buy < sell), data quality logging | `graph/workflow.py → run_analysis()` |

## Key Constants (in `guardrails.py`)

```python
MAX_TICKERS_PER_RUN = 5       # batch cap
MAX_CHAT_LENGTH = 500         # chat input cap
HIGH_BETA_THRESHOLD = 2.0     # risk warning
HIGH_VOL_THRESHOLD = 0.60     # risk warning (60% annualised vol)
HIGH_DRAWDOWN_THRESHOLD = 0.50 # risk warning (50% max drawdown)
STALE_ANALYSIS_HOURS = 2      # age threshold for stale-data banner
```

## Rules for Contributors

1. **Add new thresholds to `guardrails.py` constants** — not hardcoded in calling files.
2. **Leveraged ETF list** (`LEVERAGED_TICKERS` set) must be updated when new products launch.
3. **`CHAT_GUARDRAIL_SUFFIX`** is appended to every chat system prompt via `apply_chat_guardrails()`. If language rules need updating (e.g. new prohibited phrases), edit that string — not the individual prompt definitions.
4. **`validate_llm_output()`** auto-fixes bad LLM recommendations (swaps inverted prices, defaults missing action to HOLD). It returns warnings that should be logged — never silently ignored.
5. **`check_data_quality()`** returns `"OK" | "DEGRADED" | "INSUFFICIENT"`. INSUFFICIENT means ≥3 agent errors — treat the recommendation as unreliable and surface this prominently in the UI.
6. **Off-topic chat detection** uses substring matching on `sanitize_chat_input()`. Add new off-topic triggers to the `off_topic` list in that function.

## Chat Guardrail Rules (appended to every LLM system prompt)

- Never say a stock "will" reach a price — use "could", "historically", "signals suggest"
- Never recommend a specific dollar amount or portfolio percentage
- Always note "this is not financial advice" when giving specific entry/exit guidance
- If data is insufficient, say so rather than speculating
