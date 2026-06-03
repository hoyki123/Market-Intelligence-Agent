# Warren Brain 🧠

**Multi-agent AI investment intelligence system.** Warren Brain runs a LangGraph pipeline of 8 specialized agents — fundamentals, technicals, sentiment, institutional holdings, ontology, and risk — then synthesizes their signals through a Claude-powered reasoning agent that speaks in Warren Buffett's voice.

Two execution modes: a **static LangGraph graph** (all agents run in parallel, deterministic) and a **dynamic MCP agent** (Claude iteratively decides which tools to call and in what order, adaptive per ticker).

> **[Live Demo →](https://market-intelligence-agent-7phchpsgycqdo4thnastb7.streamlit.app)**
> *(Demo requires a live Anthropic API key — if the app shows an error, run it locally following the instructions below.)*

---

## Architecture

```
                        Static LangGraph Mode
                        ─────────────────────
START ──┬─→ FundamentalsAgent  ──┐
        ├─→ TechnicalsAgent      │
        ├─→ SentimentAgent       ├─→ BuffettBrainAgent ──→ PortfolioManagerAgent ──→ END
        ├─→ ThirteenFAgent       │   (Claude reasoning)     (weighted signal agg)
        ├─→ OntologyAgent        │
        └─→ RiskAgent ───────────┘
             (all parallel via ThreadPoolExecutor)


                        Dynamic MCP Agent Mode
                        ──────────────────────
User request ──→ Claude (claude-sonnet-4-6)
                    │
                    ├─→ decides: get_fundamentals?
                    ├─→ decides: get_risk_metrics?     ← parallel tool execution
                    ├─→ decides: get_sentiment?
                    │         (iterates until confident)
                    └─→ outputs structured recommendation JSON
```

In **static mode**, all 6 data agents always run in parallel and their scores are aggregated by fixed weights. In **dynamic MCP mode**, Claude reads each tool result and decides what to fetch next — a semiconductor company triggers different tool calls than a bank. No fixed pipeline.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` |
| LLM | Anthropic Claude `claude-sonnet-4-6` via tool use / MCP pattern |
| Dynamic agent loop | Anthropic Python SDK — `messages.stream()` + parallel `ThreadPoolExecutor` |
| Market data | yfinance, Massive.com, Finnhub |
| Institutional data | SEC EDGAR 13F filings (no API key required) |
| UI | Streamlit + Plotly |
| Cache | SQLite (TTL-based, two-layer: per-API + full-analysis) |
| Database | SQLAlchemy 2.0 ORM — SQLite (dev) / PostgreSQL (prod) |
| Config | Pydantic Settings |

---

## Agents

| Agent | What it does |
|-------|-------------|
| `FundamentalsAgent` | P/E, ROE, FCF yield, intrinsic value estimate via DCF |
| `TechnicalsAgent` | RSI, MACD, SMA50/200 crossovers, Bollinger Bands, ATR |
| `SentimentAgent` | News sentiment scoring from Massive.com + Finnhub |
| `ThirteenFAgent` | Parses SEC EDGAR 13F XML — checks if Berkshire Hathaway owns it |
| `OntologyAgent` | Supply chain mapping, competitor analysis, ETF exposure |
| `RiskAgent` | Beta, VaR(95%), max drawdown, Sharpe ratio, annualised volatility |
| `BuffettBrainAgent` | Claude synthesizes all signals in Buffett's voice — moat, concerns, conviction |
| `PortfolioManagerAgent` | Weighted score aggregation → BUY/SELL/HOLD + buy/sell price targets |

Each agent implements a `BaseAgent` contract: `analyze(ticker, state) -> dict` with a `score: float` in `[0.0, 1.0]` and always returns `{"error": ..., "score": 0.5}` on failure — the pipeline never crashes on a single agent error.

---

## Key Engineering Decisions

**Two-layer caching** — per-API-call cache (5 min for price, 30 min for news, 1 hr for fundamentals) + full analysis-result cache (1 hr). Second run for any ticker is instant regardless of which agents were slow.

**Parallel tool execution in MCP mode** — when Claude requests multiple tools in one turn, they execute concurrently via `ThreadPoolExecutor`, not sequentially. A 3-tool turn that would take 9s serially takes 3s.

**Centralized guardrails** (`src/warren_brain/guardrails.py`) — all validation and safety checks in one module: ticker format validation, batch size cap (5), LLM output sanity (buy < sell, >12× price flagged), data quality levels (OK/DEGRADED/INSUFFICIENT), high-risk flags (beta > 2.0, vol > 60%), stale data detection, chat input sanitization.

**MCP tool input validation** — Claude occasionally hallucinates the wrong ticker in a tool call. Every tool invocation is validated against the expected ticker and auto-corrected before execution.

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/warren-brain.git
cd warren-brain

python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# Fill in ANTHROPIC_API_KEY (required) and optionally FINNHUB_API_KEY, MASSIVE_API_KEY

streamlit run src/warren_brain/ui/dashboard.py
```

Minimum requirement: `ANTHROPIC_API_KEY`. The system degrades gracefully if Finnhub/Massive keys are absent — yfinance covers fundamentals and technicals.

---

## Project Structure

```
src/warren_brain/
├── agents/          # 8 agents, each a BaseAgent subclass
├── graph/           # LangGraph StateGraph + workflow
├── mcp/             # Dynamic Claude agent (tools.py + agent.py)
├── data/            # API clients: yfinance, Massive, EDGAR, Finnhub, cache, DB
├── ui/              # Streamlit dashboard
├── guardrails.py    # All validation and safety checks
├── llm.py           # Unified LLM client (OpenAI + Anthropic routing)
└── config.py        # Pydantic Settings — all env vars
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Minimum to run:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Tests

```bash
pytest tests/ -v
```

Agents are tested with mocked data layers. Live API calls are not unit tested — they're integration concerns.
