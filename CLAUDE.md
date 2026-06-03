# Warren Brain — Claude Code Project Guide

## What This Project Is
Multi-agent AI investment intelligence system. Produces Buffett-style BUY/SELL/HOLD recommendations by running 6 specialized data agents and one GPT-4o/Claude reasoning agent. Two analysis modes: static LangGraph pipeline and dynamic MCP agent.

## Tech Stack
- **Python 3.11** — use `.venv/bin/python`, never system python
- **LangGraph** — agent orchestration (static graph mode)
- **Anthropic SDK** — Claude claude-sonnet-4-6 for MCP dynamic agent and BuffettBrainAgent
- **OpenAI SDK** — GPT-4o fallback via dataexpert.io proxy
- **Streamlit** — dashboard UI at localhost:8501
- **SQLAlchemy 2.0** — ORM, supports SQLite (dev) and PostgreSQL/Supabase (prod)
- **yfinance** — financial ratios, price history
- **Massive.com API** — real-time price, news sentiment, company overview, related tickers
- **SEC EDGAR** — 13F institutional holdings (no API key required)
- **Finnhub** — news fallback, insider sentiment

## Key Commands
```bash
# Run UI
.venv/bin/streamlit run src/warren_brain/ui/dashboard.py --server.port 8501

# Run analysis from CLI
.venv/bin/python -c "from warren_brain.graph.workflow import run_analysis; import json; print(json.dumps(run_analysis('AAPL')['recommendation'], indent=2))"

# Run MCP dynamic agent
.venv/bin/python -c "from warren_brain.mcp.agent import run_mcp_analysis; import json; print(json.dumps(run_mcp_analysis('AAPL'), indent=2))"

# Test LLM connection
.venv/bin/python -c "from warren_brain.llm import complete; print(complete([{'role':'user','content':'Reply OK'}]))"

# Clear analysis cache (force fresh run)
.venv/bin/python -c "
import sqlite3
with sqlite3.connect('warren_brain.db') as c:
    print('Cleared:', c.execute(\"DELETE FROM cache WHERE key LIKE 'analysis:%' OR key LIKE 'mcp:%'\").rowcount, 'entries')
"

# Install dependencies
.venv/bin/pip install -e ".[dev]"
```

## Architecture
```
START
  ├─→ FundamentalsAgent ──┐
  ├─→ TechnicalsAgent     ├─→ BuffettBrainAgent → PortfolioManagerAgent → END
  ├─→ SentimentAgent      │   (reads all state)    (final recommendation)
  ├─→ ThirteenFAgent      │
  ├─→ OntologyAgent       │
  └─→ RiskAgent ──────────┘
       (all parallel)
```

**MCP Dynamic Agent** (alternative mode): Claude calls tools from `src/warren_brain/mcp/tools.py` in whatever order it decides — not a fixed graph.

## Key File Locations
| What | Where |
|------|-------|
| All agents | `src/warren_brain/agents/` |
| LangGraph workflow | `src/warren_brain/graph/workflow.py` |
| Shared state schema | `src/warren_brain/graph/state.py` |
| MCP tools + dispatcher | `src/warren_brain/mcp/tools.py` |
| MCP dynamic agent loop | `src/warren_brain/mcp/agent.py` |
| LLM client (OpenAI + Anthropic) | `src/warren_brain/llm.py` |
| Massive.com API client | `src/warren_brain/data/massive.py` |
| SQLite cache | `src/warren_brain/data/cache.py` |
| Database engine + ORM | `src/warren_brain/data/database.py` |
| Streamlit dashboard | `src/warren_brain/ui/dashboard.py` |
| Settings (all env vars) | `src/warren_brain/config.py` |
| Environment variables | `.env` |

## Environment Variables (.env)
```
LLM_PROVIDER=anthropic          # or "openai"
ANTHROPIC_API_KEY=...
ANTHROPIC_BASE_URL=https://www.dataexpert.io/api/v1/anthropic
ANTHROPIC_MODEL=claude-sonnet-4-6
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://www.dataexpert.io/api/v1/openai
MASSIVE_API_KEY=...             # enables real-time price, sentiment, company data
FINNHUB_API_KEY=...             # news fallback
DB_HOST=...                     # set for Supabase; leave blank for SQLite
```

## Cache TTLs
- Massive snapshot (price): 5 min
- Massive news: 30 min
- Massive overview/related: 24 hrs
- yfinance data: 1 hr
- Full analysis result: 1 hr (MCP and static)

## Proxy Note
Both OpenAI and Anthropic use the dataexpert.io proxy which returns SSE streaming format. Both SDK clients must use `stream=True` / `.messages.stream()`. See `src/warren_brain/llm.py`.

## Database
- SQLite (default): zero config, file `warren_brain.db`
- PostgreSQL/Supabase: set `DB_HOST`, `DB_USER`, `DB_PASSWORD` (raw — quote_plus applied in code), `DB_NAME`, `DB_PORT`
- Supabase free tier pauses after inactivity — DNS stops resolving when paused

## Agent Weights (configurable via .env)
```
WEIGHT_FUNDAMENTALS=0.30
WEIGHT_TECHNICALS=0.20
WEIGHT_SENTIMENT=0.15
WEIGHT_THIRTEEN_F=0.15
WEIGHT_ONTOLOGY=0.10
WEIGHT_RISK=0.10
```

## Guardrails
All validation and safety rules are centralised in `src/warren_brain/guardrails.py`. **Never duplicate checks elsewhere.** See `.claude/rules/guardrails.md` for the full enforcement map.

Key guardrails active in the system:
- **Input**: batch capped at 5 tickers, ticker format + market-data validated before analysis runs
- **Leveraged ETFs**: warned prominently (TQQQ, SQQQ, SOXL, etc. — 40+ tickers in `LEVERAGED_TICKERS`)
- **LLM output**: action validated, buy < sell enforced, prices that are >12× current price flagged
- **Data quality**: DEGRADED (some agent errors) or INSUFFICIENT (≥3 errors) surfaces warnings/errors in UI
- **Risk**: beta > 2.0, annualised vol > 60%, max drawdown > 50% each generate explicit warnings
- **Stale data**: banner shown if cached analysis is > 2 hours old
- **Chat**: 500-char input cap, off-topic detection, certainty/allocation language rules appended to every system prompt
