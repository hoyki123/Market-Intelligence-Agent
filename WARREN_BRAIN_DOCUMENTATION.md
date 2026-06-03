# Warren Brain — Complete System Documentation

## Table of Contents
1. [What is Warren Brain](#what-is-warren-brain)
2. [What Makes It Agentic AI](#what-makes-it-agentic-ai)
3. [System Architecture](#system-architecture)
4. [Shared State — The Coordination Backbone](#shared-state--the-coordination-backbone)
5. [Agent 1 — FundamentalsAgent](#agent-1--fundamentalsagent)
6. [Agent 2 — TechnicalsAgent](#agent-2--technicalsagent)
7. [Agent 3 — SentimentAgent](#agent-3--sentimentagent)
8. [Agent 4 — ThirteenFAgent](#agent-4--thirteenfagent)
9. [Agent 5 — OntologyAgent](#agent-5--ontologyagent)
10. [Agent 6 — RiskAgent](#agent-6--riskagent)
11. [Agent 7 — BuffettBrainAgent](#agent-7--buffettbrainagent)
12. [Agent 8 — PortfolioManagerAgent](#agent-8--portfoliomanageragent)
13. [LLM Layer](#llm-layer)
14. [Data Infrastructure](#data-infrastructure)
15. [Cache System](#cache-system)
16. [Database / Persistence](#database--persistence)
17. [Backtest Engine](#backtest-engine)
18. [Streamlit UI](#streamlit-ui)
19. [Configuration & Agent Weights](#configuration--agent-weights)
20. [External APIs & Keys](#external-apis--keys)
21. [Agent Score Weights (Default)](#agent-score-weights-default)
22. [Live Pipeline Example — AAPL](#live-pipeline-example--aapl)

---

## What is Warren Brain

Warren Brain is a multi-agent AI investment analysis system that produces Buffett-style buy/sell/hold recommendations for any publicly traded US stock ticker. It combines:

- **Real-time market data** from multiple APIs (yfinance, Massive.com, Finnhub, SEC EDGAR)
- **Quantitative analysis** (technical indicators, risk metrics, fundamental ratios, intrinsic value estimation)
- **GPT-4o reasoning** (sentiment synthesis, Buffett-style investment thesis)
- **LangGraph orchestration** (parallel fan-out, fan-in barrier, ordered pipeline)
- **SQLite/PostgreSQL persistence** (every analysis result saved with full agent output)

It does not trade. It produces a structured recommendation: action (BUY/SELL/HOLD), buy price, sell price, composite score, and a full breakdown of what every agent found.

---

## What Makes It Agentic AI

Warren Brain satisfies all four properties that define agentic AI systems:

### 1. Autonomous Decision-Making at Each Node
Each agent independently decides how to produce its output without being told step-by-step what to do. For example:
- FundamentalsAgent decides internally whether to enrich with Massive or stay with yfinance data
- SentimentAgent checks if Massive API is available and switches data strategies accordingly
- OntologyAgent decides between a live Massive-powered analysis vs. a hardcoded fallback dict
- RiskAgent independently fetches SPY data to compute beta, without being prompted to do so

No central controller tells agents what to call. Each agent has its own `analyze()` method that owns its entire reasoning process.

### 2. Tool Use (Not Just LLM Calls)
Agents actively call external tools and APIs as instruments:
- yfinance → price history, financial ratios, stock info
- Massive.com → news with pre-scored sentiment, real-time snapshots, related companies, ticker overview
- SEC EDGAR → 13F filings, CIK resolution, XML infotable parsing
- Finnhub → news articles, insider sentiment
- GPT-4o via proxy → only used for synthesis (summarizing scored data, Buffett-style reasoning)

The LLM is one tool among many, not the core engine. Most of the work is done by APIs and deterministic code.

### 3. Shared State as the Coordination Mechanism
No agent calls another agent directly. All communication goes through `WarrenBrainState` — a TypedDict managed by LangGraph. Agents write to their own key, read from others' keys. This decoupling is what allows parallel execution and loose coupling.

BuffettBrainAgent reads `state["fundamentals"]`, `state["technicals"]`, `state["sentiment"]`, etc. — outputs written by agents that ran before it, without knowing how they ran.

### 4. Structured Orchestration via LangGraph StateGraph
LangGraph enforces the execution order via directed edges. The graph definition:
- 6 data agents fan out from START simultaneously (true parallel execution)
- BuffettBrainAgent has edges from all 6 — LangGraph treats this as a fan-in barrier and waits for all 6 to complete before running it
- PortfolioManagerAgent runs only after BuffettBrainAgent completes
- All errors are non-fatal — agents return `{"error": "...", "score": 0.5}` and the pipeline continues

---

## System Architecture

```
START
  ├──────────────────────────────────────────────────────────────────────────┐
  ├─→ FundamentalsAgent ──────────────────────────────────────────────────── ┤
  ├─→ TechnicalsAgent ────────────────────────────────────────────────────── ┤ (parallel)
  ├─→ SentimentAgent ─────────────────────────────────────────────────────── ├──→ BuffettBrainAgent → PortfolioManagerAgent → END
  ├─→ ThirteenFAgent ─────────────────────────────────────────────────────── ┤ (fan-in wait)
  ├─→ OntologyAgent ──────────────────────────────────────────────────────── ┤
  └─→ RiskAgent ──────────────────────────────────────────────────────────── ┘
```

**File layout:**
```
src/warren_brain/
├── agents/
│   ├── base.py              # Abstract base class all agents inherit from
│   ├── fundamentals.py      # FundamentalsAgent
│   ├── technicals.py        # TechnicalsAgent
│   ├── sentiment.py         # SentimentAgent
│   ├── thirteen_f.py        # ThirteenFAgent
│   ├── ontology.py          # OntologyAgent
│   ├── risk.py              # RiskAgent
│   ├── buffett_brain.py     # BuffettBrainAgent
│   └── portfolio_manager.py # PortfolioManagerAgent
├── graph/
│   ├── state.py             # WarrenBrainState TypedDict
│   └── workflow.py          # LangGraph StateGraph builder + run_analysis()
├── data/
│   ├── market_data.py       # yfinance wrappers (price history, info, financials)
│   ├── massive.py           # Massive.com API client (news, snapshot, overview, related)
│   ├── news.py              # Finnhub + yfinance news fallback
│   ├── edgar.py             # SEC EDGAR 13F filing parser
│   ├── cache.py             # SQLite-backed TTL cache
│   ├── database.py          # SQLAlchemy engine (SQLite + PostgreSQL/Supabase)
│   ├── models.py            # ORM models (AnalysisResult, AgentSignal, BacktestResult)
│   └── repository.py        # DB write helpers (save_analysis)
├── backtest/
│   └── engine.py            # Buy-hold-sell simulator vs SPY benchmark
├── ui/
│   └── dashboard.py         # Streamlit dashboard
├── llm.py                   # Shared GPT-4o client + complete() helper
├── config.py                # Pydantic settings (reads .env)
└── main.py                  # CLI entry point
```

---

## Shared State — The Coordination Backbone

`WarrenBrainState` is a `TypedDict` defined in `src/warren_brain/graph/state.py`. It flows through every node in the graph. Each agent reads from it and writes its output back to it via LangGraph's merge mechanism.

```python
class WarrenBrainState(TypedDict, total=False):
    ticker: str                     # Input — e.g. "AAPL"

    # Agent outputs (each agent writes to its own key)
    fundamentals: dict              # Written by FundamentalsAgent
    technicals: dict                # Written by TechnicalsAgent
    sentiment: dict                 # Written by SentimentAgent
    thirteen_f: dict                # Written by ThirteenFAgent
    ontology: dict                  # Written by OntologyAgent
    risk: dict                      # Written by RiskAgent
    buffett_brain: dict             # Written by BuffettBrainAgent

    # Final output
    recommendation: dict            # Written by PortfolioManagerAgent

    # LangGraph message accumulator (for future conversational interface)
    messages: Annotated[list[BaseMessage], add_messages]

    # Non-fatal errors from any agent
    errors: list[str]
```

**Key design principle:** No agent overwrites another agent's key. When the node function returns `{"fundamentals": {...}}`, LangGraph merges only that key back into the state. All other keys are untouched.

**Error handling in the graph:** The `_node()` wrapper in `workflow.py` checks if the agent returned an `"error"` key. If it did, the error is appended to `state["errors"]` and the pipeline continues. Nothing crashes.

```python
def _node(agent, key: str):
    def node_fn(state: WarrenBrainState) -> dict:
        result = agent.analyze(state["ticker"], state)
        if "error" in result:
            errors = list(state.get("errors") or [])
            errors.append(f"{agent.name}: {result['error']}")
            return {key: result, "errors": errors}
        return {key: result}
    return node_fn
```

---

## Agent 1 — FundamentalsAgent

**File:** `src/warren_brain/agents/fundamentals.py`

**Purpose:** Evaluate intrinsic business quality and valuation from financial ratios, then estimate what the business is actually worth (intrinsic value).

### Data Sources
| Source | What it provides | When used |
|--------|-----------------|-----------|
| yfinance | P/E, forward P/E, ROE, profit margin, revenue growth, D/E ratio, free cash flow, earnings growth, beta, analyst target, 52w high/low | Always |
| Massive `/v3/reference/tickers/{ticker}` | Company description (up to 400 chars), employee count, real market cap, homepage URL | When MASSIVE_API_KEY is set |
| Massive `/v2/snapshot/.../tickers/{ticker}` | Real-time closing price (OHLC), VWAP (intraday) | When MASSIVE_API_KEY is set |

yfinance financial ratios are always used — Massive paid plan required for financial statement endpoints, which are not available on the free tier.

Massive enrichment is best-effort: wrapped in a try/except so yfinance data is used as fallback if Massive fails.

### Composite Score Computation
Score range: 0.0 (worst) to 1.0 (best). Higher = more attractive to a value investor.

| Signal | Logic |
|--------|-------|
| ROE | Normalized on [0%, 40%]. Score 1.0 at ROE=40%, 0.0 at ROE=0% |
| Profit Margin | Normalized on [0%, 30%]. 20%+ is excellent |
| P/E Ratio | Normalized on [5x, 60x] then **inverted** — lower P/E = higher score |
| Revenue Growth | Normalized on [-10%, 40%] |
| Debt/Equity | Normalized on [0, 200] then **inverted** — lower debt = higher score |
| Free Cash Flow | Binary: 1.0 if positive, 0.2 if negative |

Final score = simple average of all available signals. If a metric is missing (company doesn't report it), it's excluded from the average rather than penalizing.

### Intrinsic Value Estimate (Benjamin Graham Formula)
```
IV = EPS × (8.5 + 2 × g) × 4.4 / Y
```
- `EPS` = current price / forward P/E (proxy for next year's earnings per share)
- `g` = expected earnings growth in % (defaults to 7% if not reported)
- `Y` = 4.5% (approximate AAA bond yield)
- `8.5` = P/E for a zero-growth company (Graham's constant)

This produces a dollar estimate of intrinsic value. If it's above current price, there's upside; if below, the stock is overpriced.

### Output Written to State
```json
{
  "metrics": {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "pe_ratio": 35.5,
    "forward_pe": 29.2,
    "roe": 0.162,
    "profit_margin": 0.264,
    "revenue_growth": 0.041,
    "debt_to_equity": 145.0,
    "free_cashflow": 108000000000,
    "current_price": 213.49,
    "total_employees": 164000,
    "market_cap": 3200000000000,
    "vwap": 213.12,
    "description": "Apple Inc. designs, manufactures, and markets smartphones..."
  },
  "intrinsic_value_estimate": 487.32,
  "score": 0.748,
  "summary": "Apple Inc (AAPL) — sector: Technology. ROE: 16.2%. Net margin: 26.4%. P/E: 35.5x. Estimated intrinsic value: $487.32 (↑128.2% vs current $213.49). Fundamental score: 0.748/1.00."
}
```

---

## Agent 2 — TechnicalsAgent

**File:** `src/warren_brain/agents/technicals.py`

**Purpose:** Identify momentum, trend direction, and entry/exit timing signals from 2 years of daily price history.

### Data Sources
- yfinance `stock.history(period="2y")` — daily OHLCV (open, high, low, close, volume)
- No external APIs — all computation is local on the price series

### Indicators Computed
All calculated manually in pandas (no external TA library dependency):

| Indicator | Calculation | What it tells you |
|-----------|-------------|-------------------|
| SMA 20/50/200 | Simple rolling mean | Trend direction (short/medium/long-term) |
| RSI (14) | Wilder's RS formula using 14-day average gain/loss | Overbought (>70) or oversold (<30) |
| MACD | EMA(12) − EMA(26); signal = EMA(9) of MACD | Momentum crossover |
| MACD Histogram | MACD − signal line | Direction of momentum change |
| VWAP (20-day rolling) | (Typical price × Volume).sum / Volume.sum | Fair value benchmark vs. intraday price |
| ATR (14) | True range rolling mean | Volatility context for position sizing |
| 52-week high/low | Max/min of last 252 trading days | Distance from extremes |
| % from 52w high | (current/52w_high − 1) × 100 | How far from peak (negative = cheaper) |
| Price above SMA flags | Boolean: current > SMA20/50/200 | Trend alignment |

### Score Computation
| Signal | Logic |
|--------|-------|
| RSI | <30 → 0.85, <40 → 0.70, <55 → 0.55, <65 → 0.45, ≥65 → 0.25 |
| Above SMA20 | True → 0.65, False → 0.35 |
| Above SMA50 | True → 0.65, False → 0.35 |
| Above SMA200 | True → 0.65, False → 0.35 |
| MACD Histogram | Positive → 0.65, Negative → 0.35 |
| % from 52w High | Normalized on [0, 40%] — further from high = better entry value |

Score = average of all signals.

### Signal Label Logic
Counts how many of 4 bullish conditions are true (RSI < 50, MACD histogram positive, above SMA50, above SMA200):
- 3+ bullish AND RSI < 45 → `STRONG_BUY`
- 3+ bullish → `BUY`
- 1 or fewer bullish AND RSI > 65 → `SELL`
- 1 or fewer bullish → `WEAK`
- Otherwise → `NEUTRAL`

### Output Written to State
```json
{
  "indicators": {
    "current_price": 213.49,
    "sma20": 208.3,
    "sma50": 205.1,
    "sma200": 195.8,
    "rsi14": 58.2,
    "macd": 2.14,
    "macd_signal": 1.87,
    "macd_histogram": 0.27,
    "vwap_20d": 209.5,
    "atr14": 3.82,
    "above_sma20": true,
    "above_sma50": true,
    "above_sma200": true,
    "52w_high": 237.23,
    "52w_low": 164.08,
    "pct_from_52w_high": -10.0
  },
  "signal": "BUY",
  "score": 0.475,
  "summary": "AAPL technical signal: BUY. RSI(14): 58.2. MACD histogram: 0.270 (bullish). Trend: above SMA50, above SMA200. -10.0% from 52w high. Technical score: 0.475/1.00."
}
```

---

## Agent 3 — SentimentAgent

**File:** `src/warren_brain/agents/sentiment.py`

**Purpose:** Determine market sentiment around the ticker from recent news. Scores news as VERY_BULLISH / BULLISH / NEUTRAL / BEARISH / VERY_BEARISH, identifies key themes, produces a 1-2 sentence synthesis.

### Two Operating Modes

**Mode 1: Massive API (primary, requires MASSIVE_API_KEY)**

Massive's news endpoint (`/v2/reference/news`) returns articles with pre-scored per-ticker sentiment — each article already has `sentiment: positive/neutral/negative` and a `sentiment_reasoning` string explaining why.

This means **the LLM is NOT used for scoring**. It's only used to synthesize themes and write a summary from the already-scored data. This is much cheaper (1 LLM call vs. scoring 20+ raw headlines).

Data flow:
1. `fetch_news_with_sentiment(ticker)` → list of articles with `.sentiment` and `.sentiment_reasoning` per ticker
2. `aggregate_sentiment(articles)` → counts positive/neutral/negative, computes weighted score, assigns label
3. `_llm_summary_from_scored()` → GPT-4o given counts + top 10 headlines + top 5 reasoning snippets → returns `{"themes": [...], "summary": "..."}`
4. `fetch_insider_sentiment(ticker)` via Finnhub → monthly share purchase ratio (MSPR)

**Sentiment Score Weighting:**
```
raw_score = (positive_count × 1.0 + neutral_count × 0.5) / total_articles

raw_score ≥ 0.70 → VERY_BULLISH
raw_score ≥ 0.58 → BULLISH
raw_score ≥ 0.42 → NEUTRAL
raw_score ≥ 0.30 → BEARISH
raw_score < 0.30  → VERY_BEARISH
```

**Label to numeric score mapping:**
| Label | Score |
|-------|-------|
| VERY_BULLISH | 0.85 |
| BULLISH | 0.65 |
| NEUTRAL | 0.50 |
| BEARISH | 0.35 |
| VERY_BEARISH | 0.15 |

---

**Mode 2: Finnhub + yfinance fallback (no Massive key)**

1. `fetch_news(ticker)` → tries Finnhub first, falls back to yfinance
2. Takes up to 20 raw headlines
3. `_llm_score_headlines()` → GPT-4o receives all headlines and returns `{"label": "...", "score": ..., "themes": [...], "summary": "..."}`

In this mode, the LLM does the scoring — more expensive but works without Massive.

### Cache TTL
News cached for 30 minutes (`TTL_NEWS = 1800`). During market hours, new articles appear constantly so short TTL is appropriate.

### Output Written to State
```json
{
  "score": 0.85,
  "sentiment_label": "VERY_BULLISH",
  "sentiment_score_raw": 0.74,
  "counts": {"positive": 18, "neutral": 5, "negative": 2},
  "key_themes": ["AI expansion", "Services revenue growth", "iPhone demand"],
  "article_count": 25,
  "insider_sentiment": {"year": 2025, "month": 3, "mspr": 0.12, "change": 45000},
  "data_source": "massive",
  "summary": "Apple's sentiment is strongly bullish, driven by record Services revenue and growing AI integration. Analysts highlight strong demand for the iPhone 16 lineup despite mixed macro signals."
}
```

---

## Agent 4 — ThirteenFAgent

**File:** `src/warren_brain/agents/thirteen_f.py`

**Purpose:** Identify what institutional investors (especially Berkshire Hathaway) hold in the stock, and whether they're adding or reducing.

### What is a 13F?
SEC Form 13F is a quarterly filing required of any institutional investment manager with over $100M in AUM. It lists every equity holding ≥ $200k. Filed within 45 days of quarter end. Public at `data.sec.gov`.

### Data Sources
- SEC EDGAR `data.sec.gov/submissions/CIK{cik}.json` — filing index for an institution
- SEC EDGAR `Archives/edgar/data/{cik}/{accession}/{file}-infotable.xml` — XML with the actual holdings table
- `company_tickers.json` at `sec.gov` — maps ticker symbols to CIK numbers
- yfinance `stock.info` — `heldPercentInstitutions`, `heldPercentInsiders`, `shortRatio`

### Known Institutions Checked
```python
KNOWN_INSTITUTIONS = {
    "Berkshire Hathaway": "0001067983",
    "BlackRock": "0001364742",
    "Vanguard": "0000102909",
    "State Street": "0000093751",
}
```

### What It Does Step by Step
1. Fetch the last 2 Berkshire 13F filings from EDGAR
2. Parse the XML infotable of each — extract all holdings, sorted by `value_usd` descending
3. Search for the ticker in both filings (name match: `ticker_upper in name.upper()`)
4. If found in both, compute share count change: `(current - prior) / prior × 100`
5. Classify trend: `ADDED` (+2%), `HELD` (±2%), `REDUCED` (-2%)
6. Also pull yfinance institutional/insider/short data as secondary signals

### Score Computation
| Signal | Score |
|--------|-------|
| Berkshire ADDED | 0.80 |
| Berkshire HELD | 0.65 |
| Berkshire REDUCED | 0.40 |
| Not in Berkshire | No contribution |
| Institutional ownership % | Normalized on [30%, 90%]. High institutional ownership signals quality |
| Insider ownership % | `min(0.75, 0.40 + pct × 3)` — skin in the game is positive |
| Short ratio | Normalized on [0, 15 days] then inverted — high short interest is bearish |

### Output Written to State
```json
{
  "berkshire_position": {
    "period": "2024-09-30",
    "value_usd": 84500000000,
    "shares": 400000000,
    "shares_prev": 915000000,
    "pct_change": -56.3,
    "trend": "REDUCED"
  },
  "institutional_signal": {
    "institutional_ownership_pct": 0.592,
    "insider_ownership_pct": 0.000,
    "short_ratio": 1.2
  },
  "score": 0.620,
  "summary": "Berkshire Hathaway REDUCED AAPL — $84.50B position as of 2024-09-30. Shares changed -56.3% vs prior quarter. 13F/Institutional score: 0.620/1.00."
}
```

**Known limitation:** Full cross-institution search (finding every institution that holds a given ticker) requires paid EDGAR search APIs (sec-api.io, Refinitiv). The current implementation uses yfinance aggregate ownership stats as a proxy.

---

## Agent 5 — OntologyAgent

**File:** `src/warren_brain/agents/ontology.py`

**Purpose:** Map the company's position in its ecosystem — peers, supply chain relationships, sector ETFs, company size. Answers: "How connected and established is this business?"

### Two Operating Modes

**Mode 1: Massive API (primary)**

Calls two endpoints:
- `fetch_related_companies(ticker)` → `/v1/related-companies/{ticker}` → list of related ticker symbols (peers, co-movers, supply chain players)
- `fetch_ticker_overview(ticker)` → `/v3/reference/tickers/{ticker}` → `{description, sic_description, total_employees, market_cap, homepage_url}`

Score computation:
| Signal | Logic |
|--------|-------|
| Related company count | ≥8 → 0.70, ≥4 → 0.60, >0 → 0.50, 0 → 0.40 |
| Employee count | ≥50k → 0.72, ≥10k → 0.65, ≥1k → 0.55, <1k → 0.45 |
| Market cap | ≥$200B → 0.75, ≥$10B → 0.65, ≥$1B → 0.50, <$1B → 0.35 |

These proxies reflect Buffett's preference for large, established businesses with wide ecosystems.

**Mode 2: Static fallback (no Massive key)**

Uses a hardcoded dict of major tickers with manually curated peers and suppliers:
```python
_SECTOR_RELATIONSHIPS = {
    "NVDA": {"peers": ["AMD", "INTC", "QCOM"], "suppliers": ["ASML", "AMAT", "KLAC", "LRCX"]},
    "AAPL": {"peers": ["MSFT", "GOOGL"], "suppliers": ["TSM", "QCOM", "AVGO"]},
    "TSM":  {"peers": ["INTC", "SMIC"], "suppliers": ["ASML", "AMAT", "KLAC"]},
    ...
}
```
Score is always 0.5 in fallback mode.

### Sector ETF Mapping
Maps each sector to relevant ETFs:
```python
SECTOR_ETFS = {
    "Technology":    ["QQQ", "XLK", "SMH", "SOXX"],
    "Healthcare":    ["XLV", "IBB"],
    "Financials":    ["XLF", "KRE"],
    "Energy":        ["XLE", "VDE"],
    ...
}
```
Used to show which ETFs would be affected by this stock's performance.

### Output Written to State
```json
{
  "sector": "Electronic Computers",
  "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers...",
  "total_employees": 164000,
  "market_cap": 3200000000000,
  "related_tickers": ["MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSM", "QCOM", "AVGO"],
  "relevant_etfs": ["QQQ", "XLK", "SMH"],
  "score": 0.723,
  "data_source": "massive",
  "summary": "Apple Inc. designs, manufactures... Related companies: MSFT, GOOGL, META, AMZN, NVDA, TSM. Relevant ETFs: QQQ, XLK, SMH. Employees: 164,000. Ontology score: 0.723/1.00."
}
```

---

## Agent 6 — RiskAgent

**File:** `src/warren_brain/agents/risk.py`

**Purpose:** Quantify how risky this stock is from a Buffett perspective — not volatility as "risk" but real metrics that measure how much you could lose.

### Data Sources
- yfinance price history for the ticker (3 years of daily close)
- yfinance price history for SPY (3 years — used to compute beta)

All computation is local numpy/pandas. No external APIs.

### Metrics Computed

| Metric | Formula | What it means |
|--------|---------|---------------|
| Annualized Volatility | `daily_returns.std() × √252` | Standard deviation of returns × trading days. Lower is safer. |
| Beta vs SPY | `Cov(stock_returns, spy_returns) / Var(spy_returns)` | Market sensitivity. Beta 1.0 = moves with market. >1.5 = amplified swings. |
| VaR 95% (1-day) | 5th percentile of daily returns | On a bad day (1-in-20), you could lose this much in one day |
| Max Drawdown | `min((cumulative - cummax) / cummax)` | Worst peak-to-trough loss over 3 years |
| Sharpe Ratio | `(mean_excess_return / std_excess_return) × √252` | Risk-adjusted return above 4.5% risk-free rate. >1.0 is good. |
| Sortino Ratio | `(mean_excess_return / downside_std) × √252` | Like Sharpe but only penalizes downside volatility |

### Score Computation (higher = safer = more Buffett-friendly)
| Signal | Logic |
|--------|-------|
| Annualized volatility | Normalized on [10%, 70%] then **inverted** — lower vol = higher score |
| Beta | Normalized on deviation from 0.8 (ideal beta), range [0, 1.5], inverted |
| Max drawdown | Normalized on [0, 60%] then **inverted** — smaller drawdown = higher score |
| Sharpe ratio | Normalized on [0, 3.0] — higher Sharpe = higher score |

### Output Written to State
```json
{
  "metrics": {
    "annualized_volatility": 0.2814,
    "beta": 1.188,
    "var_95_1d": -0.0183,
    "max_drawdown": -0.3142,
    "sharpe_ratio": 0.841,
    "sortino_ratio": 1.204,
    "trading_days": 753
  },
  "score": 0.545,
  "summary": "AAPL risk profile: Beta: 1.19. Annual volatility: 28.1%. VaR(95%, 1d): -1.8%. Max drawdown: -31.4%. Sharpe: 0.84. Risk score (higher = safer): 0.545/1.00."
}
```

---

## Agent 7 — BuffettBrainAgent

**File:** `src/warren_brain/agents/buffett_brain.py`

**Purpose:** The synthesis agent. Reads all 6 prior agents' outputs from state and produces a Buffett-style investment thesis: whether to buy, at what price, with what conviction, and why.

This is the only agent that calls GPT-4o for its core reasoning (not just summarization). Every other agent does deterministic computation. BuffettBrainAgent delegates the cross-signal synthesis to GPT-4o because that synthesis is inherently qualitative — you need to weigh "cheap valuation" against "declining institutional confidence" against "bearish technicals" in a way that requires judgment.

### System Prompt (Warren Buffett persona)
```
You are Warren Brain — an AI investment analyst trained in the philosophy
of Warren Buffett and Charlie Munger. You reason with:

1. MOAT: Durable competitive advantages (brand, network effects, switching costs)
2. MANAGEMENT: Shareholder-friendly, skin in the game, long-term focus
3. FINANCIALS: High ROE (>15%), consistent FCF, low debt, predictable earnings
4. PRICE: Significant margin of safety below intrinsic value
5. CIRCLE OF COMPETENCE: Only businesses predictable 10 years out
6. PATIENCE: Prefer holding forever. Never buy on short-term noise
7. RISK: Permanent capital loss is the only real risk. Volatility = opportunity
```

### Context Building — What BuffettBrain Sees
`_build_context()` assembles a structured text block from all state fields:
- Fundamentals: P/E, ROE, intrinsic value estimate vs. current price
- Technicals: signal label (BUY/SELL/NEUTRAL), summary
- Sentiment: label, score, key themes
- 13F: Berkshire position and trend
- Ontology: supply chain and sector summary
- Risk: key risk metrics summary

This text block becomes the user message to GPT-4o. The agent doesn't cherry-pick — it feeds everything.

### GPT-4o Output Schema (JSON mode, temperature=0.1)
```json
{
  "action": "BUY | SELL | HOLD",
  "buy_price": 195.00,
  "sell_price": 320.00,
  "conviction": "HIGH | MEDIUM | LOW",
  "moat_assessment": "STRONG — brand loyalty and services ecosystem create significant switching costs",
  "reasoning": "Apple's business is exceptional but the current P/E of 35.5x leaves little margin of safety. The intrinsic value estimate of $487 is compelling but assumes strong growth continuation. At current prices, a patient investor should wait for a correction to below $195.",
  "concerns": [
    "Berkshire has reduced AAPL position by 56% — a meaningful signal from Buffett himself",
    "P/E of 35x assumes significant growth but revenue growth is only 4.1%",
    "High dependence on iPhone (52% of revenue) creates concentration risk"
  ]
}
```

### Why JSON Mode + Streaming
The OpenAI client uses `response_format={"type": "json_object"}` to guarantee parseable JSON. `stream=True` is forced because the dataexpert.io proxy returns SSE format regardless of stream parameter — the `complete()` helper in `llm.py` handles iterating chunks and assembling the full string.

### Output Written to State
The raw JSON dict from GPT-4o, exactly as returned. PortfolioManagerAgent reads `state["buffett_brain"]["action"]`, `["buy_price"]`, `["sell_price"]`, `["conviction"]`, `["reasoning"]`, `["concerns"]`.

---

## Agent 8 — PortfolioManagerAgent

**File:** `src/warren_brain/agents/portfolio_manager.py`

**Purpose:** Final aggregation layer. Combines all 6 quantitative scores into a weighted composite, defers to BuffettBrain's action/prices/conviction, and produces the final recommendation dict.

### Composite Score Computation
Reads the `score` field from each of the 6 data agents and applies configurable weights:

```python
weights = {
    "fundamentals": 0.30,   # Highest weight — most important to Buffett
    "technicals":   0.20,   # Second highest — entry timing matters
    "sentiment":    0.15,   # Market mood
    "thirteen_f":   0.15,   # Institutional validation
    "ontology":     0.10,   # Ecosystem quality
    "risk":         0.10,   # Risk penalty
}
```

Weights are normalized: if an agent failed and returned no score, its weight is redistributed across available agents.

`composite_score = Σ(score × weight) / Σ(weights of available agents)`

### Action Derivation
1. If BuffettBrainAgent returned a valid `action` (BUY/SELL/HOLD) → use it directly
2. Fallback if BuffettBrain failed: `composite ≥ 0.65 → BUY`, `≤ 0.35 → SELL`, else `HOLD`

### Price Derivation
Priority order:
1. BuffettBrain's `buy_price` and `sell_price` (GPT-4o calculated)
2. Fundamentals intrinsic value estimate: `buy = IV × 0.80` (20% margin of safety), `sell = IV × 1.15`
3. Last resort: `buy = current_price × 0.90`, `sell = current_price × 1.20`

### Confidence Derivation
1. Use BuffettBrain's `conviction` (HIGH/MEDIUM/LOW) if available
2. Fallback from composite score: `≥0.70 or ≤0.30 → HIGH`, `≥0.60 or ≤0.40 → MEDIUM`, else `LOW`

### Rationale
1. Use BuffettBrain's `reasoning` string if available (GPT-4o Buffett voice)
2. Fallback: Assembles from fundamentals/technicals/sentiment summaries

### Output (Final Recommendation)
```json
{
  "ticker": "AAPL",
  "composite_score": 0.667,
  "action": "HOLD",
  "buy_price": 389.68,
  "sell_price": 560.16,
  "confidence": "MEDIUM",
  "rationale": "Apple is a remarkable company with a strong brand and loyal customer base, giving it a durable competitive advantage. However, at 35.5x P/E with only 4.1% revenue growth, there is limited margin of safety. The intrinsic value estimate is higher than current price, but Berkshire's position reduction warrants caution.",
  "backtest": {
    "status": "pending",
    "note": "Run warren backtest <ticker> to compute metrics."
  },
  "agent_scores": {
    "fundamentals": 0.748,
    "technicals": 0.475,
    "sentiment": 0.850,
    "thirteen_f": 0.620,
    "ontology": 0.723,
    "risk": 0.545
  }
}
```

---

## LLM Layer

**File:** `src/warren_brain/llm.py`

**Purpose:** Centralized GPT-4o client that handles the dataexpert.io proxy correctly.

### The Proxy Problem
The dataexpert.io proxy (`https://www.dataexpert.io/api/v1/openai`) always returns responses in SSE (Server-Sent Events) streaming format, even when `stream=False` is passed. Without `stream=True`, the OpenAI SDK receives a raw `data: {...}` string instead of a parsed response object, causing `AttributeError: 'str' object has no attribute 'choices'`.

### The Fix
The `complete()` function always passes `stream=True`:
```python
def complete(messages, model=None, temperature=None, json_mode=False) -> str:
    client = get_openai_client()
    kwargs = {
        "model": model or settings.openai_model,
        "temperature": temperature if temperature is not None else settings.openai_temperature,
        "messages": messages,
        "stream": True,                          # Always — proxy requires this
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    content = ""
    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return content
```

### Proxy Authentication
```python
def get_openai_client() -> OpenAI:
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
        kwargs["default_headers"] = {"x-session-id": "warren-brain"}  # required by proxy
    return OpenAI(**kwargs)
```

The `x-session-id` header is required by the dataexpert.io proxy. Without it, the proxy returns `403 Missing session identifier`.

---

## Data Infrastructure

### `data/market_data.py` — yfinance Wrappers
All yfinance calls are cached in SQLite before returning.

| Function | What it fetches | Cache key |
|----------|----------------|-----------|
| `fetch_price_history(ticker, period_years)` | Daily OHLCV DataFrame | `price:{period}:{ticker}` |
| `fetch_info(ticker)` | Full stock info dict (150+ fields) | `info:{ticker}` |
| `fetch_financials(ticker)` | Income statement, balance sheet, cash flow | `financials:{ticker}` |
| `fetch_key_metrics(ticker)` | ~25 most useful ratios extracted from info | Uses `fetch_info` |

### `data/massive.py` — Massive.com API Client
Polygon-compatible API. Auth via `?apiKey=` query parameter.

| Function | Endpoint | Cache TTL |
|----------|----------|-----------|
| `fetch_news_with_sentiment(ticker)` | `/v2/reference/news` | 30 min |
| `fetch_ticker_overview(ticker)` | `/v3/reference/tickers/{ticker}` | 24 hrs |
| `fetch_related_companies(ticker)` | `/v1/related-companies/{ticker}` | 24 hrs |
| `fetch_snapshot(ticker)` | `/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}` | 5 min |

All calls use `tenacity` retry with exponential backoff (3 attempts, 1-4 second wait). Non-2xx responses raise immediately.

### `data/news.py` — Finnhub + yfinance News
`fetch_news(ticker)`:
1. Tries Finnhub `/v1/company-news` if `FINNHUB_API_KEY` is set
2. Falls back to `yfinance.Ticker.news` if Finnhub fails or no key

`fetch_insider_sentiment(ticker)`:
- Calls Finnhub `/v1/stock/insider-sentiment`
- Returns latest monthly share purchase ratio (MSPR) and absolute share count change

### `data/edgar.py` — SEC EDGAR 13F Parser
`get_cik(ticker)`:
- Downloads `sec.gov/files/company_tickers.json` (master ticker→CIK map)
- Finds the matching entry and returns zero-padded 10-digit CIK

`fetch_13f_holdings(cik, max_filings=4)`:
1. Fetches `data.sec.gov/submissions/CIK{cik}.json` — institution's filing history
2. Finds 13F-HR forms and their accession numbers
3. For each filing, calls `_parse_13f_xml()`:
   - Downloads the filing index JSON to find the infotable XML filename
   - Downloads the XML
   - Parses all `<infoTable>` elements: name, CUSIP, value (×1000 for USD), shares
   - Returns sorted by `value_usd` descending

---

## Cache System

**File:** `src/warren_brain/data/cache.py`

A thin SQLite-backed TTL cache that prevents redundant API calls within a session and across sessions.

### Schema
```sql
CREATE TABLE cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,      -- JSON serialized
    expires_at REAL NOT NULL  -- Unix timestamp
)
```

### TTL Per Data Type
| Data type | TTL | Reason |
|-----------|-----|--------|
| Massive snapshot (price) | 5 minutes | Real-time price changes constantly |
| Massive news | 30 minutes | News articles published frequently during market hours |
| Massive overview | 24 hours | Company description/employees rarely changes |
| Massive related companies | 24 hours | Peer relationships are stable |
| yfinance info/financials | 1 hour (default) | Good balance between freshness and API rate limits |
| SEC EDGAR 13F | 1 hour (default) | Filed quarterly — very stable |
| News (Finnhub/yfinance) | 1 hour (default) | Articles don't change once published |

### Cache Singleton
```python
_cache: DataCache | None = None

def get_cache() -> DataCache:
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache
```

All data modules call `get_cache()` — same cache instance across all agents in a run.

---

## Database / Persistence

**File:** `src/warren_brain/data/database.py`

Supports SQLite (default, zero setup) and PostgreSQL/Supabase (production). Switched via environment variables.

### Connection URL Building
```python
def _build_url() -> str:
    if settings.db_host:
        # PostgreSQL — URL-encode password to handle special chars (@, #, %, etc.)
        password = quote_plus(settings.db_password)
        return (
            f"postgresql+psycopg2://{settings.db_user}:{password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
            f"?sslmode=require"   # Required by Supabase
        )
    return settings.database_url   # SQLite fallback
```

`quote_plus` from `urllib.parse` converts special characters like `MyP@ssword` → `MyP%40ssword`. Raw password is stored in `.env` without manual encoding — the code handles encoding automatically.

### SQLite vs PostgreSQL Engine Config
| Parameter | SQLite | PostgreSQL |
|-----------|--------|-----------|
| `check_same_thread` | False (Streamlit multi-thread) | N/A |
| `poolclass` | `StaticPool` (single connection) | Default (connection pool) |
| `pool_size` | N/A | 5 |
| `max_overflow` | N/A | 10 |
| `pool_pre_ping` | No | Yes (recycles stale connections) |
| PRAGMA | `foreign_keys=ON` | N/A |
| SSL | No | `sslmode=require` |

### ORM Models (`data/models.py`)

**AnalysisResult** — one row per ticker per run:
- `id`, `ticker`, `analyzed_at`
- `action`, `buy_price`, `sell_price`, `confidence`, `composite_score`, `rationale`, `moat_assessment` (denormalized for fast queries)
- `raw_output: JSON` — full pipeline output from all agents

**AgentSignal** — one row per agent per run:
- `analysis_id` (FK → AnalysisResult)
- `agent_name`, `score`, `summary`, `raw_output`
- Enables querying score history per agent over time

**BacktestResult** — linked to AnalysisResult or standalone:
- `buy_price_target`, `sell_price_target`, `years`
- `cagr`, `total_return`, `sharpe`, `sortino`, `max_drawdown`, `alpha_vs_spy`, `spy_cagr`, `annualized_volatility`
- `raw_output` — full backtest including trade log

---

## Backtest Engine

**File:** `src/warren_brain/backtest/engine.py`

**Purpose:** Simulate what would have happened if you had followed the Warren Brain recommendation over the past N years, and compare against SPY (buy-and-hold benchmark).

### Strategy
- **Buy signal:** Price drops to or below `buy_price` → invest 100% of cash
- **Sell signal:** Price rises to or above `sell_price` → sell 100% of position
- Repeat for the full historical period
- Start with $1.00 (normalized), track portfolio value each day

### Metrics Computed
| Metric | Formula |
|--------|---------|
| CAGR | `(final_value / initial_value)^(1/years) - 1` |
| Total Return | `final_value / initial_value - 1` |
| Sharpe | `mean(excess_returns) / std(excess_returns) × √252` |
| Sortino | `mean(excess_returns) / std(negative_excess_returns) × √252` |
| Max Drawdown | `min((cumulative - cummax) / cummax)` |
| Alpha vs SPY | `stock_CAGR - spy_CAGR` |
| Annualized Volatility | `daily_returns.std() × √252` |

Also returns a `trades` log: `[{"day": 45, "action": "BUY", "price": 192.30}, ...]`

### Current Status
The backtest is not automatically run during the main pipeline. `PortfolioManagerAgent._backtest_stub()` returns `{"status": "pending", "note": "Run warren backtest <ticker>..."}`. Running the backtest separately and storing results in `BacktestResult` is a pending integration.

---

## Streamlit UI

**File:** `src/warren_brain/ui/dashboard.py`

Run with: `warren ui` or `streamlit run src/warren_brain/ui/dashboard.py`

### UI Structure
**Sidebar:**
- Ticker input (comma-separated, e.g. "NVDA, AAPL, TSM")
- Analyze button
- Progress bar + status text per ticker

**Per-ticker tab:**
1. **Header row (4 metrics):** Action, Buy Price, Sell Price, Confidence
2. **Gauge chart (Plotly):** Composite score 0-1 with color zones (red <0.35, orange 0.35-0.55, light green 0.55-0.70, green 0.70-1.0)
3. **BuffettBrain rationale** + Key Risks (yellow warning cards) + Moat Assessment (info box)
4. **Agent Signal Scores:** 6 metric cards (score + delta from 0.5, red if below 0.5)
5. **Expandable sections:** Fundamentals Detail | Technicals Detail | Sentiment Detail | 13F/Institutional | Risk Metrics | Raw JSON

---

## Configuration & Agent Weights

**File:** `src/warren_brain/config.py`

Uses `pydantic-settings` to read from `.env` automatically. All fields have defaults.

```python
class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.1
    openai_base_url: str = ""        # proxy URL

    # APIs
    finnhub_api_key: str = ""
    massive_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # Database
    database_url: str = "sqlite:///warren_brain.db"
    db_host: str = ""                # set to use PostgreSQL
    db_user: str = "postgres"
    db_password: str = ""            # raw — quote_plus applied in database.py
    db_name: str = "postgres"
    db_port: int = 5432

    # Agent weights (must sum to 1.0)
    weight_fundamentals: float = 0.30
    weight_technicals: float = 0.20
    weight_sentiment: float = 0.15
    weight_thirteen_f: float = 0.15
    weight_ontology: float = 0.10
    weight_risk: float = 0.10

    # Data settings
    price_history_years: int = 5
    news_lookback_days: int = 30
```

---

## External APIs & Keys

| API | Key env var | Free tier | Used for |
|-----|------------|-----------|----------|
| OpenAI / dataexpert.io proxy | `OPENAI_API_KEY` | No (paid) | GPT-4o in BuffettBrain + SentimentAgent |
| Massive.com | `MASSIVE_API_KEY` | Yes (news, snapshot, overview, related) | Primary news sentiment, real-time price, company data |
| Finnhub | `FINNHUB_API_KEY` | Yes (limited) | News fallback, insider sentiment |
| SEC EDGAR | None required | Always free | 13F filings, CIK lookup |
| yfinance | None required | Always free | Financial ratios, price history, stock info |
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | Yes (limited) | Not yet wired (placeholder) |

**Massive.com free plan endpoints (confirmed working):**
- `/v2/reference/news` — news with per-ticker sentiment
- `/v1/related-companies/{ticker}` — related tickers
- `/v3/reference/tickers/{ticker}` — company overview
- `/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}` — real-time price

**Massive.com paid plan only (returns NOT_AUTHORIZED on free):**
- `/stocks/financials/v1/ratios`
- `/stocks/financials/v1/income-statements`
- Benzinga partner endpoints
- ETF Global partner endpoints

---

## Agent Score Weights (Default)

| Agent | Weight | Rationale |
|-------|--------|-----------|
| FundamentalsAgent | 30% | Buffett's first question: is the business good and cheap? |
| TechnicalsAgent | 20% | Entry timing — even great businesses have bad entry points |
| SentimentAgent | 15% | Market mood affects near-term price and risk |
| ThirteenFAgent | 15% | If Buffett himself holds it, that is signal |
| OntologyAgent | 10% | Ecosystem and business quality context |
| RiskAgent | 10% | Risk penalty — high volatility means capital can be lost |

All weights are configurable via `.env`. They must sum to 1.0.

---

## Live Pipeline Example — AAPL

Actual output from a live run on 2026-05-10:

**Agent scores:**
```
FundamentalsAgent  → 0.748  (high ROE, strong margin, but P/E = 35.5x is expensive)
TechnicalsAgent    → 0.475  (RSI neutral, above key MAs but only 10% from 52w high)
SentimentAgent     → 0.850  (VERY_BULLISH, 18/25 articles positive, Massive scored)
ThirteenFAgent     → 0.620  (Berkshire REDUCED by 56% — negative signal)
OntologyAgent      → 0.723  (164k employees, $3.2T market cap, 8 related tickers)
RiskAgent          → 0.545  (beta 1.19, 28% annualized vol, Sharpe 0.84)
```

**Composite score:** 0.667 (= 0.748×0.30 + 0.475×0.20 + 0.850×0.15 + 0.620×0.15 + 0.723×0.10 + 0.545×0.10)

**BuffettBrain decision:** HOLD, MEDIUM conviction

**Buy price:** $389.68 (BuffettBrain-generated)
**Sell price:** $560.16 (BuffettBrain-generated)

**Rationale (GPT-4o, Buffett voice):**
> "Apple is a remarkable company with a strong brand and loyal customer base, which gives it a significant moat. However, the current P/E ratio of 35.5x suggests that the market is pricing in a lot of future growth, leaving little margin of safety. While the intrinsic value estimate is higher than the current price, the high valuation and Berkshire's significant position reduction warrant a cautious approach."

**Time to complete:** ~45 seconds (6 agents in parallel, then 2 sequential)
