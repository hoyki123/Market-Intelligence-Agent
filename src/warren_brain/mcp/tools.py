"""MCP tool definitions and executor for the Warren Brain dynamic agent.

Each tool wraps an existing agent/data function. Claude decides which
tools to call and in what order — it does not run all of them always.
"""

from __future__ import annotations

from typing import Any


TOOLS = [
    {
        "name": "get_fundamentals",
        "description": (
            "Fetches fundamental financial metrics for a stock: P/E ratio, ROE, "
            "profit margin, debt/equity ratio, free cash flow, revenue growth, "
            "current price, and a Benjamin Graham intrinsic value estimate. "
            "Returns a composite quality/value score 0-1."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_technicals",
        "description": (
            "Computes technical indicators from 2 years of daily price history: "
            "RSI(14), MACD histogram, SMA 20/50/200, VWAP, ATR, 52-week high/low, "
            "and a BUY/SELL/NEUTRAL/STRONG_BUY/WEAK signal. "
            "Use this to assess entry timing and momentum."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_sentiment",
        "description": (
            "Fetches recent news articles with pre-scored sentiment from Massive.com "
            "and returns VERY_BULLISH/BULLISH/NEUTRAL/BEARISH/VERY_BEARISH label, "
            "positive/neutral/negative article counts, key themes, and "
            "insider sentiment (monthly share purchase ratio)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_institutional_holdings",
        "description": (
            "Checks Berkshire Hathaway's latest two 13F SEC filings for this ticker "
            "— position size in USD, share count, and trend (ADDED/HELD/REDUCED). "
            "Also returns general institutional ownership %, insider ownership %, "
            "and short ratio from yfinance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_company_overview",
        "description": (
            "Returns company description, employee count, market cap, related "
            "peer/supply-chain tickers, and relevant sector ETFs. "
            "Use this to understand the business model, competitive position, "
            "and ecosystem relationships."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_risk_metrics",
        "description": (
            "Computes risk metrics from 3 years of daily price history benchmarked "
            "against SPY: beta, annualized volatility, VaR(95% 1-day), max drawdown, "
            "Sharpe ratio, and Sortino ratio. Higher score = lower risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_realtime_price",
        "description": (
            "Fetches a real-time price snapshot from Massive.com: "
            "current price, OHLC for the day, VWAP, and volume. "
            "More accurate than yfinance which has a 15-minute delay."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> Any:
    """Dispatch a Claude tool call to the appropriate agent or data function."""
    ticker = tool_input.get("ticker", "").upper()

    if name == "get_fundamentals":
        from warren_brain.agents.fundamentals import FundamentalsAgent
        return FundamentalsAgent().analyze(ticker, {})

    if name == "get_technicals":
        from warren_brain.agents.technicals import TechnicalsAgent
        return TechnicalsAgent().analyze(ticker, {})

    if name == "get_sentiment":
        from warren_brain.agents.sentiment import SentimentAgent
        return SentimentAgent().analyze(ticker, {})

    if name == "get_institutional_holdings":
        from warren_brain.agents.thirteen_f import ThirteenFAgent
        return ThirteenFAgent().analyze(ticker, {})

    if name == "get_company_overview":
        from warren_brain.agents.ontology import OntologyAgent
        return OntologyAgent().analyze(ticker, {})

    if name == "get_risk_metrics":
        from warren_brain.agents.risk import RiskAgent
        return RiskAgent().analyze(ticker, {})

    if name == "get_realtime_price":
        from warren_brain.data.massive import fetch_snapshot
        return fetch_snapshot(ticker) or {"error": "No snapshot available"}

    return {"error": f"Unknown tool: {name}"}
