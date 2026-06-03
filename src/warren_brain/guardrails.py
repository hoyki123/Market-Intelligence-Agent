"""Centralized guardrails for Warren Brain.

All validation, sanity checks, and safety rules live here.
Import from this module — never duplicate these checks in agents or UI.
"""

from __future__ import annotations

import sqlite3
import time

# ── Known leveraged / inverse ETFs ───────────────────────────────────────────

LEVERAGED_TICKERS: set[str] = {
    "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "SPXS", "UPRO", "SPXU",
    "UVXY", "SVXY", "LABU", "LABD", "FNGU", "FNGD", "TECL", "TECS",
    "NAIL", "DRN", "DRV", "UDOW", "SDOW", "TNA", "TZA", "NUGT", "DUST",
    "JNUG", "JDST", "FAS", "FAZ", "ERX", "ERY", "CURE", "HIBL", "HIBS",
    "DPST", "WEBL", "WEBS", "WANT", "BNKU", "DFEN", "RETL", "PILL",
}

MAX_TICKERS_PER_RUN = 5
MAX_CHAT_LENGTH = 500
HIGH_BETA_THRESHOLD = 2.0
HIGH_VOL_THRESHOLD = 0.60
HIGH_DRAWDOWN_THRESHOLD = 0.50
STALE_ANALYSIS_HOURS = 2


# ── 1. Input validation ───────────────────────────────────────────────────────

def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Checks format and market data existence."""
    t = ticker.upper().strip()

    if not t:
        return False, "Ticker cannot be empty."
    if not t.replace(".", "").replace("-", "").isalpha() or len(t) > 6:
        return False, f"'{t}' is not a valid ticker format (letters only, max 6 chars)."

    try:
        import yfinance as yf
        info = yf.Ticker(t).info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return False, f"No market data found for '{t}'. Check the ticker symbol."
        return True, ""
    except Exception as e:
        return False, f"Could not verify '{t}': {e}"


def validate_batch(tickers: list[str]) -> tuple[list[str], str | None]:
    """Cap batch at MAX_TICKERS_PER_RUN. Returns (capped_list, warning_or_None)."""
    if len(tickers) > MAX_TICKERS_PER_RUN:
        return (
            tickers[:MAX_TICKERS_PER_RUN],
            f"Batch capped at {MAX_TICKERS_PER_RUN} tickers. "
            f"Dropped: {', '.join(tickers[MAX_TICKERS_PER_RUN:])}",
        )
    return tickers, None


def sanitize_chat_input(question: str) -> tuple[bool, str]:
    """Return (is_valid, error_message) for a chat question."""
    if not question or not question.strip():
        return False, "Please enter a question."
    if len(question) > MAX_CHAT_LENGTH:
        return False, f"Question too long (max {MAX_CHAT_LENGTH} chars). Please be more concise."

    off_topic = ["recipe", "weather", "sports score", "movie", "song lyrics", "who won the game"]
    q = question.lower()
    for trigger in off_topic:
        if trigger in q:
            return False, "I can only answer questions about this stock and investment analysis."

    return True, ""


# ── 2. LLM output validation ──────────────────────────────────────────────────

def validate_llm_output(rec: dict, current_price: float | None = None) -> tuple[dict, list[str]]:
    """
    Validate and auto-fix LLM recommendation output.
    Returns (fixed_rec, list_of_warnings).
    """
    warnings: list[str] = []
    rec = dict(rec)  # don't mutate original

    # Required fields
    if rec.get("action") not in ("BUY", "SELL", "HOLD"):
        warnings.append(f"Invalid action '{rec.get('action')}' — defaulted to HOLD.")
        rec["action"] = "HOLD"

    if not rec.get("conviction") and not rec.get("confidence"):
        warnings.append("Missing conviction/confidence — defaulted to LOW.")
        rec["conviction"] = "LOW"

    if not rec.get("reasoning") and not rec.get("rationale"):
        warnings.append("No reasoning provided by LLM.")

    # Price sanity
    buy = rec.get("buy_price")
    sell = rec.get("sell_price")

    if buy is not None and sell is not None:
        try:
            buy, sell = float(buy), float(sell)
            if buy <= 0 or sell <= 0:
                warnings.append("Prices must be positive — cleared invalid prices.")
                rec["buy_price"] = rec["sell_price"] = None
            elif buy >= sell:
                warnings.append(f"Buy ${buy:,.2f} ≥ sell ${sell:,.2f} — prices swapped.")
                rec["buy_price"], rec["sell_price"] = min(buy, sell), max(buy, sell)
            elif current_price and sell > current_price * 12:
                warnings.append(
                    f"Sell price ${sell:,.0f} is >12× current price ${current_price:.2f} — may be unrealistic."
                )
        except (TypeError, ValueError):
            warnings.append("Could not parse buy/sell prices as numbers — cleared.")
            rec["buy_price"] = rec["sell_price"] = None

    return rec, warnings


def check_action_vs_composite(action: str, composite: float) -> str | None:
    """Return a warning string if LLM action contradicts the composite score."""
    if action == "BUY" and composite < 0.40:
        return f"LLM recommends BUY but composite score is only {composite:.2f} — signals conflict."
    if action == "SELL" and composite > 0.60:
        return f"LLM recommends SELL but composite score is {composite:.2f} — signals conflict."
    return None


# ── 3. Data quality ───────────────────────────────────────────────────────────

def check_data_quality(result: dict) -> tuple[str, list[str]]:
    """
    Returns (quality_level, warnings).
    quality_level: "OK" | "DEGRADED" | "INSUFFICIENT"
    """
    agent_keys = ["fundamentals", "technicals", "sentiment", "thirteen_f", "ontology", "risk"]
    errors = [k for k in agent_keys if isinstance(result.get(k), dict) and "error" in result[k]]

    warnings: list[str] = []
    if errors:
        warnings.append(f"Data unavailable from: {', '.join(errors)}.")

    # Sparse fundamentals check
    metrics = (result.get("fundamentals") or {}).get("metrics", {})
    missing = [f for f in ("pe_ratio", "roe", "profit_margin") if not metrics.get(f)]
    if len(missing) >= 2:
        warnings.append(f"Missing key fundamental metrics: {', '.join(missing)}.")

    if len(errors) >= 3:
        return "INSUFFICIENT", warnings
    if errors or missing:
        return "DEGRADED", warnings
    return "OK", warnings


# ── 4. Risk warnings ──────────────────────────────────────────────────────────

def check_high_risk(result: dict) -> list[str]:
    """Return list of high-risk warning strings (empty if none)."""
    warnings: list[str] = []
    metrics = (result.get("risk") or {}).get("metrics", {})

    beta = metrics.get("beta")
    vol = metrics.get("annualized_volatility")
    mdd = metrics.get("max_drawdown")

    if beta and beta > HIGH_BETA_THRESHOLD:
        warnings.append(
            f"High beta ({beta:.2f}) — moves >2× the market. Severe downside in corrections."
        )
    if vol and vol > HIGH_VOL_THRESHOLD:
        warnings.append(
            f"Extreme annual volatility ({vol:.0%}). Position sizing and stop-losses are critical."
        )
    if mdd and abs(mdd) > HIGH_DRAWDOWN_THRESHOLD:
        warnings.append(
            f"Max drawdown of {mdd:.0%} over 3 years — has lost >50% of value before."
        )
    return warnings


def is_leveraged_etf(ticker: str) -> bool:
    return ticker.upper() in LEVERAGED_TICKERS


# ── 5. Stale data detection ───────────────────────────────────────────────────

def cache_age_seconds(ticker: str, is_mcp: bool = False) -> int | None:
    """Return age in seconds of the cached analysis, or None if not cached."""
    key = f"{'mcp' if is_mcp else 'analysis'}:{ticker.upper()}"
    try:
        with sqlite3.connect("warren_brain.db") as conn:
            row = conn.execute(
                "SELECT expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row:
                ttl_remaining = row[0] - time.time()
                age = 3600 - ttl_remaining  # TTL is 1 hour
                return max(0, int(age))
    except Exception:
        pass
    return None


# ── 6. Chat guardrails ────────────────────────────────────────────────────────

CHAT_GUARDRAIL_SUFFIX = """

GUARDRAILS — follow these strictly:
- Never say a stock "will" reach a price or "will" go up/down. Use "could", "historically", "signals suggest".
- Never recommend a specific dollar amount or portfolio percentage (e.g. "put $10,000 in" or "allocate 20%").
- Always note "this is not financial advice" when giving specific entry or exit guidance.
- If you lack sufficient data to answer confidently, say so rather than speculating.
"""


def apply_chat_guardrails(system_prompt: str) -> str:
    return system_prompt + CHAT_GUARDRAIL_SUFFIX


# ── 7. MCP tool validation ────────────────────────────────────────────────────

def validate_tool_input(tool_name: str, tool_input: dict, expected_ticker: str) -> tuple[bool, str]:
    """Ensure Claude is calling tools for the right ticker."""
    ticker = tool_input.get("ticker", "").upper()
    expected = expected_ticker.upper()

    if not ticker:
        return False, f"Tool '{tool_name}' called without a ticker."
    if ticker != expected:
        return False, (
            f"Tool '{tool_name}' called for '{ticker}' but analysis is for '{expected}'. "
            f"Correcting to '{expected}'."
        )
    return True, ""
