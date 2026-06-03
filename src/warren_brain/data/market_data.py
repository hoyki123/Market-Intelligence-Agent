"""Market data fetching via yfinance with SQLite caching."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from warren_brain.config import settings
from warren_brain.data.cache import get_cache


def _cache_key(prefix: str, ticker: str) -> str:
    return f"{prefix}:{ticker.upper()}"


def fetch_price_history(ticker: str, period_years: int | None = None) -> pd.DataFrame:
    years = period_years or settings.price_history_years
    period = f"{years}y"

    cache = get_cache()
    key = _cache_key(f"price:{period}", ticker)
    cached = cache.get(key)
    if cached:
        return pd.DataFrame(cached)

    stock = yf.Ticker(ticker.upper())
    df = stock.history(period=period, auto_adjust=True)
    df.index = df.index.strftime("%Y-%m-%d")
    cache.set(key, df.to_dict())
    return df


def fetch_info(ticker: str) -> dict:
    cache = get_cache()
    key = _cache_key("info", ticker)
    cached = cache.get(key)
    if cached:
        return cached

    stock = yf.Ticker(ticker.upper())
    info = stock.info or {}
    cache.set(key, info)
    return info


def fetch_financials(ticker: str) -> dict:
    """Returns income statement, balance sheet, and cash flow as dicts."""
    cache = get_cache()
    key = _cache_key("financials", ticker)
    cached = cache.get(key)
    if cached:
        return cached

    stock = yf.Ticker(ticker.upper())
    result = {
        "income_stmt": stock.income_stmt.to_dict() if stock.income_stmt is not None else {},
        "balance_sheet": stock.balance_sheet.to_dict() if stock.balance_sheet is not None else {},
        "cash_flow": stock.cash_flow.to_dict() if stock.cash_flow is not None else {},
    }
    cache.set(key, result)
    return result


def fetch_key_metrics(ticker: str) -> dict:
    """Extracts the most useful fundamental ratios from yfinance info."""
    info = fetch_info(ticker)

    def safe(key: str, default=None):
        val = info.get(key)
        return val if val not in (None, "None", "N/A") else default

    return {
        "ticker": ticker.upper(),
        "name": safe("longName", ticker),
        "sector": safe("sector", "Unknown"),
        "industry": safe("industry", "Unknown"),
        "market_cap": safe("marketCap"),
        "enterprise_value": safe("enterpriseValue"),
        "pe_ratio": safe("trailingPE"),
        "forward_pe": safe("forwardPE"),
        "peg_ratio": safe("pegRatio"),
        "price_to_book": safe("priceToBook"),
        "price_to_sales": safe("priceToSalesTrailing12Months"),
        "ev_to_ebitda": safe("enterpriseToEbitda"),
        "roe": safe("returnOnEquity"),
        "roa": safe("returnOnAssets"),
        "profit_margin": safe("profitMargins"),
        "operating_margin": safe("operatingMargins"),
        "revenue_growth": safe("revenueGrowth"),
        "earnings_growth": safe("earningsGrowth"),
        "debt_to_equity": safe("debtToEquity"),
        "current_ratio": safe("currentRatio"),
        "quick_ratio": safe("quickRatio"),
        "free_cashflow": safe("freeCashflow"),
        "dividend_yield": safe("dividendYield"),
        "beta": safe("beta"),
        "52w_high": safe("fiftyTwoWeekHigh"),
        "52w_low": safe("fiftyTwoWeekLow"),
        "current_price": safe("currentPrice") or safe("regularMarketPrice"),
        "analyst_target": safe("targetMeanPrice"),
        "recommendation_mean": safe("recommendationMean"),
    }
