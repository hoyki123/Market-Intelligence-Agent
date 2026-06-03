"""Massive.com API client — auth via ?apiKey= query param (Polygon-compatible).

Available on free plan:
  - News with per-ticker sentiment scores  (/v2/reference/news)
  - Related companies                       (/v1/related-companies/{ticker})
  - Ticker overview + description           (/v3/reference/tickers/{ticker})
  - Real-time snapshot (price/OHLC/volume)  (/v2/snapshot/locale/us/markets/stocks/tickers/{ticker})
  - OHLC aggregates                         (/v2/aggs/ticker/{ticker}/range/...)

NOT available on free plan (returns NOT_AUTHORIZED):
  - /stocks/financials/v1/ratios
  - /stocks/financials/v1/income-statements
  - Benzinga partner endpoints
  - ETF Global partner endpoints
"""

from __future__ import annotations

from datetime import datetime, timedelta

# Per-type TTLs (seconds)
TTL_SNAPSHOT   = 5 * 60       #  5 min  — real-time price, changes constantly
TTL_NEWS       = 30 * 60      # 30 min  — news refreshes frequently during market hours
TTL_OVERVIEW   = 24 * 3600    # 24 hrs  — company description/employees rarely changes
TTL_RELATED    = 24 * 3600    # 24 hrs  — related companies are stable

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from warren_brain.config import settings
from warren_brain.data.cache import get_cache

BASE_URL = "https://api.massive.com"


def _is_available() -> bool:
    return bool(settings.massive_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def _get(path: str, params: dict | None = None) -> dict:
    """Authenticated GET to Massive API. Raises on non-2xx."""
    all_params = {**(params or {}), "apiKey": settings.massive_api_key}
    resp = requests.get(f"{BASE_URL}{path}", params=all_params, timeout=12)
    resp.raise_for_status()
    return resp.json()


# ── News & Sentiment ──────────────────────────────────────────────────────────

def fetch_news_with_sentiment(ticker: str, days: int = 30, limit: int = 25) -> list[dict]:
    """
    Fetch recent news for a ticker with pre-scored per-ticker sentiment.
    Each article includes insights[].sentiment (positive/neutral/negative)
    and insights[].sentiment_reasoning — no LLM needed for scoring.
    """
    if not _is_available():
        return []

    cache = get_cache()
    key = f"massive:news:{ticker.upper()}:{days}"
    if cached := cache.get(key):
        return cached

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    data = _get("/v2/reference/news", {
        "ticker": ticker.upper(),
        "published_utc.gte": start.strftime("%Y-%m-%d"),
        "limit": limit,
        "order": "desc",
        "sort": "published_utc",
    })

    articles = []
    for item in data.get("results", []):
        # Extract per-ticker insight for this specific ticker
        insight = next(
            (i for i in item.get("insights", []) if i.get("ticker") == ticker.upper()),
            item.get("insights", [{}])[0] if item.get("insights") else {},
        )
        articles.append({
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "published_utc": item.get("published_utc", ""),
            "publisher": item.get("publisher", {}).get("name", ""),
            "url": item.get("article_url", ""),
            "tickers": item.get("tickers", []),
            "sentiment": insight.get("sentiment", "neutral"),
            "sentiment_reasoning": insight.get("sentiment_reasoning", ""),
        })

    cache.set(key, articles, ttl=TTL_NEWS)
    return articles


def aggregate_sentiment(articles: list[dict]) -> dict:
    """
    Compute aggregate sentiment stats from a list of Massive news articles.
    Returns score [0,1], label, counts, and key reasoning snippets.
    """
    if not articles:
        return {"score": 0.5, "label": "NEUTRAL", "counts": {}, "reasoning": []}

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    reasoning = []

    for a in articles:
        s = a.get("sentiment", "neutral").lower()
        counts[s] = counts.get(s, 0) + 1
        if r := a.get("sentiment_reasoning", ""):
            reasoning.append(r)

    total = sum(counts.values()) or 1
    # Weighted score: positive=1.0, neutral=0.5, negative=0.0
    raw = (counts["positive"] * 1.0 + counts["neutral"] * 0.5) / total

    if raw >= 0.70:
        label = "VERY_BULLISH"
    elif raw >= 0.58:
        label = "BULLISH"
    elif raw >= 0.42:
        label = "NEUTRAL"
    elif raw >= 0.30:
        label = "BEARISH"
    else:
        label = "VERY_BEARISH"

    return {
        "score": round(raw, 3),
        "label": label,
        "counts": counts,
        "reasoning": reasoning[:5],  # top 5 snippets for LLM context
        "article_count": len(articles),
    }


# ── Company Data ──────────────────────────────────────────────────────────────

def fetch_ticker_overview(ticker: str) -> dict:
    """Fetch company overview including description, market cap, employees."""
    if not _is_available():
        return {}

    cache = get_cache()
    key = f"massive:overview:{ticker.upper()}"
    if cached := cache.get(key):
        return cached

    data = _get(f"/v3/reference/tickers/{ticker.upper()}")
    result = data.get("results", {})
    cache.set(key, result, ttl=TTL_OVERVIEW)
    return result


def fetch_related_companies(ticker: str) -> list[str]:
    """Return list of related ticker symbols (peers, supply chain, co-movers)."""
    if not _is_available():
        return []

    cache = get_cache()
    key = f"massive:related:{ticker.upper()}"
    if cached := cache.get(key):
        return cached

    data = _get(f"/v1/related-companies/{ticker.upper()}")
    tickers = [r["ticker"] for r in data.get("results", []) if "ticker" in r]
    cache.set(key, tickers, ttl=TTL_RELATED)
    return tickers


def fetch_snapshot(ticker: str) -> dict:
    """Fetch real-time price snapshot (OHLC, volume, VWAP)."""
    if not _is_available():
        return {}

    # Snapshots change by the minute — short cache
    cache = get_cache()
    key = f"massive:snapshot:{ticker.upper()}"
    if cached := cache.get(key):
        return cached

    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}")
    result = data.get("ticker", {})
    cache.set(key, result, ttl=TTL_SNAPSHOT)
    return result
