"""News and sentiment data via Finnhub (free tier) with yfinance news as fallback."""

from __future__ import annotations

from datetime import datetime, timedelta

import requests

from warren_brain.config import settings
from warren_brain.data.cache import get_cache

FINNHUB_BASE = "https://finnhub.io/api/v1"


def fetch_news(ticker: str, days: int | None = None) -> list[dict]:
    """
    Returns recent news articles for a ticker.
    Tries Finnhub first, falls back to yfinance news.
    """
    lookback = days or settings.news_lookback_days
    cache = get_cache()
    key = f"news:{ticker.upper()}:{lookback}"
    cached = cache.get(key)
    if cached:
        return cached

    articles: list[dict] = []

    if settings.finnhub_api_key:
        articles = _fetch_finnhub_news(ticker, lookback)

    if not articles:
        articles = _fetch_yfinance_news(ticker)

    cache.set(key, articles)
    return articles


def _fetch_finnhub_news(ticker: str, days: int) -> list[dict]:
    end = datetime.now()
    start = end - timedelta(days=days)

    params = {
        "symbol": ticker.upper(),
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": settings.finnhub_api_key,
    }
    try:
        resp = requests.get(f"{FINNHUB_BASE}/company-news", params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        return [
            {
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "datetime": datetime.fromtimestamp(a.get("datetime", 0)).isoformat(),
                "sentiment": a.get("sentiment"),
            }
            for a in raw[:50]
        ]
    except Exception:
        return []


def _fetch_yfinance_news(ticker: str) -> list[dict]:
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker.upper())
        raw = stock.news or []
        return [
            {
                "headline": a.get("title", ""),
                "summary": "",
                "source": a.get("publisher", ""),
                "url": a.get("link", ""),
                "datetime": datetime.fromtimestamp(a.get("providerPublishTime", 0)).isoformat(),
                "sentiment": None,
            }
            for a in raw[:30]
        ]
    except Exception:
        return []


def fetch_insider_sentiment(ticker: str) -> dict | None:
    """Finnhub aggregate insider sentiment (requires paid tier for some tickers)."""
    if not settings.finnhub_api_key:
        return None

    cache = get_cache()
    key = f"insider_sentiment:{ticker.upper()}"
    cached = cache.get(key)
    if cached:
        return cached

    try:
        params = {"symbol": ticker.upper(), "token": settings.finnhub_api_key}
        resp = requests.get(f"{FINNHUB_BASE}/stock/insider-sentiment", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        latest = sorted(data, key=lambda x: (x.get("year", 0), x.get("month", 0)))[-1]
        result = {
            "year": latest.get("year"),
            "month": latest.get("month"),
            "mspr": latest.get("mspr"),  # monthly share purchase ratio
            "change": latest.get("change"),
        }
        cache.set(key, result)
        return result
    except Exception:
        return None
