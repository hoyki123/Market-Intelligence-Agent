"""SentimentAgent — news sentiment via Massive.com (pre-scored) + GPT-4o summary.

Data flow:
  1. Fetch articles from Massive /v2/reference/news (includes per-ticker sentiment scores)
  2. Aggregate positive/neutral/negative counts → composite score
  3. Use GPT-4o only to produce a 1-2 sentence summary from the pre-scored reasoning
     (much cheaper than asking LLM to score raw headlines)
  Fallback: Finnhub → yfinance news if Massive key not set.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from warren_brain.agents.base import BaseAgent
from warren_brain.config import settings
from warren_brain.llm import complete
from warren_brain.data.massive import (
    fetch_news_with_sentiment,
    aggregate_sentiment,
)
from warren_brain.data.news import fetch_news, fetch_insider_sentiment

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState


class SentimentAgent(BaseAgent):
    name = "SentimentAgent"

    pass

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            use_massive = bool(settings.massive_api_key)

            if use_massive:
                articles = fetch_news_with_sentiment(ticker)
                agg = aggregate_sentiment(articles)
                summary = self._llm_summary_from_scored(ticker, agg, articles)
                insider = fetch_insider_sentiment(ticker)
                return {
                    "score": self._label_to_score(agg["label"]),
                    "sentiment_label": agg["label"],
                    "sentiment_score_raw": agg["score"],
                    "counts": agg["counts"],
                    "key_themes": summary.get("themes", []),
                    "article_count": agg["article_count"],
                    "insider_sentiment": insider,
                    "data_source": "massive",
                    "summary": summary.get("summary", ""),
                }
            else:
                return self._analyze_fallback(ticker)

        except Exception as e:
            return {"error": str(e), "score": 0.5, "sentiment_label": "NEUTRAL"}

    def _llm_summary_from_scored(self, ticker: str, agg: dict, articles: list[dict]) -> dict:
        counts = agg["counts"]
        reasoning_snippets = "\n".join(f"- {r}" for r in agg["reasoning"])
        top_titles = "\n".join(
            f"- [{a['sentiment'].upper()}] {a['title']}" for a in articles[:10]
        )
        content = complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial news analyst. Synthesise pre-scored sentiment data into themes and a brief summary. Respond with JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Ticker: {ticker}\n"
                        f"Sentiment counts: {counts['positive']} positive, "
                        f"{counts['neutral']} neutral, {counts['negative']} negative\n\n"
                        f"Top headlines:\n{top_titles}\n\n"
                        f"Analyst reasoning snippets:\n{reasoning_snippets}\n\n"
                        "Return JSON with:\n"
                        '  "themes": list of up to 3 key themes (strings)\n'
                        '  "summary": 1-2 sentence synthesis of the overall sentiment picture\n'
                    ),
                },
            ],
            temperature=0.0,
            json_mode=True,
        )
        return json.loads(content)

    def _label_to_score(self, label: str) -> float:
        return {
            "VERY_BULLISH": 0.85,
            "BULLISH": 0.65,
            "NEUTRAL": 0.50,
            "BEARISH": 0.35,
            "VERY_BEARISH": 0.15,
        }.get(label, 0.5)

    # ── Fallback (no Massive key) ─────────────────────────────────────────────

    def _analyze_fallback(self, ticker: str) -> dict:
        articles = fetch_news(ticker)
        insider = fetch_insider_sentiment(ticker)

        if not articles:
            return {
                "score": 0.5,
                "sentiment_label": "NEUTRAL",
                "article_count": 0,
                "data_source": "none",
                "summary": f"No recent news found for {ticker}.",
            }

        headlines = [a["headline"] for a in articles[:20] if a["headline"]]
        result = self._llm_score_headlines(ticker, headlines)
        return {
            "score": self._label_to_score(result["label"]),
            "sentiment_label": result["label"],
            "sentiment_score_raw": result["score"],
            "key_themes": result["themes"],
            "article_count": len(articles),
            "insider_sentiment": insider,
            "data_source": "finnhub/yfinance",
            "summary": result["summary"],
        }

    def _llm_score_headlines(self, ticker: str, headlines: list[str]) -> dict:
        headlines_text = "\n".join(f"- {h}" for h in headlines)
        content = complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial news sentiment analyst. Respond with JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Ticker: {ticker}\n\nHeadlines:\n{headlines_text}\n\n"
                        "Return JSON with:\n"
                        '  "label": VERY_BULLISH | BULLISH | NEUTRAL | BEARISH | VERY_BEARISH\n'
                        '  "score": float -1.0 to 1.0\n'
                        '  "themes": list of up to 3 themes\n'
                        '  "summary": 1-2 sentences\n'
                    ),
                },
            ],
            temperature=0.0,
            json_mode=True,
        )
        return json.loads(content)
