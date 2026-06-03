"""OntologyAgent — maps peers, supply chain, and ETF exposure via Massive.com.

Upgrades from hardcoded dict to live data:
  - /v1/related-companies/{ticker}     → real peers/co-movers
  - /v3/reference/tickers/{ticker}     → company description, sector, employees
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from warren_brain.agents.base import BaseAgent
from warren_brain.config import settings
from warren_brain.data.market_data import fetch_info
from warren_brain.data.massive import fetch_related_companies, fetch_ticker_overview

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState

SECTOR_ETFS: dict[str, list[str]] = {
    "Technology": ["QQQ", "XLK", "SMH", "SOXX"],
    "Healthcare": ["XLV", "IBB"],
    "Financials": ["XLF", "KRE"],
    "Energy": ["XLE", "VDE"],
    "Consumer Cyclical": ["XLY", "IYC"],
    "Consumer Defensive": ["XLP"],
    "Industrials": ["XLI"],
    "Materials": ["XLB"],
    "Utilities": ["XLU"],
    "Real Estate": ["VNQ", "IYR"],
    "Communication Services": ["XLC"],
}


class OntologyAgent(BaseAgent):
    name = "OntologyAgent"

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            use_massive = bool(settings.massive_api_key)

            if use_massive:
                return self._analyze_massive(ticker, state)
            else:
                return self._analyze_fallback(ticker)

        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _analyze_massive(self, ticker: str, state: "WarrenBrainState") -> dict:
        overview = fetch_ticker_overview(ticker)
        related = fetch_related_companies(ticker)

        sector = overview.get("sic_description") or fetch_info(ticker).get("sector", "Unknown")
        description = overview.get("description", "")
        employees = overview.get("total_employees")
        market_cap = overview.get("market_cap")
        etfs = SECTOR_ETFS.get(fetch_info(ticker).get("sector", ""), [])

        score = self._compute_score_massive(overview, related)

        return {
            "sector": sector,
            "description": description[:400] if description else "",
            "total_employees": employees,
            "market_cap": market_cap,
            "related_tickers": related,
            "relevant_etfs": etfs,
            "score": score,
            "data_source": "massive",
            "summary": self._summarize_massive(ticker, overview, related, etfs, score),
        }

    def _compute_score_massive(self, overview: dict, related: list[str]) -> float:
        signals = []

        # More related companies = more ecosystem connections = higher quality business
        rel_count = len(related)
        if rel_count >= 8:
            signals.append(0.70)
        elif rel_count >= 4:
            signals.append(0.60)
        elif rel_count > 0:
            signals.append(0.50)
        else:
            signals.append(0.40)

        # Large, established companies (>10k employees) score higher
        employees = overview.get("total_employees") or 0
        if employees >= 50_000:
            signals.append(0.72)
        elif employees >= 10_000:
            signals.append(0.65)
        elif employees >= 1_000:
            signals.append(0.55)
        else:
            signals.append(0.45)

        # Market cap: large-cap (>$10B) preferred by Buffett
        market_cap = overview.get("market_cap") or 0
        if market_cap >= 200e9:
            signals.append(0.75)
        elif market_cap >= 10e9:
            signals.append(0.65)
        elif market_cap >= 1e9:
            signals.append(0.50)
        else:
            signals.append(0.35)

        return round(sum(signals) / len(signals), 3) if signals else 0.5

    def _summarize_massive(
        self,
        ticker: str,
        overview: dict,
        related: list[str],
        etfs: list[str],
        score: float,
    ) -> str:
        parts = []
        name = overview.get("name", ticker)
        if desc := overview.get("description", ""):
            parts.append(desc[:250].rstrip() + "...")
        if related:
            parts.append(f"Related companies: {', '.join(related[:6])}.")
        if etfs:
            parts.append(f"Relevant ETFs: {', '.join(etfs[:3])}.")
        if emp := overview.get("total_employees"):
            parts.append(f"Employees: {emp:,}.")
        parts.append(f"Ontology score: {score:.2f}/1.00.")
        return " ".join(parts)

    # ── Fallback (no Massive key) ─────────────────────────────────────────────

    _SECTOR_RELATIONSHIPS: dict[str, dict] = {
        "NVDA": {"peers": ["AMD", "INTC", "QCOM"], "suppliers": ["ASML", "AMAT", "KLAC", "LRCX"]},
        "AAPL": {"peers": ["MSFT", "GOOGL", "SSNLF"], "suppliers": ["TSM", "QCOM", "AVGO"]},
        "TSM":  {"peers": ["INTC", "SMIC"], "suppliers": ["ASML", "AMAT", "KLAC"]},
        "AMD":  {"peers": ["NVDA", "INTC"], "suppliers": ["TSMC", "ASML"]},
        "MSFT": {"peers": ["GOOGL", "AMZN", "CRM"]},
        "OXY":  {"peers": ["CVX", "XOM", "COP"]},
    }

    def _analyze_fallback(self, ticker: str) -> dict:
        info = fetch_info(ticker)
        sector = info.get("sector", "Unknown")
        relationships = self._SECTOR_RELATIONSHIPS.get(ticker.upper(), {})
        etfs = SECTOR_ETFS.get(sector, [])

        return {
            "sector": sector,
            "relationships": relationships,
            "relevant_etfs": etfs,
            "score": 0.5,
            "data_source": "static",
            "summary": self._summarize_fallback(ticker, sector, relationships, etfs),
        }

    def _summarize_fallback(self, ticker: str, sector: str, rel: dict, etfs: list[str]) -> str:
        parts = [f"{ticker} — sector: {sector}."]
        if suppliers := rel.get("suppliers"):
            parts.append(f"Key suppliers: {', '.join(suppliers)}.")
        if peers := rel.get("peers"):
            parts.append(f"Key peers: {', '.join(peers)}.")
        if etfs:
            parts.append(f"Relevant ETFs: {', '.join(etfs[:3])}.")
        parts.append("Ontology score: 0.50/1.00 (static fallback — set MASSIVE_API_KEY for live data).")
        return " ".join(parts)
