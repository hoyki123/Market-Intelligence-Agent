"""FundamentalsAgent — evaluates valuation, profitability, and financial health.

Data sources:
  - yfinance: financial ratios (P/E, ROE, D/E, FCF, margins, growth)
  - Massive /v3/reference/tickers: company description, employees, real market cap
  - Massive /v2/snapshot: real-time price (more accurate than yfinance delayed price)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from warren_brain.agents.base import BaseAgent
from warren_brain.config import settings
from warren_brain.data.market_data import fetch_key_metrics
from warren_brain.data.massive import fetch_ticker_overview, fetch_snapshot

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState


class FundamentalsAgent(BaseAgent):
    name = "FundamentalsAgent"

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            metrics = fetch_key_metrics(ticker)

            # Enrich with Massive data when available
            if settings.massive_api_key:
                metrics = self._enrich_with_massive(ticker, metrics)

            score = self._compute_score(metrics)
            intrinsic = self._estimate_intrinsic_value(metrics)
            summary = self._summarize(metrics, score, intrinsic)

            return {
                "metrics": metrics,
                "intrinsic_value_estimate": intrinsic,
                "score": score,
                "summary": summary,
            }
        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _enrich_with_massive(self, ticker: str, metrics: dict) -> dict:
        """
        Layer Massive data on top of yfinance metrics.
        Massive has more reliable real-time price and richer company description.
        Financial ratios (P/E, ROE etc.) still come from yfinance — Massive
        requires a paid plan for those endpoints.
        """
        try:
            overview = fetch_ticker_overview(ticker)
            snapshot = fetch_snapshot(ticker)

            # Real-time price from snapshot (more accurate than yfinance delayed)
            day = snapshot.get("day", {})
            if close := day.get("c"):
                metrics["current_price"] = close
            if vwap := day.get("vw"):
                metrics["vwap"] = float(vwap)

            # Richer company context from overview
            if desc := overview.get("description"):
                metrics["description"] = desc
            if emp := overview.get("total_employees"):
                metrics["total_employees"] = emp
            if mc := overview.get("market_cap"):
                metrics["market_cap"] = mc
            if homepage := overview.get("homepage_url"):
                metrics["homepage_url"] = homepage

        except Exception:
            pass  # Massive enrichment is best-effort; yfinance data still used

        return metrics

    def _compute_score(self, m: dict) -> float:
        """
        Composite fundamental score [0, 1].
        Higher is more attractive from a value/quality standpoint.
        """
        signals = []

        # Profitability: ROE > 15% is good
        if (roe := m.get("roe")) is not None:
            signals.append(self._score(roe, 0.0, 0.40))

        # Profit margin > 20% is excellent
        if (pm := m.get("profit_margin")) is not None:
            signals.append(self._score(pm, 0.0, 0.30))

        # P/E: lower is cheaper (Buffett prefers < 20); invert the score
        if (pe := m.get("pe_ratio")) is not None and pe > 0:
            signals.append(1.0 - self._score(pe, 5, 60))

        # Revenue growth: 15%+ is strong
        if (rg := m.get("revenue_growth")) is not None:
            signals.append(self._score(rg, -0.10, 0.40))

        # Debt-to-equity: lower is safer
        if (dte := m.get("debt_to_equity")) is not None:
            signals.append(1.0 - self._score(dte, 0, 200))

        # Free cash flow positive is good
        if (fcf := m.get("free_cashflow")) is not None:
            signals.append(1.0 if fcf > 0 else 0.2)

        return round(sum(signals) / len(signals), 3) if signals else 0.5

    def _estimate_intrinsic_value(self, m: dict) -> float | None:
        """
        Simplified DCF via earnings power:
        IV = EPS * (8.5 + 2*g) * 4.4 / Y  (Benjamin Graham formula)
        where g = expected earnings growth (%), Y = current AAA bond yield (~4.5%)
        """
        current_price = m.get("current_price")
        if not current_price:
            return None

        # Use forward P/E with estimated EPS as proxy
        forward_pe = m.get("forward_pe")
        eps_growth = m.get("earnings_growth") or 0.07  # default 7% growth

        if forward_pe and forward_pe > 0 and current_price:
            eps = current_price / forward_pe
            g = eps_growth * 100
            y = 4.5  # approximate AAA yield
            iv = eps * (8.5 + 2 * g) * 4.4 / y
            return round(iv, 2)

        return None

    def _summarize(self, m: dict, score: float, intrinsic: float | None) -> str:
        parts = [f"{m['name']} ({m['ticker']}) — sector: {m['sector']}."]

        if m.get("roe"):
            parts.append(f"ROE: {m['roe']:.1%}.")
        if m.get("profit_margin"):
            parts.append(f"Net margin: {m['profit_margin']:.1%}.")
        if m.get("pe_ratio"):
            parts.append(f"P/E: {m['pe_ratio']:.1f}x.")
        if m.get("debt_to_equity"):
            parts.append(f"D/E: {m['debt_to_equity']:.1f}.")
        if intrinsic and m.get("current_price"):
            upside = (intrinsic / m["current_price"] - 1) * 100
            parts.append(
                f"Estimated intrinsic value: ${intrinsic:.2f} "
                f"({'↑' if upside > 0 else '↓'}{abs(upside):.1f}% vs current ${m['current_price']:.2f})."
            )

        parts.append(f"Fundamental score: {score:.2f}/1.00.")
        return " ".join(parts)
