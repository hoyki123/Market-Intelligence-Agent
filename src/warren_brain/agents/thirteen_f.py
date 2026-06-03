"""13FAgent — tracks institutional ownership trends from SEC EDGAR filings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from warren_brain.agents.base import BaseAgent
from warren_brain.data.edgar import get_cik, fetch_13f_holdings

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState

# Well-known institutional investors and their CIKs
# Extend this list as needed
KNOWN_INSTITUTIONS = {
    "Berkshire Hathaway": "0001067983",
    "BlackRock": "0001364742",
    "Vanguard": "0000102909",
    "State Street": "0000093751",
}


class ThirteenFAgent(BaseAgent):
    name = "13FAgent"

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            # Check if Buffett/Berkshire holds the ticker — the flagship signal
            berkshire_position = self._check_berkshire_position(ticker)

            # Generic institutional ownership signal
            institutional_signal = self._estimate_institutional_signal(ticker, state)

            score = self._compute_score(berkshire_position, institutional_signal)

            return {
                "berkshire_position": berkshire_position,
                "institutional_signal": institutional_signal,
                "score": score,
                "summary": self._summarize(ticker, berkshire_position, score),
            }
        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _check_berkshire_position(self, ticker: str) -> dict | None:
        """Check Berkshire Hathaway's latest 13F for this ticker."""
        cik = KNOWN_INSTITUTIONS["Berkshire Hathaway"]
        filings = fetch_13f_holdings(cik, max_filings=2)
        if not filings:
            return None

        ticker_upper = ticker.upper()
        latest = filings[0]
        prev = filings[1] if len(filings) > 1 else None

        def find_holding(filing: dict) -> dict | None:
            for h in filing.get("holdings", []):
                name = h.get("name", "").upper()
                if ticker_upper in name or name.startswith(ticker_upper):
                    return h
            return None

        current = find_holding(latest)
        if not current:
            return None

        result = {
            "period": latest["period"],
            "value_usd": current["value_usd"],
            "shares": current["shares"],
        }

        if prev:
            prev_holding = find_holding(prev)
            if prev_holding:
                pct_change = (current["shares"] - prev_holding["shares"]) / prev_holding["shares"] * 100
                result["shares_prev"] = prev_holding["shares"]
                result["pct_change"] = round(pct_change, 2)
                result["trend"] = "ADDED" if pct_change > 2 else ("REDUCED" if pct_change < -2 else "HELD")

        return result

    def _estimate_institutional_signal(self, ticker: str, state: "WarrenBrainState") -> dict:
        """
        Use yfinance institutional ownership data as a proxy.
        Full 13F cross-institution search requires a data provider (sec-api.io, Refinitiv).
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker.upper())
            info = stock.info or {}

            held_pct = info.get("heldPercentInstitutions")
            insider_pct = info.get("heldPercentInsiders")
            short_ratio = info.get("shortRatio")

            return {
                "institutional_ownership_pct": held_pct,
                "insider_ownership_pct": insider_pct,
                "short_ratio": short_ratio,
            }
        except Exception:
            return {}

    def _compute_score(self, berkshire: dict | None, inst: dict) -> float:
        signals = []

        # Buffett holding = strong signal
        if berkshire:
            trend = berkshire.get("trend", "HELD")
            if trend == "ADDED":
                signals.append(0.80)
            elif trend == "HELD":
                signals.append(0.65)
            elif trend == "REDUCED":
                signals.append(0.40)

        # High institutional ownership (> 70%) indicates quality
        held_pct = inst.get("institutional_ownership_pct")
        if held_pct is not None:
            signals.append(self._score(held_pct, 0.3, 0.9))

        # Insider ownership (> 5%) is positive (skin in the game)
        insider_pct = inst.get("insider_ownership_pct")
        if insider_pct is not None:
            signals.append(min(0.75, 0.40 + insider_pct * 3))

        # Short ratio: high short interest (> 8 days) is a bearish signal
        short_ratio = inst.get("short_ratio")
        if short_ratio is not None:
            signals.append(1.0 - self._score(short_ratio, 0, 15))

        return round(sum(signals) / len(signals), 3) if signals else 0.5

    def _summarize(self, ticker: str, berkshire: dict | None, score: float) -> str:
        parts = []
        if berkshire:
            trend = berkshire.get("trend", "HOLDS")
            val = berkshire["value_usd"] / 1e9
            parts.append(
                f"Berkshire Hathaway {trend} {ticker} — "
                f"${val:.2f}B position as of {berkshire['period']}."
            )
            if "pct_change" in berkshire:
                parts.append(f"Shares changed {berkshire['pct_change']:+.1f}% vs prior quarter.")
        else:
            parts.append(f"No Berkshire Hathaway position found for {ticker}.")
        parts.append(f"13F/Institutional score: {score:.2f}/1.00.")
        return " ".join(parts)
