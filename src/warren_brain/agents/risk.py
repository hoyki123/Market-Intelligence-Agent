"""RiskAgent — volatility, beta, VaR, and drawdown assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from warren_brain.agents.base import BaseAgent
from warren_brain.data.market_data import fetch_price_history

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState

SPY_TICKER = "SPY"


class RiskAgent(BaseAgent):
    name = "RiskAgent"

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            stock_df = fetch_price_history(ticker, period_years=3)
            spy_df = fetch_price_history(SPY_TICKER, period_years=3)

            if stock_df.empty:
                return {"error": "No price history", "score": 0.5}

            metrics = self._compute_metrics(stock_df, spy_df)
            score = self._compute_score(metrics)

            return {
                "metrics": metrics,
                "score": score,
                "summary": self._summarize(ticker, metrics, score),
            }
        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _compute_metrics(self, stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> dict:
        stock_close = stock_df["Close"].astype(float)

        returns = stock_close.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252))

        # Beta vs SPY
        beta = self._compute_beta(stock_close, spy_df)

        # VaR (95%, 1-day)
        var_95 = float(np.percentile(returns, 5))

        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # Sharpe (annualized, risk-free ~4.5%)
        rf_daily = 0.045 / 252
        excess = returns - rf_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

        # Sortino (downside deviation only)
        downside = returns[returns < 0]
        sortino = (
            float(excess.mean() / (downside.std() * np.sqrt(252)))
            if not downside.empty and downside.std() > 0
            else 0
        )

        return {
            "annualized_volatility": round(ann_vol, 4),
            "beta": round(beta, 3) if beta is not None else None,
            "var_95_1d": round(var_95, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "trading_days": len(returns),
        }

    def _compute_beta(self, stock_close: pd.Series, spy_df: pd.DataFrame) -> float | None:
        try:
            spy_close = spy_df["Close"].astype(float)
            stock_returns = stock_close.pct_change().dropna()
            spy_returns = spy_close.pct_change().dropna()

            # Align on common dates
            combined = pd.concat([stock_returns, spy_returns], axis=1).dropna()
            combined.columns = ["stock", "spy"]

            cov = np.cov(combined["stock"], combined["spy"])
            return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else None
        except Exception:
            return None

    def _compute_score(self, m: dict) -> float:
        """Higher score = lower risk (more Buffett-friendly)."""
        signals = []

        # Volatility: < 20% annual is low-risk
        vol = m.get("annualized_volatility")
        if vol is not None:
            signals.append(1.0 - self._score(vol, 0.10, 0.70))

        # Beta: close to 1.0 or < 1.0 preferred
        beta = m.get("beta")
        if beta is not None:
            signals.append(1.0 - self._score(abs(beta - 0.8), 0, 1.5))

        # Max drawdown: smaller is better
        mdd = m.get("max_drawdown")
        if mdd is not None:
            signals.append(1.0 - self._score(abs(mdd), 0, 0.60))

        # Sharpe: higher is better
        sharpe = m.get("sharpe_ratio")
        if sharpe is not None:
            signals.append(self._score(sharpe, 0, 3.0))

        return round(sum(signals) / len(signals), 3) if signals else 0.5

    def _summarize(self, ticker: str, m: dict, score: float) -> str:
        parts = [f"{ticker} risk profile:"]
        if m.get("beta"):
            parts.append(f"Beta: {m['beta']:.2f}.")
        if m.get("annualized_volatility"):
            parts.append(f"Annual volatility: {m['annualized_volatility']:.1%}.")
        if m.get("var_95_1d"):
            parts.append(f"VaR(95%, 1d): {m['var_95_1d']:.1%}.")
        if m.get("max_drawdown"):
            parts.append(f"Max drawdown: {m['max_drawdown']:.1%}.")
        if m.get("sharpe_ratio"):
            parts.append(f"Sharpe: {m['sharpe_ratio']:.2f}.")
        parts.append(f"Risk score (higher = safer): {score:.2f}/1.00.")
        return " ".join(parts)
