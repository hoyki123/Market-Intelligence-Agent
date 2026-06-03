"""PortfolioManagerAgent — aggregates all signals into final recommendation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from warren_brain.agents.base import BaseAgent
from warren_brain.config import settings
from warren_brain.data.market_data import fetch_key_metrics

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState


class PortfolioManagerAgent(BaseAgent):
    name = "PortfolioManagerAgent"

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            composite = self._compute_composite_score(state)
            action = self._derive_action(composite, state)
            prices = self._derive_prices(ticker, state, action)
            confidence = self._derive_confidence(composite, state)
            backtest_stub = self._backtest_stub(state)

            return {
                "ticker": ticker,
                "composite_score": round(composite, 3),
                "action": action,
                "buy_price": prices["buy"],
                "sell_price": prices["sell"],
                "confidence": confidence,
                "rationale": self._build_rationale(ticker, state, action, composite),
                "backtest": backtest_stub,
                "agent_scores": self._collect_agent_scores(state),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e), "action": "HOLD", "confidence": "LOW"}

    def _compute_composite_score(self, state: "WarrenBrainState") -> float:
        weights = {
            "fundamentals": settings.weight_fundamentals,
            "technicals": settings.weight_technicals,
            "sentiment": settings.weight_sentiment,
            "thirteen_f": settings.weight_thirteen_f,
            "ontology": settings.weight_ontology,
            "risk": settings.weight_risk,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for key, weight in weights.items():
            agent_state = state.get(key, {})
            score = agent_state.get("score") if isinstance(agent_state, dict) else None
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _derive_action(self, composite: float, state: "WarrenBrainState") -> str:
        buffett = state.get("buffett_brain", {})
        llm_action = buffett.get("action", "HOLD") if isinstance(buffett, dict) else "HOLD"

        # Override with composite score if LLM is unavailable
        if llm_action in ("BUY", "SELL", "HOLD"):
            return llm_action

        if composite >= 0.65:
            return "BUY"
        elif composite <= 0.35:
            return "SELL"
        return "HOLD"

    def _derive_prices(self, ticker: str, state: "WarrenBrainState", action: str) -> dict:
        buffett = state.get("buffett_brain", {})

        buy_price = None
        sell_price = None

        if isinstance(buffett, dict):
            buy_price = buffett.get("buy_price")
            sell_price = buffett.get("sell_price")

        # Fallback: use fundamentals intrinsic value estimate
        if buy_price is None or sell_price is None:
            fund = state.get("fundamentals", {})
            iv = fund.get("intrinsic_value_estimate") if isinstance(fund, dict) else None
            metrics = fund.get("metrics", {}) if isinstance(fund, dict) else {}
            current = metrics.get("current_price")

            if iv and current:
                margin = 0.20  # Buffett's margin of safety
                buy_price = buy_price or round(iv * (1 - margin), 2)
                sell_price = sell_price or round(iv * 1.15, 2)
            elif current:
                buy_price = buy_price or round(current * 0.90, 2)
                sell_price = sell_price or round(current * 1.20, 2)

        return {"buy": buy_price, "sell": sell_price}

    def _derive_confidence(self, composite: float, state: "WarrenBrainState") -> str:
        buffett = state.get("buffett_brain", {})
        llm_conviction = buffett.get("conviction", "") if isinstance(buffett, dict) else ""

        if llm_conviction in ("HIGH", "MEDIUM", "LOW"):
            return llm_conviction

        if composite >= 0.70 or composite <= 0.30:
            return "HIGH"
        elif composite >= 0.60 or composite <= 0.40:
            return "MEDIUM"
        return "LOW"

    def _build_rationale(
        self, ticker: str, state: "WarrenBrainState", action: str, composite: float
    ) -> str:
        buffett = state.get("buffett_brain", {})
        llm_reasoning = buffett.get("reasoning", "") if isinstance(buffett, dict) else ""

        if llm_reasoning:
            return llm_reasoning

        # Fallback: assemble from agent summaries
        parts = [f"Composite signal for {ticker}: {composite:.2f}/1.00 → {action}."]
        for key in ("fundamentals", "technicals", "sentiment"):
            agent_state = state.get(key, {})
            if isinstance(agent_state, dict) and (s := agent_state.get("summary")):
                parts.append(s)

        return " ".join(parts)

    def _backtest_stub(self, state: "WarrenBrainState") -> dict:
        """Populated by BacktestEngine after workflow completes."""
        return {"status": "pending", "note": "Run warren backtest <ticker> to compute metrics."}

    def _collect_agent_scores(self, state: "WarrenBrainState") -> dict:
        keys = ["fundamentals", "technicals", "sentiment", "thirteen_f", "ontology", "risk"]
        return {
            k: state.get(k, {}).get("score") if isinstance(state.get(k), dict) else None
            for k in keys
        }
