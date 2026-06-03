"""Unit tests for agents — mock external dependencies."""

from unittest.mock import patch, MagicMock
import pytest

from warren_brain.agents.fundamentals import FundamentalsAgent
from warren_brain.agents.technicals import TechnicalsAgent
from warren_brain.agents.risk import RiskAgent
from warren_brain.graph.state import WarrenBrainState


MOCK_METRICS = {
    "ticker": "NVDA",
    "name": "NVIDIA Corporation",
    "sector": "Technology",
    "industry": "Semiconductors",
    "market_cap": 2_000_000_000_000,
    "pe_ratio": 35.0,
    "forward_pe": 28.0,
    "roe": 0.38,
    "profit_margin": 0.25,
    "revenue_growth": 0.20,
    "debt_to_equity": 45.0,
    "free_cashflow": 15_000_000_000,
    "current_price": 850.0,
    "earnings_growth": 0.15,
}


def _empty_state(ticker="NVDA") -> WarrenBrainState:
    return {"ticker": ticker, "errors": [], "messages": []}


class TestFundamentalsAgent:
    @patch("warren_brain.agents.fundamentals.fetch_key_metrics", return_value=MOCK_METRICS)
    def test_returns_score_and_summary(self, _mock):
        agent = FundamentalsAgent()
        result = agent.analyze("NVDA", _empty_state())

        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert "summary" in result
        assert "intrinsic_value_estimate" in result

    @patch("warren_brain.agents.fundamentals.fetch_key_metrics", side_effect=Exception("API down"))
    def test_graceful_error(self, _mock):
        agent = FundamentalsAgent()
        result = agent.analyze("NVDA", _empty_state())
        assert "error" in result
        assert result.get("score") == 0.5


class TestTechnicalsAgent:
    @patch("warren_brain.agents.technicals.fetch_price_history")
    def test_returns_indicators(self, mock_fetch):
        import pandas as pd
        import numpy as np

        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        mock_fetch.return_value = pd.DataFrame({
            "Open": np.random.uniform(800, 900, n),
            "High": np.random.uniform(900, 950, n),
            "Low": np.random.uniform(750, 800, n),
            "Close": np.random.uniform(820, 880, n),
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        }, index=dates)

        agent = TechnicalsAgent()
        result = agent.analyze("NVDA", _empty_state())

        assert "score" in result
        assert "indicators" in result
        assert "signal" in result

    @patch("warren_brain.agents.technicals.fetch_price_history")
    def test_empty_history_returns_error(self, mock_fetch):
        import pandas as pd
        mock_fetch.return_value = pd.DataFrame()
        agent = TechnicalsAgent()
        result = agent.analyze("NVDA", _empty_state())
        assert "error" in result


class TestRiskAgent:
    @patch("warren_brain.agents.risk.fetch_price_history")
    def test_computes_metrics(self, mock_fetch):
        import pandas as pd
        import numpy as np

        n = 756
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        price = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.015, n)) * 500)

        def side_effect(ticker, period_years=None):
            return pd.DataFrame({"Close": price}, index=dates)

        mock_fetch.side_effect = side_effect

        agent = RiskAgent()
        result = agent.analyze("NVDA", _empty_state())

        assert "metrics" in result
        m = result["metrics"]
        assert "annualized_volatility" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
