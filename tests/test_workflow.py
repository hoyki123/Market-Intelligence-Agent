"""Smoke test for the full LangGraph workflow (mocks all agents)."""

from unittest.mock import patch, MagicMock
import pytest


@pytest.mark.integration
def test_run_analysis_returns_recommendation():
    """Full pipeline smoke test — mocks LLM calls but hits yfinance."""
    from unittest.mock import patch

    mock_llm_response = MagicMock()
    mock_llm_response.choices[0].message.content = (
        '{"label": "BULLISH", "score": 0.6, "themes": ["AI demand"], "summary": "Positive outlook."}'
    )

    mock_buffett_response = MagicMock()
    mock_buffett_response.choices[0].message.content = (
        '{"action": "BUY", "buy_price": 800.0, "sell_price": 1000.0, '
        '"conviction": "HIGH", "moat_assessment": "STRONG — dominant GPU position", '
        '"reasoning": "NVDA has durable AI moat.", "concerns": ["High P/E", "Competition from AMD"]}'
    )

    with patch("openai.resources.chat.completions.Completions.create") as mock_create:
        mock_create.side_effect = [mock_llm_response, mock_buffett_response]

        from warren_brain.graph.workflow import run_analysis
        result = run_analysis("NVDA")

    assert "recommendation" in result
    assert "ticker" in result
    assert result["ticker"] == "NVDA"
    rec = result["recommendation"]
    assert "action" in rec
