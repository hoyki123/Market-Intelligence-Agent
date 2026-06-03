# Testing Conventions

## Running Tests
```bash
.venv/bin/pytest tests/ -v
```

## What to Test
- Agent `analyze()` methods: mock the data layer, assert output shape and score range [0, 1]
- `execute_tool()` dispatcher: assert each tool name routes to the correct agent
- `complete()` in `llm.py`: mock the SDK client, assert streaming concatenation works
- Cache: assert TTL expiry returns None, assert set/get roundtrip

## What NOT to Test
- Live API calls (yfinance, Massive, EDGAR) — these are integration concerns, not unit tests
- LLM output content — Claude's reasoning is non-deterministic
- Streamlit UI — test the data layer, not the rendering

## Mocking Pattern
```python
from unittest.mock import patch, MagicMock

def test_fundamentals_agent_score_range():
    with patch("warren_brain.data.market_data.fetch_key_metrics") as mock:
        mock.return_value = {"pe_ratio": 20, "roe": 0.18, "profit_margin": 0.25,
                             "revenue_growth": 0.10, "debt_to_equity": 50,
                             "free_cashflow": 1e9, "current_price": 100,
                             "forward_pe": 18, "earnings_growth": 0.10,
                             "name": "Test Corp", "ticker": "TEST", "sector": "Tech"}
        result = FundamentalsAgent().analyze("TEST", {})
        assert 0.0 <= result["score"] <= 1.0
        assert "intrinsic_value_estimate" in result
```

## Agent Output Contract
Every agent must return a dict with at minimum:
- `score: float` in [0.0, 1.0]
- `summary: str`
- On error: `{"error": str, "score": 0.5}`
