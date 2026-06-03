"""Tests for the data layer — these hit real APIs (yfinance), so run with internet access."""

import pytest
from warren_brain.data.market_data import fetch_key_metrics, fetch_price_history
from warren_brain.data.cache import DataCache


class TestDataCache:
    def test_set_and_get(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        cache.set("key1", {"value": 42})
        assert cache.get("key1") == {"value": 42}

    def test_expired_returns_none(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "test.db"), ttl_seconds=-1)
        cache.set("key2", {"value": 99})
        assert cache.get("key2") is None

    def test_invalidate(self, tmp_path):
        cache = DataCache(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        cache.set("key3", {"value": 1})
        cache.invalidate("key3")
        assert cache.get("key3") is None


@pytest.mark.integration
class TestMarketData:
    """Integration tests — require internet access."""

    def test_fetch_key_metrics_returns_dict(self):
        metrics = fetch_key_metrics("AAPL")
        assert isinstance(metrics, dict)
        assert metrics["ticker"] == "AAPL"
        assert "current_price" in metrics

    def test_fetch_price_history_shape(self):
        df = fetch_price_history("AAPL", period_years=1)
        assert not df.empty
        assert "Close" in df.columns
        assert len(df) > 200  # roughly 252 trading days per year
