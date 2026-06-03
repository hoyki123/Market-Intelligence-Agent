"""TechnicalsAgent — momentum, trend, and entry/exit signals via pandas-ta."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from warren_brain.agents.base import BaseAgent
from warren_brain.data.market_data import fetch_price_history

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState


class TechnicalsAgent(BaseAgent):
    name = "TechnicalsAgent"

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            df = fetch_price_history(ticker, period_years=2)
            if df.empty:
                return {"error": "No price history", "score": 0.5}

            # Reset index if it's a DatetimeIndex or string
            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()

            indicators = self._compute_indicators(df)
            score = self._compute_score(indicators)
            signal = self._determine_signal(indicators)

            return {
                "indicators": indicators,
                "signal": signal,
                "score": score,
                "summary": self._summarize(ticker, indicators, signal, score),
            }
        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        try:
            import pandas_ta as ta  # type: ignore

            df.ta.strategy("common")  # computes RSI, MACD, BBands, SMA, EMA
        except ImportError:
            pass

        # Manual fallbacks if pandas-ta not available
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # VWAP (20-day rolling approximation)
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()

        # ATR (14) for volatility context
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        def last(s: pd.Series) -> float | None:
            val = s.dropna().iloc[-1] if not s.dropna().empty else None
            return round(float(val), 4) if val is not None else None

        current_price = last(close)
        return {
            "current_price": current_price,
            "sma20": last(sma20),
            "sma50": last(sma50),
            "sma200": last(sma200),
            "rsi14": last(rsi),
            "macd": last(macd),
            "macd_signal": last(macd_signal),
            "macd_histogram": last(macd_hist),
            "vwap_20d": last(vwap),
            "atr14": last(atr),
            "above_sma20": current_price > last(sma20) if current_price and last(sma20) else None,
            "above_sma50": current_price > last(sma50) if current_price and last(sma50) else None,
            "above_sma200": current_price > last(sma200) if current_price and last(sma200) else None,
            "52w_high": round(float(close.tail(252).max()), 2),
            "52w_low": round(float(close.tail(252).min()), 2),
            "pct_from_52w_high": round((current_price / close.tail(252).max() - 1) * 100, 2) if current_price else None,
        }

    def _compute_score(self, ind: dict) -> float:
        signals = []

        # RSI: oversold (< 40) is bullish entry, overbought (> 70) is bearish
        rsi = ind.get("rsi14")
        if rsi is not None:
            if rsi < 30:
                signals.append(0.85)
            elif rsi < 40:
                signals.append(0.70)
            elif rsi < 55:
                signals.append(0.55)
            elif rsi < 65:
                signals.append(0.45)
            else:
                signals.append(0.25)

        # Trend: price above key MAs
        for key in ("above_sma20", "above_sma50", "above_sma200"):
            val = ind.get(key)
            if val is not None:
                signals.append(0.65 if val else 0.35)

        # MACD histogram: positive = bullish momentum
        hist = ind.get("macd_histogram")
        if hist is not None:
            signals.append(0.65 if hist > 0 else 0.35)

        # Distance from 52w high: buying near lows is better value
        pct = ind.get("pct_from_52w_high")
        if pct is not None:
            # -20% below high gives score ~0.65
            signals.append(self._score(abs(pct), 0, 40))

        return round(sum(signals) / len(signals), 3) if signals else 0.5

    def _determine_signal(self, ind: dict) -> str:
        rsi = ind.get("rsi14") or 50
        macd_hist = ind.get("macd_histogram") or 0
        above_50 = ind.get("above_sma50", True)
        above_200 = ind.get("above_sma200", True)

        bullish = sum([rsi < 50, macd_hist > 0, above_50, above_200])
        if bullish >= 3 and rsi < 45:
            return "STRONG_BUY"
        elif bullish >= 3:
            return "BUY"
        elif bullish <= 1 and rsi > 65:
            return "SELL"
        elif bullish <= 1:
            return "WEAK"
        return "NEUTRAL"

    def _summarize(self, ticker: str, ind: dict, signal: str, score: float) -> str:
        parts = [f"{ticker} technical signal: {signal}."]
        if ind.get("rsi14"):
            parts.append(f"RSI(14): {ind['rsi14']:.1f}.")
        if ind.get("macd_histogram"):
            direction = "bullish" if ind["macd_histogram"] > 0 else "bearish"
            parts.append(f"MACD histogram: {ind['macd_histogram']:.3f} ({direction}).")
        trend = []
        if ind.get("above_sma50"):
            trend.append("above SMA50")
        if ind.get("above_sma200"):
            trend.append("above SMA200")
        if trend:
            parts.append(f"Trend: {', '.join(trend)}.")
        if ind.get("pct_from_52w_high") is not None:
            parts.append(f"{ind['pct_from_52w_high']:.1f}% from 52w high.")
        parts.append(f"Technical score: {score:.2f}/1.00.")
        return " ".join(parts)
