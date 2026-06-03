"""Backtesting engine — simulates a buy-hold-sell strategy using historical price data.

Strategy: Buy when price <= buy_price, sell when price >= sell_price.
Benchmarks against SPY (buy-and-hold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warren_brain.data.market_data import fetch_price_history


def run_backtest(
    ticker: str,
    buy_price: float,
    sell_price: float,
    years: int = 5,
) -> dict:
    """
    Simulate the Warren Brain recommended entry/exit strategy.

    Returns:
        Dict with CAGR, Sharpe, Sortino, MaxDrawdown, Alpha vs SPY, trade log.
    """
    df = fetch_price_history(ticker, period_years=years)
    spy = fetch_price_history("SPY", period_years=years)

    if df.empty:
        return {"error": f"No price history for {ticker}"}

    close = df["Close"].astype(float).reset_index(drop=True)
    spy_close = spy["Close"].astype(float).reset_index(drop=True)

    portfolio_values, trades = _simulate_strategy(close, buy_price, sell_price)
    spy_values = spy_close / spy_close.iloc[0]

    metrics = _compute_metrics(portfolio_values, spy_values, ticker, years)
    metrics["trades"] = trades
    metrics["buy_price_target"] = buy_price
    metrics["sell_price_target"] = sell_price
    return metrics


def _simulate_strategy(
    close: pd.Series, buy_price: float, sell_price: float
) -> tuple[pd.Series, list[dict]]:
    """
    Simple threshold strategy:
    - Buy 100% when price dips to or below buy_price (first opportunity)
    - Sell 100% when price reaches or exceeds sell_price
    - Repeat until end of series
    """
    cash = 1.0
    shares = 0.0
    portfolio = []
    trades = []
    in_position = False

    for i, price in enumerate(close):
        if not in_position and price <= buy_price and cash > 0:
            shares = cash / price
            cash = 0.0
            in_position = True
            trades.append({"day": i, "action": "BUY", "price": round(price, 2)})
        elif in_position and price >= sell_price:
            cash = shares * price
            shares = 0.0
            in_position = False
            trades.append({"day": i, "action": "SELL", "price": round(price, 2)})

        portfolio.append(cash + shares * price)

    return pd.Series(portfolio) / portfolio[0], trades


def _compute_metrics(
    portfolio: pd.Series,
    benchmark: pd.Series,
    ticker: str,
    years: int,
) -> dict:
    min_len = min(len(portfolio), len(benchmark))
    portfolio = portfolio.iloc[:min_len]
    benchmark = benchmark.iloc[:min_len]

    port_returns = portfolio.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()

    rf_daily = 0.045 / 252
    excess = port_returns - rf_daily

    # CAGR
    total_return = portfolio.iloc[-1] / portfolio.iloc[0]
    cagr = total_return ** (1 / years) - 1

    # Sharpe
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # Sortino
    downside = excess[excess < 0]
    sortino = (excess.mean() / (downside.std() * np.sqrt(252))) if (not downside.empty and downside.std() > 0) else 0

    # Max drawdown
    cummax = portfolio.cummax()
    drawdown = (portfolio - cummax) / cummax
    max_drawdown = float(drawdown.min())

    # Alpha vs benchmark
    bench_cagr = benchmark.iloc[-1] ** (1 / years) - 1
    alpha = cagr - bench_cagr

    # Win rate (fraction of months portfolio beat benchmark)
    port_monthly = portfolio.resample("ME", on=pd.date_range(start="2020", periods=len(portfolio), freq="D")).last().pct_change().dropna()
    bench_monthly = benchmark.resample("ME", on=pd.date_range(start="2020", periods=len(benchmark), freq="D")).last().pct_change().dropna()

    return {
        "ticker": ticker,
        "years": years,
        "CAGR": f"{cagr:.1%}",
        "total_return": f"{(total_return - 1):.1%}",
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDrawdown": f"{max_drawdown:.1%}",
        "Alpha_vs_SPY": f"{alpha:.1%}",
        "SPY_CAGR": f"{bench_cagr:.1%}",
        "annualized_volatility": f"{port_returns.std() * np.sqrt(252):.1%}",
    }
