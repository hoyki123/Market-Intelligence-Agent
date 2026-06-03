"""Repository — save and query analysis results from the database.

All public functions handle their own session lifecycle.
Call init_db() once at startup before using any of these.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, desc

from warren_brain.data.database import get_session
from warren_brain.data.models import AnalysisResult, AgentSignal, BacktestResult


# ── Write ─────────────────────────────────────────────────────────────────────

def save_analysis(pipeline_output: dict) -> int:
    """
    Persist a full pipeline output (from workflow.run_analysis) to the DB.
    Returns the new AnalysisResult.id.
    """
    ticker = pipeline_output.get("ticker", "UNKNOWN")
    rec = pipeline_output.get("recommendation", {})
    buffett = pipeline_output.get("buffett_brain", {})

    with get_session() as session:
        analysis = AnalysisResult(
            ticker=ticker,
            action=rec.get("action"),
            buy_price=rec.get("buy_price"),
            sell_price=rec.get("sell_price"),
            confidence=rec.get("confidence"),
            composite_score=rec.get("composite_score"),
            rationale=rec.get("rationale"),
            moat_assessment=buffett.get("moat_assessment") if isinstance(buffett, dict) else None,
            raw_output=pipeline_output,
        )
        session.add(analysis)
        session.flush()  # get analysis.id before adding children

        # Save per-agent scores as child rows
        agent_keys = ["fundamentals", "technicals", "sentiment", "thirteen_f", "ontology", "risk"]
        agent_name_map = {
            "fundamentals": "FundamentalsAgent",
            "technicals": "TechnicalsAgent",
            "sentiment": "SentimentAgent",
            "thirteen_f": "13FAgent",
            "ontology": "OntologyAgent",
            "risk": "RiskAgent",
        }
        for key in agent_keys:
            agent_data = pipeline_output.get(key, {})
            if not isinstance(agent_data, dict):
                continue
            session.add(AgentSignal(
                analysis_id=analysis.id,
                agent_name=agent_name_map[key],
                score=agent_data.get("score"),
                summary=agent_data.get("summary"),
                raw_output=agent_data,
            ))

        session.commit()
        return analysis.id


def save_backtest(backtest_output: dict, analysis_id: int | None = None) -> int:
    """
    Persist a backtest result. Optionally link to an existing AnalysisResult.
    Returns the new BacktestResult.id.
    """
    with get_session() as session:
        record = BacktestResult(
            analysis_id=analysis_id,
            ticker=backtest_output.get("ticker", "UNKNOWN"),
            buy_price_target=backtest_output.get("buy_price_target"),
            sell_price_target=backtest_output.get("sell_price_target"),
            years=backtest_output.get("years"),
            cagr=backtest_output.get("CAGR"),
            total_return=backtest_output.get("total_return"),
            sharpe=backtest_output.get("Sharpe"),
            sortino=backtest_output.get("Sortino"),
            max_drawdown=backtest_output.get("MaxDrawdown"),
            alpha_vs_spy=backtest_output.get("Alpha_vs_SPY"),
            spy_cagr=backtest_output.get("SPY_CAGR"),
            annualized_volatility=backtest_output.get("annualized_volatility"),
            raw_output=backtest_output,
        )
        session.add(record)
        session.commit()
        return record.id


# ── Read ──────────────────────────────────────────────────────────────────────

def get_latest_analysis(ticker: str) -> dict | None:
    """Return the most recent analysis for a ticker, or None."""
    with get_session() as session:
        row = session.scalars(
            select(AnalysisResult)
            .where(AnalysisResult.ticker == ticker.upper())
            .order_by(desc(AnalysisResult.analyzed_at))
            .limit(1)
        ).first()
        return row.raw_output if row else None


def get_analysis_history(ticker: str, limit: int = 20) -> list[dict]:
    """Return the last N analyses for a ticker as lightweight dicts."""
    with get_session() as session:
        rows = session.scalars(
            select(AnalysisResult)
            .where(AnalysisResult.ticker == ticker.upper())
            .order_by(desc(AnalysisResult.analyzed_at))
            .limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "ticker": r.ticker,
                "analyzed_at": r.analyzed_at.isoformat(),
                "action": r.action,
                "buy_price": r.buy_price,
                "sell_price": r.sell_price,
                "confidence": r.confidence,
                "composite_score": r.composite_score,
            }
            for r in rows
        ]


def get_all_tickers() -> list[str]:
    """Return every ticker that has at least one stored analysis."""
    with get_session() as session:
        rows = session.execute(
            select(AnalysisResult.ticker).distinct().order_by(AnalysisResult.ticker)
        ).scalars().all()
        return list(rows)


def get_score_history(ticker: str, agent_name: str, limit: int = 50) -> list[dict]:
    """Return score history for a specific agent + ticker over time."""
    with get_session() as session:
        rows = session.execute(
            select(AgentSignal.score, AnalysisResult.analyzed_at)
            .join(AnalysisResult, AgentSignal.analysis_id == AnalysisResult.id)
            .where(
                AnalysisResult.ticker == ticker.upper(),
                AgentSignal.agent_name == agent_name,
            )
            .order_by(desc(AnalysisResult.analyzed_at))
            .limit(limit)
        ).all()
        return [{"score": r.score, "analyzed_at": r.analyzed_at.isoformat()} for r in rows]


def get_backtest_history(ticker: str, limit: int = 10) -> list[dict]:
    """Return backtest history for a ticker."""
    with get_session() as session:
        rows = session.scalars(
            select(BacktestResult)
            .where(BacktestResult.ticker == ticker.upper())
            .order_by(desc(BacktestResult.backtested_at))
            .limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "ticker": r.ticker,
                "backtested_at": r.backtested_at.isoformat(),
                "cagr": r.cagr,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "alpha_vs_spy": r.alpha_vs_spy,
            }
            for r in rows
        ]
