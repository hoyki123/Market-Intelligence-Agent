"""SQLAlchemy ORM models — identical schema on SQLite and PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from warren_brain.data.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisResult(Base):
    """
    One row per ticker per analysis run.
    Stores the final recommendation and the full raw pipeline output.
    """

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    # Recommendation fields (denormalized for fast queries / dashboard filters)
    action: Mapped[str | None] = mapped_column(String(10))          # BUY | SELL | HOLD
    buy_price: Mapped[float | None] = mapped_column(Float)
    sell_price: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str | None] = mapped_column(String(10))      # HIGH | MEDIUM | LOW
    composite_score: Mapped[float | None] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    moat_assessment: Mapped[str | None] = mapped_column(Text)

    # Full pipeline output — everything from all agents
    raw_output: Mapped[dict | None] = mapped_column(JSON)

    # Relationships
    agent_signals: Mapped[list[AgentSignal]] = relationship(
        "AgentSignal", back_populates="analysis", cascade="all, delete-orphan"
    )
    backtest: Mapped[BacktestResult | None] = relationship(
        "BacktestResult", back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:
        return f"<AnalysisResult {self.ticker} {self.action} @ {self.analyzed_at:%Y-%m-%d}>"


class AgentSignal(Base):
    """
    Individual agent score + summary per analysis run.
    Lets you query score history per agent over time.
    """

    __tablename__ = "agent_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    summary: Mapped[str | None] = mapped_column(Text)
    raw_output: Mapped[dict | None] = mapped_column(JSON)

    analysis: Mapped[AnalysisResult] = relationship("AnalysisResult", back_populates="agent_signals")

    def __repr__(self) -> str:
        return f"<AgentSignal {self.agent_name} score={self.score}>"


class BacktestResult(Base):
    """
    Backtest metrics linked to an analysis run.
    Also supports standalone backtest runs (analysis_id nullable).
    """

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("analysis_results.id", ondelete="SET NULL"), nullable=True, index=True
    )
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    backtested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    # Strategy parameters
    buy_price_target: Mapped[float | None] = mapped_column(Float)
    sell_price_target: Mapped[float | None] = mapped_column(Float)
    years: Mapped[int | None] = mapped_column(Integer)

    # Metrics (stored as strings to preserve formatting, e.g. "18.4%")
    cagr: Mapped[str | None] = mapped_column(String(20))
    total_return: Mapped[str | None] = mapped_column(String(20))
    sharpe: Mapped[float | None] = mapped_column(Float)
    sortino: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[str | None] = mapped_column(String(20))
    alpha_vs_spy: Mapped[str | None] = mapped_column(String(20))
    spy_cagr: Mapped[str | None] = mapped_column(String(20))
    annualized_volatility: Mapped[str | None] = mapped_column(String(20))

    # Full backtest output including trade log
    raw_output: Mapped[dict | None] = mapped_column(JSON)

    analysis: Mapped[AnalysisResult | None] = relationship(
        "AnalysisResult", back_populates="backtest"
    )

    def __repr__(self) -> str:
        return f"<BacktestResult {self.ticker} CAGR={self.cagr}>"
