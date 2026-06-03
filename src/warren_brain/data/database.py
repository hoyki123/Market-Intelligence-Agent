"""Database engine and session factory.

Supports SQLite (default, zero-setup) and PostgreSQL/Supabase (production).

SQLite  → set DATABASE_URL=sqlite:///warren_brain.db  (default)
Postgres → set DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT in .env
           The password is URL-encoded automatically via quote_plus, so
           special characters (@ # % etc.) in the password work as-is.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import StaticPool

from warren_brain.config import settings


class Base(DeclarativeBase):
    pass


def _build_url() -> str:
    """Build the DB connection URL, encoding the password when using Postgres."""
    if settings.db_host:
        password = quote_plus(settings.db_password)
        return (
            f"postgresql+psycopg2://{settings.db_user}:{password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
            f"?sslmode=require"
        )
    return settings.database_url


def _build_engine():
    url = _build_url()

    if url.startswith("sqlite"):
        # SQLite needs special pool settings for multi-thread use (e.g. Streamlit)
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        # Enforce foreign keys — SQLite skips them by default
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(conn, _record):
            conn.execute("PRAGMA foreign_keys=ON")

    else:
        # PostgreSQL / Supabase — use connection pooling with SSL support.
        # sslmode=require is handled via the URL query param (?sslmode=require)
        # so no extra connect_args are needed here.
        engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   # recycle stale connections after network drops
            echo=False,
        )

    return engine


engine = _build_engine()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session() -> Session:
    """Return a new DB session. Caller is responsible for closing it."""
    return SessionLocal()


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    from warren_brain.data import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)
