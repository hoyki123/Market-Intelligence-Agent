"""SQLite-backed cache to avoid redundant API calls during a session."""

import json
import sqlite3
import time
from pathlib import Path


class DataCache:
    def __init__(self, db_path: str = "warren_brain.db", ttl_seconds: int = 3600):
        self._db_path = db_path
        self._ttl = ttl_seconds
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )"""
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def get(self, key: str) -> dict | list | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row and row[1] > time.time():
            return json.loads(row[0])
        return None

    def set(self, key: str, value: dict | list, ttl: int | None = None) -> None:
        expires_at = time.time() + (ttl if ttl is not None else self._ttl)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, default=str), expires_at),
            )

    def invalidate(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear_expired(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))


# Module-level singleton
_cache: DataCache | None = None


def get_cache() -> DataCache:
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache
