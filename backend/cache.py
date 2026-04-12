"""SQLite-based data cache for stock market data."""

import sqlite3
import json
import time
import os
import threading
from pathlib import Path
from typing import Optional

CACHE_DIR = Path.home() / ".rustui_cache"
CACHE_DB = CACHE_DIR / "market_data.db"

# TTL constants (seconds)
TTL_REALTIME = 30        # real-time quotes: 30 s
TTL_INTRADAY = 300       # intraday bars: 5 min
TTL_DAILY = 3600         # daily OHLCV: 1 hour
TTL_STATIC = 86400       # static info (stock list): 24 h

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection."""
    if not hasattr(_local, "conn"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CACHE_DB), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
        _init_schema(conn)
    return _local.conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL,
            ts      REAL NOT NULL,
            ttl     REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON cache(ts)")
    conn.commit()


def get(key: str) -> Optional[object]:
    """Return cached value or None if missing/expired."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT value, ts, ttl FROM cache WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    value, ts, ttl = row
    if time.time() - ts > ttl:
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        return None
    return json.loads(value)


def set(key: str, value: object, ttl: float = TTL_DAILY) -> None:
    """Store value in cache with given TTL."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO cache(key, value, ts, ttl) VALUES(?,?,?,?)",
        (key, json.dumps(value, default=str), time.time(), ttl),
    )
    conn.commit()


def delete(key: str) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
    conn.commit()


def clear_expired() -> int:
    """Remove all expired entries, return count deleted."""
    conn = _get_conn()
    now = time.time()
    cur = conn.execute(
        "DELETE FROM cache WHERE (? - ts) > ttl", (now,)
    )
    conn.commit()
    return cur.rowcount


def clear_all() -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM cache")
    conn.commit()


def stats() -> dict:
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    now = time.time()
    expired = conn.execute(
        "SELECT COUNT(*) FROM cache WHERE (? - ts) > ttl", (now,)
    ).fetchone()[0]
    db_size = os.path.getsize(str(CACHE_DB)) if CACHE_DB.exists() else 0
    return {"total": total, "expired": expired, "db_size_bytes": db_size}
