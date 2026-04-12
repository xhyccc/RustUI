"""Tests for the SQLite cache module."""

import time
import pytest
from backend import cache


@pytest.fixture(autouse=True)
def clean_cache(tmp_path, monkeypatch):
    """Redirect cache DB to a temp directory for each test."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DB", tmp_path / "test.db")
    # Reset thread-local connection so a fresh DB is opened
    if hasattr(cache._local, "conn"):
        cache._local.conn.close()
        del cache._local.conn
    yield
    if hasattr(cache._local, "conn"):
        cache._local.conn.close()
        del cache._local.conn


def test_set_and_get():
    cache.set("key1", {"a": 1}, ttl=60)
    result = cache.get("key1")
    assert result == {"a": 1}


def test_get_missing_returns_none():
    assert cache.get("nonexistent") is None


def test_get_expired_returns_none():
    cache.set("expiring", [1, 2, 3], ttl=0.01)
    time.sleep(0.05)
    assert cache.get("expiring") is None


def test_set_overwrites():
    cache.set("k", "first", ttl=60)
    cache.set("k", "second", ttl=60)
    assert cache.get("k") == "second"


def test_delete():
    cache.set("to_delete", 42, ttl=60)
    cache.delete("to_delete")
    assert cache.get("to_delete") is None


def test_clear_all():
    cache.set("a", 1, ttl=60)
    cache.set("b", 2, ttl=60)
    cache.clear_all()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_clear_expired():
    cache.set("fresh", "ok", ttl=3600)
    cache.set("stale", "bye", ttl=0.01)
    time.sleep(0.05)
    deleted = cache.clear_expired()
    assert deleted == 1
    assert cache.get("fresh") == "ok"


def test_stats():
    cache.set("x", 1, ttl=3600)
    cache.set("y", 2, ttl=0.01)
    time.sleep(0.05)
    s = cache.stats()
    assert s["total"] == 2
    assert s["expired"] == 1
    assert s["db_size_bytes"] > 0


def test_stores_various_types():
    cases = [
        ("int", 42),
        ("float", 3.14),
        ("str", "hello"),
        ("list", [1, 2, 3]),
        ("dict", {"nested": {"val": True}}),
        ("none_val", None),
    ]
    for key, val in cases:
        cache.set(key, val, ttl=60)
        assert cache.get(key) == val
