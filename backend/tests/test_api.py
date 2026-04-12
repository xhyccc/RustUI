"""Integration tests for the FastAPI backend."""

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture()
def client():
    return TestClient(app)


# ── /health ──────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "cache" in data


# ── /api/stocks ──────────────────────────────────────────────────────────────

def test_list_stocks_success(client):
    mock_df = pd.DataFrame(
        {"code": ["000001", "600519"], "name": ["平安银行", "贵州茅台"], "exchange": ["SZ", "SH"]}
    )
    with patch("backend.main.data_sources.get_stock_list", return_value=mock_df):
        resp = client.get("/api/stocks")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2
    assert data[0]["code"] == "000001"


# ── /api/quote/{code} ────────────────────────────────────────────────────────

def test_get_quote_success(client):
    mock_quote = {
        "code": "000001",
        "name": "平安银行",
        "close": 12.50,
        "open": 12.40,
        "high": 12.60,
        "low": 12.30,
        "volume": 50_000_000,
        "amount": 6.25e8,
        "prev_close": 12.35,
        "bid": 12.49,
        "ask": 12.51,
        "date": "2024-01-15",
        "time": "15:00:00",
    }
    with patch("backend.main.data_sources.get_realtime_quote", return_value=mock_quote):
        resp = client.get("/api/quote/000001")
    assert resp.status_code == 200
    assert resp.json()["close"] == 12.50


def test_get_quote_not_found(client):
    with patch("backend.main.data_sources.get_realtime_quote", return_value={}):
        resp = client.get("/api/quote/999999")
    assert resp.status_code == 404


# ── /api/history/{code} ──────────────────────────────────────────────────────

def _make_ohlcv_df(n: int = 30) -> pd.DataFrame:
    np.random.seed(1)
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.random.randint(1e6, 1e7, n).astype(float),
        }
    )


def test_get_history_success(client):
    mock_df = _make_ohlcv_df()
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.get("/api/history/000001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == "000001"
    assert len(data["data"]) == 30


def test_get_history_not_found(client):
    with patch(
        "backend.main.data_sources.get_daily_history", return_value=pd.DataFrame()
    ):
        resp = client.get("/api/history/999999")
    assert resp.status_code == 404


# ── /api/indicators/{code} ───────────────────────────────────────────────────

def test_get_indicators_success(client):
    mock_df = _make_ohlcv_df(60)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.get("/api/indicators/000001")
    assert resp.status_code == 200
    data = resp.json()
    assert "indicators" in data
    assert "sma5" in data["indicators"]
    assert "macd" in data["indicators"]
    assert "rsi14" in data["indicators"]


# ── /api/alpha ────────────────────────────────────────────────────────────────

def test_alpha_success(client):
    mock_df = _make_ohlcv_df(60)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.post(
            "/api/alpha",
            json={
                "code": "000001",
                "expression": "rank(delta(close, 5)) / rank(volume)",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["expression"] == "rank(delta(close, 5)) / rank(volume)"
    assert len(data["values"]) == 60


def test_alpha_invalid_expression(client):
    mock_df = _make_ohlcv_df(60)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.post(
            "/api/alpha",
            json={"code": "000001", "expression": "close +* 1"},
        )
    assert resp.status_code == 400


def test_alpha_security_blocked(client):
    mock_df = _make_ohlcv_df(60)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.post(
            "/api/alpha",
            json={"code": "000001", "expression": "__import__('os').listdir('/')"},
        )
    assert resp.status_code == 400


# ── /api/cache ────────────────────────────────────────────────────────────────

def test_clear_cache(client):
    with patch("backend.main.cache.clear_all") as m:
        resp = client.delete("/api/cache")
    assert resp.status_code == 200
    m.assert_called_once()


def test_clear_expired_cache(client):
    with patch("backend.main.cache.clear_expired", return_value=5) as m:
        resp = client.delete("/api/cache/expired")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 5


# ── /api/alpha/backtest ───────────────────────────────────────────────────────

def test_backtest_alpha_success(client):
    mock_df = _make_ohlcv_df(120)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.post(
            "/api/alpha/backtest",
            json={
                "code": "000001",
                "expression": "delta(close, 5)",
                "start": "2023-01-01",
                "end": "2023-12-31",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == "000001"
    assert data["expression"] == "delta(close, 5)"
    assert "alpha" in data
    assert "beta" in data
    assert "sharpe_ratio" in data
    assert "annualized_return" in data
    assert "annualized_volatility" in data
    assert isinstance(data["strategy_returns"], list)
    assert isinstance(data["cumulative_returns"], list)
    assert isinstance(data["dates"], list)
    assert len(data["strategy_returns"]) == len(data["dates"])


def test_backtest_alpha_stats_are_numeric(client):
    mock_df = _make_ohlcv_df(120)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.post(
            "/api/alpha/backtest",
            json={"code": "000001", "expression": "rank(delta(close, 5))"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["alpha"], float)
    assert isinstance(data["beta"], float)
    assert isinstance(data["sharpe_ratio"], float)


def test_backtest_alpha_with_risk_free(client):
    mock_df = _make_ohlcv_df(120)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.post(
            "/api/alpha/backtest",
            json={
                "code": "000001",
                "expression": "delta(close, 1)",
                "annual_risk_free": 0.03,
            },
        )
    assert resp.status_code == 200
    assert resp.json()["sharpe_ratio"] is not None


def test_backtest_alpha_invalid_expression(client):
    mock_df = _make_ohlcv_df(60)
    with patch("backend.main.data_sources.get_daily_history", return_value=mock_df):
        resp = client.post(
            "/api/alpha/backtest",
            json={"code": "000001", "expression": "close +* 1"},
        )
    assert resp.status_code == 400


def test_backtest_alpha_not_found(client):
    with patch(
        "backend.main.data_sources.get_daily_history", return_value=pd.DataFrame()
    ):
        resp = client.post(
            "/api/alpha/backtest",
            json={"code": "999999", "expression": "close"},
        )
    assert resp.status_code == 404
