"""Tests for the technical indicators module."""

import numpy as np
import pandas as pd
import pytest
from backend import indicators


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """Synthetic 100-day OHLCV DataFrame."""
    np.random.seed(42)
    n = 100
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1
    high = np.maximum(close, open_) + np.abs(np.random.randn(n)) * 0.1
    low = np.minimum(close, open_) - np.abs(np.random.randn(n)) * 0.1
    volume = np.random.randint(100_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# ── SMA / EMA / WMA ──────────────────────────────────────────────────────────

def test_sma_length(ohlcv):
    result = indicators.sma(ohlcv["close"], 5)
    assert len(result) == len(ohlcv)


def test_sma_simple_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = indicators.sma(s, 3)
    assert round(result.iloc[-1], 6) == pytest.approx(4.0)


def test_ema_length(ohlcv):
    result = indicators.ema(ohlcv["close"], 12)
    assert len(result) == len(ohlcv)


def test_ema_converges_to_mean():
    """EMA of a constant series should equal the constant."""
    s = pd.Series([5.0] * 50)
    result = indicators.ema(s, 12)
    assert result.iloc[-1] == pytest.approx(5.0, abs=1e-6)


def test_wma_length(ohlcv):
    result = indicators.wma(ohlcv["close"], 5)
    assert len(result) == len(ohlcv)


# ── MACD ─────────────────────────────────────────────────────────────────────

def test_macd_columns(ohlcv):
    m = indicators.macd(ohlcv["close"])
    assert set(m.columns) == {"macd", "signal", "histogram"}
    assert len(m) == len(ohlcv)


def test_macd_histogram_equals_diff(ohlcv):
    m = indicators.macd(ohlcv["close"])
    diff = m["macd"] - m["signal"]
    pd.testing.assert_series_equal(
        m["histogram"].round(10), diff.round(10), check_names=False
    )


# ── RSI ──────────────────────────────────────────────────────────────────────

def test_rsi_range(ohlcv):
    r = indicators.rsi(ohlcv["close"], 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_length(ohlcv):
    r = indicators.rsi(ohlcv["close"], 14)
    assert len(r) == len(ohlcv)


# ── Bollinger Bands ──────────────────────────────────────────────────────────

def test_bollinger_bands_columns(ohlcv):
    bb = indicators.bollinger_bands(ohlcv["close"])
    assert set(bb.columns) == {"upper", "mid", "lower"}


def test_bollinger_bands_upper_geq_lower(ohlcv):
    bb = indicators.bollinger_bands(ohlcv["close"])
    valid = bb.dropna()
    assert (valid["upper"] >= valid["lower"]).all()


def test_bollinger_bands_mid_between(ohlcv):
    bb = indicators.bollinger_bands(ohlcv["close"])
    valid = bb.dropna()
    assert (valid["mid"] <= valid["upper"]).all()
    assert (valid["mid"] >= valid["lower"]).all()


# ── KDJ ──────────────────────────────────────────────────────────────────────

def test_kdj_columns(ohlcv):
    k = indicators.kdj(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    assert set(k.columns) == {"K", "D", "J"}
    assert len(k) == len(ohlcv)


# ── ATR ──────────────────────────────────────────────────────────────────────

def test_atr_non_negative(ohlcv):
    a = indicators.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    assert (a.dropna() >= 0).all()


# ── OBV ──────────────────────────────────────────────────────────────────────

def test_obv_length(ohlcv):
    o = indicators.obv(ohlcv["close"], ohlcv["volume"])
    assert len(o) == len(ohlcv)


# ── VWAP ─────────────────────────────────────────────────────────────────────

def test_vwap_length(ohlcv):
    v = indicators.vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    assert len(v) == len(ohlcv)


# ── CCI ──────────────────────────────────────────────────────────────────────

def test_cci_length(ohlcv):
    c = indicators.cci(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    assert len(c) == len(ohlcv)


# ── compute_all ──────────────────────────────────────────────────────────────

def test_compute_all_keys(ohlcv):
    result = indicators.compute_all(ohlcv)
    expected_keys = {
        "sma5", "sma10", "sma20", "sma60", "sma120", "sma250",
        "ema12", "ema26",
        "macd", "macd_signal", "macd_hist",
        "rsi14",
        "bb_upper", "bb_mid", "bb_lower",
        "kdj_k", "kdj_d", "kdj_j",
        "atr14",
        "obv",
        "vwap",
        "cci20",
        "vol_ma5", "vol_ma20",
    }
    assert expected_keys.issubset(result.keys())


def test_compute_all_lengths(ohlcv):
    result = indicators.compute_all(ohlcv)
    n = len(ohlcv)
    for key, vals in result.items():
        assert len(vals) == n, f"{key} has wrong length"


def test_compute_all_no_exception_small_df():
    """compute_all should not raise on a 5-row DataFrame."""
    df = pd.DataFrame(
        {
            "open": [10.0] * 5,
            "high": [11.0] * 5,
            "low": [9.0] * 5,
            "close": [10.5] * 5,
            "volume": [1_000_000.0] * 5,
        }
    )
    result = indicators.compute_all(df)
    assert isinstance(result, dict)
