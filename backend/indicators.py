"""
Technical indicators computed from OHLCV pandas DataFrames.

All functions accept a DataFrame with columns [open, high, low, close, volume]
and return a DataFrame or Series with the computed indicator values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import series_to_list as _s


# ──────────────────────────────────────────────────────────────────────────────
# Moving averages
# ──────────────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted moving average."""
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        w = weights[-len(x):]
        return float(np.dot(x, w) / w.sum())

    return series.rolling(window=period, min_periods=1).apply(_wma, raw=True)


# ──────────────────────────────────────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────────────────────────────────────

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD indicator.

    Returns DataFrame with columns: macd, signal, histogram.
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": hist}
    )


# ──────────────────────────────────────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


# ──────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ──────────────────────────────────────────────────────────────────────────────

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: upper, mid, lower."""
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=1).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower})


# ──────────────────────────────────────────────────────────────────────────────
# KDJ (Stochastic)
# ──────────────────────────────────────────────────────────────────────────────

def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> pd.DataFrame:
    """KDJ indicator (Chinese stochastic oscillator)."""
    low_n = low.rolling(window=n, min_periods=1).min()
    high_n = high.rolling(window=n, min_periods=1).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = (close - low_n) / denom * 100

    k = rsv.ewm(com=m1 - 1, adjust=False, min_periods=1).mean()
    d = k.ewm(com=m2 - 1, adjust=False, min_periods=1).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"K": k, "D": d, "J": j})


# ──────────────────────────────────────────────────────────────────────────────
# ATR / Volatility
# ──────────────────────────────────────────────────────────────────────────────

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False, min_periods=1).mean().rename("atr")


# ──────────────────────────────────────────────────────────────────────────────
# OBV (On-Balance Volume)
# ──────────────────────────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum().rename("obv")


# ──────────────────────────────────────────────────────────────────────────────
# VWAP (intraday approximation over a window)
# ──────────────────────────────────────────────────────────────────────────────

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Volume-Weighted Average Price (rolling approximation)."""
    typical = (high + low + close) / 3
    cum_vol = volume.rolling(window=period, min_periods=1).sum()
    cum_tp_vol = (typical * volume).rolling(window=period, min_periods=1).sum()
    return (cum_tp_vol / cum_vol.replace(0, np.nan)).rename("vwap")


# ──────────────────────────────────────────────────────────────────────────────
# CCI (Commodity Channel Index)
# ──────────────────────────────────────────────────────────────────────────────

def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    mean_tp = tp.rolling(window=period, min_periods=1).mean()
    mean_dev = tp.rolling(window=period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return ((tp - mean_tp) / (0.015 * mean_dev.replace(0, np.nan))).rename("cci")


# ──────────────────────────────────────────────────────────────────────────────
# Volume moving average
# ──────────────────────────────────────────────────────────────────────────────

def volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume simple moving average."""
    return sma(volume, period).rename(f"vol_ma{period}")


# ──────────────────────────────────────────────────────────────────────────────
# Compute all standard indicators for a given OHLCV DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> dict:
    """
    Given an OHLCV DataFrame, compute all standard indicators.

    Returns a dict mapping indicator name → list of float values
    (NaN → None for JSON serialization).
    """
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    result: dict = {}

    # SMAs
    for p in [5, 10, 20, 60, 120, 250]:
        result[f"sma{p}"] = _s(sma(c, p))

    # EMAs
    for p in [12, 26]:
        result[f"ema{p}"] = _s(ema(c, p))

    # MACD
    m = macd(c)
    result["macd"] = _s(m["macd"])
    result["macd_signal"] = _s(m["signal"])
    result["macd_hist"] = _s(m["histogram"])

    # RSI
    result["rsi14"] = _s(rsi(c, 14))

    # Bollinger
    bb = bollinger_bands(c)
    result["bb_upper"] = _s(bb["upper"])
    result["bb_mid"] = _s(bb["mid"])
    result["bb_lower"] = _s(bb["lower"])

    # KDJ
    k = kdj(h, lo, c)
    result["kdj_k"] = _s(k["K"])
    result["kdj_d"] = _s(k["D"])
    result["kdj_j"] = _s(k["J"])

    # ATR
    result["atr14"] = _s(atr(h, lo, c, 14))

    # OBV
    result["obv"] = _s(obv(c, v))

    # VWAP
    result["vwap"] = _s(vwap(h, lo, c, v))

    # CCI
    result["cci20"] = _s(cci(h, lo, c, 20))

    # Volume MA
    result["vol_ma5"] = _s(volume_ma(v, 5))
    result["vol_ma20"] = _s(volume_ma(v, 20))

    return result
