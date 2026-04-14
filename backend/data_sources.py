"""
Data sources: yfinance, akshare, Sina Finance (新浪财经).

All public functions return normalized pandas DataFrames:
  - OHLCV columns:  open, high, low, close, volume
  - Index:          datetime (UTC-aware) for intraday, date for daily
"""

from __future__ import annotations

import re
import time
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import akshare as ak
import yfinance as yf

from . import cache

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_symbol(symbol: str) -> str:
    """Ensure A-share codes are 6-digit strings."""
    return symbol.strip().zfill(6)


def _cache_key(*parts) -> str:
    return ":".join(str(p) for p in parts)


# ──────────────────────────────────────────────────────────────────────────────
# Stock list
# ──────────────────────────────────────────────────────────────────────────────

def get_stock_list() -> pd.DataFrame:
    """Return all A-share stocks (code, name, exchange) via akshare."""
    key = _cache_key("stock_list")
    cached = cache.get(key)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "code", "name": "name"})
        df["exchange"] = df["code"].apply(
            lambda c: "SH" if c.startswith("6") else "SZ"
        )
        cache.set(key, df.to_dict(orient="records"), cache.TTL_STATIC)
        return df
    except Exception as exc:
        logger.warning("get_stock_list failed: %s", exc)
        return pd.DataFrame(columns=["code", "name", "exchange"])


# ──────────────────────────────────────────────────────────────────────────────
# Real-time quote  (Sina Finance)
# ──────────────────────────────────────────────────────────────────────────────

_SINA_RE = re.compile(r'var hq_str_\w+=(?:"([^"]*)")?')


def _sina_symbol(code: str) -> str:
    c = _normalize_symbol(code)
    return ("sh" if c.startswith("6") else "sz") + c


def get_realtime_quote(code: str) -> dict:
    """Fetch real-time quote from Sina Finance (free, no subscription needed)."""
    key = _cache_key("rt", code)
    cached = cache.get(key)
    if cached is not None:
        return cached

    sym = _sina_symbol(code)
    url = f"https://hq.sinajs.cn/list={sym}"
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = "gbk"
        m = _SINA_RE.search(resp.text)
        if not m or not m.group(1):
            return {}
        fields = m.group(1).split(",")
        # Sina returns 32 fields for A-shares
        if len(fields) < 32:
            return {}
        quote = {
            "code": code,
            "name": fields[0],
            "open": float(fields[1]),
            "prev_close": float(fields[2]),
            "close": float(fields[3]),
            "high": float(fields[4]),
            "low": float(fields[5]),
            "bid": float(fields[6]),
            "ask": float(fields[7]),
            "volume": int(fields[8]),
            "amount": float(fields[9]),
            "date": fields[30],
            "time": fields[31],
        }
        cache.set(key, quote, cache.TTL_REALTIME)
        return quote
    except Exception as exc:
        logger.warning("get_realtime_quote(%s) failed: %s", code, exc)
        return {}


def get_realtime_quotes_batch(codes: list[str]) -> list[dict]:
    """Fetch multiple real-time quotes in a single Sina request."""
    if not codes:
        return []
    syms = ",".join(_sina_symbol(c) for c in codes)
    url = f"https://hq.sinajs.cn/list={syms}"
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0",
    }
    results: list[dict] = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "gbk"
        for code, m in zip(codes, _SINA_RE.finditer(resp.text)):
            if not m.group(1):
                continue
            fields = m.group(1).split(",")
            if len(fields) < 32:
                continue
            quote = {
                "code": code,
                "name": fields[0],
                "open": float(fields[1]),
                "prev_close": float(fields[2]),
                "close": float(fields[3]),
                "high": float(fields[4]),
                "low": float(fields[5]),
                "bid": float(fields[6]),
                "ask": float(fields[7]),
                "volume": int(fields[8]),
                "amount": float(fields[9]),
                "date": fields[30],
                "time": fields[31],
            }
            results.append(quote)
            cache.set(_cache_key("rt", code), quote, cache.TTL_REALTIME)
    except Exception as exc:
        logger.warning("get_realtime_quotes_batch failed: %s", exc)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Historical daily OHLCV  (akshare)
# ──────────────────────────────────────────────────────────────────────────────

def get_daily_history(
    code: str,
    start: str = "",
    end: str = "",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Return daily OHLCV for an A-share stock via akshare.

    Parameters
    ----------
    code:   6-digit A-share code, e.g. "000001"
    start:  ISO date string "YYYY-MM-DD" (default: 1 year ago)
    end:    ISO date string "YYYY-MM-DD" (default: today)
    adjust: price-adjustment type "qfq" (前复权) | "hfq" (后复权) | "" (不复权)
    """
    code = _normalize_symbol(code)
    if not end:
        end = date.today().strftime("%Y%m%d")
    else:
        end = end.replace("-", "")
    if not start:
        start = (date.today() - timedelta(days=365)).strftime("%Y%m%d")
    else:
        start = start.replace("-", "")

    key = _cache_key("daily", code, start, end, adjust)
    cached = cache.get(key)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
        }
        df = df.rename(columns=col_map)
        df = df[df.columns.intersection(list(col_map.values()))]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        cache.set(key, df.to_dict(orient="records"), cache.TTL_DAILY)
        return df
    except Exception as exc:
        logger.warning("get_daily_history(%s) failed: %s", code, exc)
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Intraday minute-level data (akshare)
# ──────────────────────────────────────────────────────────────────────────────

def get_intraday(code: str, period: int = 15) -> pd.DataFrame:
    """
    Return intraday OHLCV bars.

    Parameters
    ----------
    code:   A-share code
    period: bar interval in minutes (1, 5, 15, 30, 60)
    """
    code = _normalize_symbol(code)
    key = _cache_key("intraday", code, period)
    cached = cache.get(key)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=code,
            start_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            end_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            period=str(period),
            adjust="qfq",
        )
        col_map = {
            "时间": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns=col_map)
        df = df[df.columns.intersection(list(col_map.values()))]
        cache.set(key, df.to_dict(orient="records"), cache.TTL_INTRADAY)
        return df
    except Exception as exc:
        logger.warning("get_intraday(%s, %d) failed: %s", code, period, exc)
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# yfinance fallback (for international stocks / indices)
# ──────────────────────────────────────────────────────────────────────────────

def get_yfinance_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: str = "",
    end: str = "",
) -> pd.DataFrame:
    """
    Fetch OHLCV via yfinance.  Useful for indices (^GSPC, ^HSI) or ETFs.
    For A-shares append .SS (Shanghai) or .SZ (Shenzhen) suffix.

    Supports both ``period`` (e.g. "1y") and explicit ``start``/``end`` dates
    (ISO format "YYYY-MM-DD").  When ``start`` or ``end`` are provided they
    take precedence over ``period``.
    """
    key = _cache_key("yf", ticker, period, interval, start, end)
    cached = cache.get(key)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        t = yf.Ticker(ticker)
        if start or end:
            df = t.history(
                start=start or None,
                end=end or None,
                interval=interval,
            )
        else:
            df = t.history(period=period, interval=interval)
        if df.empty:
            return df
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]]
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = "date"
        df = df.reset_index()
        cache.set(key, df.to_dict(orient="records"), cache.TTL_DAILY)
        return df
    except Exception as exc:
        logger.warning("get_yfinance_history(%s) failed: %s", ticker, exc)
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Chinese ETF / Fund history  (akshare)
# ──────────────────────────────────────────────────────────────────────────────

def get_fund_history(
    code: str,
    start: str = "",
    end: str = "",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Return daily OHLCV for a Chinese ETF or open-end fund via akshare.

    Parameters
    ----------
    code:   Fund / ETF code, e.g. "510300" (CSI 300 ETF)
    start:  ISO date string "YYYY-MM-DD" (default: 1 year ago)
    end:    ISO date string "YYYY-MM-DD" (default: today)
    adjust: Price-adjustment type "qfq" | "hfq" | "" (default: "qfq")
    """
    code = code.strip()
    if not end:
        end = date.today().strftime("%Y%m%d")
    else:
        end = end.replace("-", "")
    if not start:
        start = (date.today() - timedelta(days=365)).strftime("%Y%m%d")
    else:
        start = start.replace("-", "")

    key = _cache_key("fund", code, start, end, adjust)
    cached = cache.get(key)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
        }
        df = df.rename(columns=col_map)
        df = df[df.columns.intersection(list(col_map.values()))]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        cache.set(key, df.to_dict(orient="records"), cache.TTL_DAILY)
        return df
    except Exception as exc:
        logger.warning("get_fund_history(%s) failed: %s", code, exc)
        return pd.DataFrame()
