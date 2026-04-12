"""
FastAPI backend server for RustUI stock analysis tool.

Start with:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
Or:
    python -m backend.main
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import cache, data_sources, indicators
from .alpha_engine import AlphaEngine, AlphaExpressionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RustUI Stock Backend",
    description="Real-time A-share data, technical indicators, and alpha factors",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class AlphaRequest(BaseModel):
    code: str
    expression: str
    start: Optional[str] = None
    end: Optional[str] = None
    adjust: str = "qfq"


class BatchQuoteRequest(BaseModel):
    codes: List[str]


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cache": cache.stats()}


# ──────────────────────────────────────────────────────────────────────────────
# Stock list
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/stocks")
def list_stocks():
    """Return all A-share stock codes and names."""
    df = data_sources.get_stock_list()
    return {"data": df.to_dict(orient="records")}


# ──────────────────────────────────────────────────────────────────────────────
# Real-time quotes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/quote/{code}")
def get_quote(code: str):
    """Single real-time quote from Sina Finance."""
    quote = data_sources.get_realtime_quote(code)
    if not quote:
        raise HTTPException(status_code=404, detail=f"Quote not found for {code}")
    return quote


@app.post("/api/quotes/batch")
def get_quotes_batch(req: BatchQuoteRequest):
    """Batch real-time quotes from Sina Finance."""
    quotes = data_sources.get_realtime_quotes_batch(req.codes)
    return {"data": quotes}


# ──────────────────────────────────────────────────────────────────────────────
# Historical OHLCV
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/history/{code}")
def get_history(
    code: str,
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    adjust: str = Query(default="qfq"),
):
    """Daily OHLCV history via akshare."""
    df = data_sources.get_daily_history(code, start or "", end or "", adjust)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No history found for {code}"
        )
    return {"code": code, "data": df.to_dict(orient="records")}


@app.get("/api/intraday/{code}")
def get_intraday(
    code: str,
    period: int = Query(default=15, ge=1, le=60),
):
    """Intraday minute bars via akshare."""
    df = data_sources.get_intraday(code, period)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No intraday data for {code}"
        )
    return {"code": code, "period": period, "data": df.to_dict(orient="records")}


@app.get("/api/yfinance/{ticker}")
def get_yfinance(
    ticker: str,
    period: str = Query(default="1y"),
    interval: str = Query(default="1d"),
):
    """Historical OHLCV via yfinance (international / indices)."""
    df = data_sources.get_yfinance_history(ticker, period, interval)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No yfinance data for {ticker}"
        )
    return {"ticker": ticker, "data": df.to_dict(orient="records")}


# ──────────────────────────────────────────────────────────────────────────────
# Technical indicators
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/indicators/{code}")
def get_indicators(
    code: str,
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    adjust: str = Query(default="qfq"),
):
    """Compute all standard technical indicators for a stock."""
    df = data_sources.get_daily_history(code, start or "", end or "", adjust)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {code}")

    dates = df["date"].tolist()
    result = indicators.compute_all(df)
    return {
        "code": code,
        "dates": dates,
        "ohlcv": {
            "open": df["open"].tolist(),
            "high": df["high"].tolist(),
            "low": df["low"].tolist(),
            "close": df["close"].tolist(),
            "volume": df["volume"].tolist(),
        },
        "indicators": result,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Alpha factor
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/alpha")
def compute_alpha(req: AlphaRequest):
    """
    Evaluate a custom alpha factor expression over historical data.

    Example expressions:
        rank(delta(close, 5)) / rank(volume)
        (close - delay(close, 20)) / delay(close, 20)
        ts_rank(volume, 20) - ts_rank(close, 20)
        corr(returns, log(volume), 10)
    """
    df = data_sources.get_daily_history(
        req.code, req.start or "", req.end or "", req.adjust
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {req.code}")

    try:
        engine = AlphaEngine(df)
        values = engine.eval_to_list(req.expression)
    except AlphaExpressionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dates = df["date"].tolist()
    return {
        "code": req.code,
        "expression": req.expression,
        "dates": dates,
        "values": values,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cache management
# ──────────────────────────────────────────────────────────────────────────────

@app.delete("/api/cache")
def clear_cache():
    """Clear all cached data."""
    cache.clear_all()
    return {"status": "cleared"}


@app.delete("/api/cache/expired")
def clear_expired_cache():
    """Remove only expired cache entries."""
    n = cache.clear_expired()
    return {"status": "ok", "deleted": n}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
