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
from .backtest import BacktestResult

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


class BacktestRequest(BaseModel):
    code: str
    expression: str
    start: Optional[str] = None
    end: Optional[str] = None
    adjust: str = "qfq"
    annual_risk_free: float = 0.0
    trading_days: int = 252
    asset_type: str = "stock"
    """Asset type: 'stock' (A-share via akshare) | 'fund' (ETF/fund via akshare) | 'yfinance' (international via yfinance)."""


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


@app.post("/api/alpha/backtest")
def backtest_alpha(req: BacktestRequest):
    """
    Backtest a custom alpha factor expression using backtrader and return
    alpha (Jensen's), beta, and Sharpe ratio over the requested date range.

    The backtest is powered by `backtrader <https://github.com/mementum/backtrader>`_.
    The alpha factor signal is computed on the historical OHLCV data, then a
    :class:`AlphaSignalStrategy` translates the **lagged** signal into long /
    flat / short positions (no look-ahead bias).  Jensen's alpha and beta are
    estimated via OLS regression; the Sharpe ratio is extracted from
    backtrader's built-in ``SharpeRatio`` analyser.

    Parameters
    ----------
    code:             Asset code or ticker (interpretation depends on *asset_type*)
    expression:       Alpha factor expression, e.g. "rank(delta(close, 5))"
    start:            ISO date "YYYY-MM-DD" (default: 1 year ago)
    end:              ISO date "YYYY-MM-DD" (default: today)
    adjust:           Price adjustment: "qfq" | "hfq" | "" (default: "qfq")
    annual_risk_free: Annualised risk-free rate (default: 0.0)
    trading_days:     Trading days per year for annualisation (default: 252)
    asset_type:       Data source — "stock" (A-share via akshare, default) |
                      "fund" (Chinese ETF/fund via akshare) |
                      "yfinance" (international ticker via yfinance)
    """
    asset_type = (req.asset_type or "stock").lower()
    if asset_type not in {"stock", "fund", "yfinance"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported asset_type '{asset_type}'. Choose 'stock', 'fund', or 'yfinance'.",
        )

    if asset_type == "fund":
        df = data_sources.get_fund_history(
            req.code, req.start or "", req.end or "", req.adjust
        )
    elif asset_type == "yfinance":
        df = data_sources.get_yfinance_history(
            req.code,
            start=req.start or "",
            end=req.end or "",
        )
    else:
        df = data_sources.get_daily_history(
            req.code, req.start or "", req.end or "", req.adjust
        )

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {req.code}")

    try:
        engine = AlphaEngine(df)
        result = engine.backtest(
            req.expression,
            annual_risk_free=req.annual_risk_free,
            trading_days=req.trading_days,
        )
    except AlphaExpressionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "code": req.code,
        "expression": req.expression,
        "asset_type": asset_type,
        "start": req.start,
        "end": req.end,
        "alpha": result.alpha,
        "beta": result.beta,
        "sharpe_ratio": result.sharpe_ratio,
        "annualized_return": result.annualized_return,
        "annualized_volatility": result.annualized_volatility,
        "dates": result.dates,
        "strategy_returns": result.strategy_returns,
        "cumulative_returns": result.cumulative_returns,
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
