"""
Backtrader-based backtest engine for alpha factor strategies.

The engine:
  1. Takes a pre-computed alpha signal series from AlphaEngine.
  2. Feeds OHLCV data into a backtrader Cerebro instance via PandasData.
  3. Runs AlphaSignalStrategy, which translates the lagged signal into
     long (+1) / flat (0) / short (-1) positions using order_target_percent.
  4. Extracts alpha (Jensen's), beta, Sharpe ratio, annualised return, and
     annualised volatility from backtrader's built-in analyzers plus OLS.

Supports any asset whose historical data can be expressed as a standard
OHLCV DataFrame (A-share stocks, Chinese ETF/funds, international tickers).
"""

from __future__ import annotations

# ── Python ≥ 3.10 compatibility patch for backtrader ──────────────────────────
# backtrader still references removed top-level names (collections.Callable etc.)
import collections
import collections.abc

for _bt_compat_name in (
    "Callable",
    "Iterable",
    "Iterator",
    "Mapping",
    "MutableMapping",
    "MutableSequence",
    "MutableSet",
    "Sequence",
    "Set",
    "Sized",
    "Hashable",
    "Container",
):
    if not hasattr(collections, _bt_compat_name):
        setattr(collections, _bt_compat_name, getattr(collections.abc, _bt_compat_name))
# ──────────────────────────────────────────────────────────────────────────────

import math
from dataclasses import dataclass, field
from typing import Optional

import backtrader as bt
import numpy as np
import pandas as pd

from .utils import nan_to_none


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Statistics returned by :func:`run_backtest`."""

    alpha: Optional[float]
    """Jensen's alpha (annualised) from OLS regression."""

    beta: Optional[float]
    """Systematic risk (slope of strategy returns vs market returns)."""

    sharpe_ratio: Optional[float]
    """Annualised Sharpe ratio from backtrader's SharpeRatio analyser."""

    annualized_return: Optional[float]
    """Annualised strategy return from backtrader's Returns analyser."""

    annualized_volatility: Optional[float]
    """Annualised volatility of daily strategy returns."""

    dates: list = field(default_factory=list)
    """Trade dates (YYYY-MM-DD strings) corresponding to each daily return."""

    strategy_returns: list = field(default_factory=list)
    """Daily strategy portfolio returns (NaN → None)."""

    cumulative_returns: list = field(default_factory=list)
    """Cumulative strategy returns from the first bar (NaN → None)."""


# ──────────────────────────────────────────────────────────────────────────────
# Backtrader Strategy
# ──────────────────────────────────────────────────────────────────────────────

class AlphaSignalStrategy(bt.Strategy):
    """
    Backtrader strategy driven by a pre-computed, pre-shifted alpha signal.

    ``params.signal_map`` is a ``{date_str: signal_value}`` dictionary built
    from the alpha factor series *already shifted by one bar* so that today's
    position is based on yesterday's signal (no look-ahead bias).

    Position sizing uses :meth:`order_target_percent`:
      * signal > 0  → long  95 % of portfolio value
      * signal < 0  → short 95 % of portfolio value
      * signal == 0 → flat
    """

    params = (("signal_map", {}),)

    def next(self) -> None:
        dt_str = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        signal = self.p.signal_map.get(dt_str)

        if signal is None or (isinstance(signal, float) and math.isnan(signal)):
            self.order_target_percent(target=0.0)
            return

        pos_sign = int(np.sign(float(signal)))
        target_pct = 0.95 * pos_sign
        self.order_target_percent(target=target_pct)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    signal_series: pd.Series,
    initial_cash: float = 100_000.0,
    annual_risk_free: float = 0.0,
    trading_days: int = 252,
) -> BacktestResult:
    """
    Run a backtrader backtest using a pre-computed alpha factor signal.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a ``date`` column (YYYY-MM-DD strings or
        datetime-convertible) and columns ``open``, ``high``, ``low``,
        ``close``, ``volume``.
    signal_series:
        Alpha factor signal as a :class:`pandas.Series` aligned to *df*.
        The series is lagged by one bar internally (use yesterday's signal to
        take today's position) to avoid look-ahead bias.
    initial_cash:
        Starting portfolio cash.  Default 100 000.
    annual_risk_free:
        Annualised risk-free rate used for excess-return / Sharpe
        calculations.  Default 0.
    trading_days:
        Number of trading days per year for annualisation.  Default 252.

    Returns
    -------
    BacktestResult
        Dataclass with ``alpha``, ``beta``, ``sharpe_ratio``,
        ``annualized_return``, ``annualized_volatility``, ``dates``,
        ``strategy_returns``, and ``cumulative_returns``.
    """
    _empty = BacktestResult(
        alpha=None,
        beta=None,
        sharpe_ratio=None,
        annualized_return=None,
        annualized_volatility=None,
    )

    if df is None or len(df) < 2:
        return _empty

    # ── Prepare a datetime-indexed DataFrame for PandasData ──────────────────
    bt_df = df.copy()
    bt_df["date"] = pd.to_datetime(bt_df["date"])
    bt_df = bt_df.set_index("date").sort_index()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in bt_df.columns:
            bt_df[col] = 0.0
    bt_df["openinterest"] = 0.0

    # ── Build the signal map (lagged by 1 bar) ────────────────────────────────
    date_strs: list[str] = [
        str(d) for d in pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()
    ]
    shifted_vals = signal_series.shift(1).tolist()
    signal_map: dict[str, float] = {}
    for d, v in zip(date_strs, shifted_vals):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            signal_map[d] = float(v)

    # ── Set up Cerebro ────────────────────────────────────────────────────────
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    data_feed = bt.feeds.PandasData(dataname=bt_df)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(AlphaSignalStrategy, signal_map=signal_map)

    daily_rf = annual_risk_free / trading_days
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=daily_rf,
        annualize=True,
        timeframe=bt.TimeFrame.Days,
        factor=trading_days,
    )
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        _name="timereturn",
        timeframe=bt.TimeFrame.Days,
    )
    cerebro.addanalyzer(
        bt.analyzers.Returns,
        _name="returns",
        tann=trading_days,
    )

    results = cerebro.run()
    strat = results[0]

    # ── Extract Sharpe ratio ──────────────────────────────────────────────────
    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    sharpe_ratio = (
        round(float(sharpe_raw), 6)
        if sharpe_raw is not None and math.isfinite(float(sharpe_raw))
        else None
    )

    # ── Extract time-series of daily portfolio returns ────────────────────────
    tr_analysis = strat.analyzers.timereturn.get_analysis()
    if not tr_analysis:
        return _empty

    sorted_keys = sorted(tr_analysis.keys())
    strat_dates = [dt.strftime("%Y-%m-%d") for dt in sorted_keys]
    strat_returns_raw = [tr_analysis[dt] for dt in sorted_keys]

    # ── Annualised return from Returns analyser ───────────────────────────────
    ann_return_raw = strat.analyzers.returns.get_analysis().get("rnorm")
    annualized_return = (
        round(float(ann_return_raw), 6)
        if ann_return_raw is not None
        else None
    )

    # ── Annualised volatility ─────────────────────────────────────────────────
    strat_arr = np.array(
        [r for r in strat_returns_raw if r is not None and math.isfinite(r)],
        dtype=float,
    )
    annualized_volatility = (
        round(float(np.std(strat_arr, ddof=1)) * math.sqrt(trading_days), 6)
        if len(strat_arr) > 1
        else None
    )

    # ── Beta and Jensen's alpha via OLS ──────────────────────────────────────
    market_ret_series = df["close"].pct_change()
    market_map = {d: r for d, r in zip(date_strs, market_ret_series.tolist())}

    mkt_arr = np.array(
        [market_map.get(d, np.nan) for d in strat_dates], dtype=float
    )
    strat_aligned = np.array(strat_returns_raw, dtype=float)
    mask = np.isfinite(mkt_arr) & np.isfinite(strat_aligned)
    mkt_clean, strat_clean = mkt_arr[mask], strat_aligned[mask]

    if len(mkt_clean) >= 2 and np.var(mkt_clean) > 0:
        beta_val = float(
            np.cov(strat_clean, mkt_clean)[0, 1] / np.var(mkt_clean)
        )
        alpha_daily = float(
            np.mean(strat_clean) - beta_val * np.mean(mkt_clean)
        )
        ann_alpha = round(alpha_daily * trading_days, 6)
        beta_val = round(beta_val, 6)
    else:
        beta_val = None
        ann_alpha = (
            round(float(np.mean(strat_clean)) * trading_days, 6)
            if len(strat_clean) > 0
            else None
        )

    # ── Cumulative returns ────────────────────────────────────────────────────
    cum_rets: list = []
    cum = 1.0
    for r in strat_returns_raw:
        if r is not None and math.isfinite(r):
            cum *= 1.0 + r
        cum_rets.append(cum - 1.0)

    return BacktestResult(
        alpha=ann_alpha,
        beta=beta_val,
        sharpe_ratio=sharpe_ratio,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        dates=strat_dates,
        strategy_returns=[nan_to_none(r) for r in strat_returns_raw],
        cumulative_returns=[nan_to_none(r) for r in cum_rets],
    )
