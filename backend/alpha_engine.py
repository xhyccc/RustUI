"""
Alpha Factor Expression Engine.

Provides a sandboxed expression evaluator for custom alpha factors.
Supports:
  - Data fields: open, high, low, close, volume, vwap, returns, amount
  - Arithmetic: +, -, *, /, ** (unary -)
  - Time-series functions: delay(x, d), delta(x, d), ts_mean(x, d),
      ts_std(x, d), ts_rank(x, d), ts_min(x, d), ts_max(x, d), ts_sum(x, d)
  - Cross-sectional functions: rank(x), zscore(x)
  - Math functions: abs(x), log(x), sign(x), sqrt(x), pow(x, n)
  - Conditional: if_else(cond, a, b)
  - Correlation: corr(x, y, d)

Usage:
    engine = AlphaEngine(df)        # df must have OHLCV columns
    result = engine.eval("rank(delta(close, 5)) / rank(volume)")
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .utils import nan_to_none

# Allowed AST node types for security
_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Attribute,
    ast.Load,
}


class AlphaExpressionError(ValueError):
    """Raised when an expression is invalid or unsafe."""


# ──────────────────────────────────────────────────────────────────────────────
# Time-series helpers (operate on pandas Series)
# ──────────────────────────────────────────────────────────────────────────────

def _delay(x: pd.Series, d: int) -> pd.Series:
    return x.shift(int(d))


def _delta(x: pd.Series, d: int) -> pd.Series:
    return x - x.shift(int(d))


def _ts_mean(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(window=int(d), min_periods=1).mean()


def _ts_std(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(window=int(d), min_periods=1).std()


def _ts_min(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(window=int(d), min_periods=1).min()


def _ts_max(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(window=int(d), min_periods=1).max()


def _ts_sum(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(window=int(d), min_periods=1).sum()


def _ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Time-series rank: percentile rank within rolling window."""
    return x.rolling(window=int(d), min_periods=1).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=True
    )


def _rank(x: pd.Series) -> pd.Series:
    """Cross-sectional rank (rank along time axis, pct)."""
    return x.rank(pct=True)


def _zscore(x: pd.Series) -> pd.Series:
    mu, sigma = x.mean(), x.std()
    if sigma == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sigma


def _corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(window=int(d), min_periods=2).corr(y)


def _if_else(
    cond: Union[pd.Series, bool],
    a: Union[pd.Series, float],
    b: Union[pd.Series, float],
) -> pd.Series:
    return pd.Series(np.where(cond, a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Safe AST validator
# ──────────────────────────────────────────────────────────────────────────────

def _validate_ast(node: ast.AST) -> None:
    """Recursively check that only allowed node types appear."""
    if type(node) not in _ALLOWED_NODES:
        raise AlphaExpressionError(
            f"Disallowed expression node: {type(node).__name__}"
        )
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


# ──────────────────────────────────────────────────────────────────────────────
# AlphaEngine
# ──────────────────────────────────────────────────────────────────────────────

class AlphaEngine:
    """Evaluate alpha factor expressions over an OHLCV DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df: DataFrame with columns open, high, low, close, volume
            Optional: amount (成交额), vwap
        """
        self._df = df.copy()
        self._build_namespace()

    def _build_namespace(self) -> None:
        df = self._df
        c = df.get("close", pd.Series(dtype=float))
        o = df.get("open", pd.Series(dtype=float))
        h = df.get("high", pd.Series(dtype=float))
        lo = df.get("low", pd.Series(dtype=float))
        v = df.get("volume", pd.Series(dtype=float))
        amt = df.get("amount", pd.Series(dtype=float))

        # Compute vwap if not provided
        if "vwap" in df.columns:
            vwap_s = df["vwap"]
        elif not v.empty and not (h.empty or lo.empty or c.empty):
            typical = (h + lo + c) / 3
            vwap_s = (typical * v).cumsum() / v.cumsum().replace(0, np.nan)
        else:
            vwap_s = pd.Series(dtype=float)

        returns = c.pct_change()

        self._ns: dict = {
            # Data fields
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": v,
            "amount": amt,
            "vwap": vwap_s,
            "returns": returns,
            # Time-series functions
            "delay": _delay,
            "delta": _delta,
            "ts_mean": _ts_mean,
            "ts_std": _ts_std,
            "ts_min": _ts_min,
            "ts_max": _ts_max,
            "ts_sum": _ts_sum,
            "ts_rank": _ts_rank,
            "corr": _corr,
            # Cross-sectional
            "rank": _rank,
            "zscore": _zscore,
            # Math
            "abs": lambda x: x.abs() if hasattr(x, "abs") else abs(x),
            "log": lambda x: np.log(x) if hasattr(x, "__len__") else math.log(x),
            "sign": lambda x: np.sign(x),
            "sqrt": lambda x: np.sqrt(x),
            "pow": lambda x, n: x ** n,
            "if_else": _if_else,
            # Constants
            "nan": float("nan"),
            "inf": float("inf"),
        }

    def eval(self, expression: str) -> pd.Series:
        """
        Evaluate an alpha factor expression and return a pd.Series.

        Raises AlphaExpressionError on invalid or unsafe input.
        """
        expression = expression.strip()
        if not expression:
            raise AlphaExpressionError("Empty expression")

        # Parse and validate AST
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise AlphaExpressionError(f"Syntax error: {exc}") from exc

        _validate_ast(tree.body)

        # Compile and evaluate in restricted namespace
        try:
            code = compile(tree, "<alpha>", "eval")
            result = eval(code, {"__builtins__": {}}, self._ns)  # noqa: S307
        except Exception as exc:
            raise AlphaExpressionError(f"Evaluation error: {exc}") from exc

        if isinstance(result, (int, float)):
            result = pd.Series(
                [result] * len(self._df), index=self._df.index
            )
        if not isinstance(result, pd.Series):
            raise AlphaExpressionError(
                f"Expression must return a Series, got {type(result).__name__}"
            )
        result.index = self._df.index
        return result

    def eval_to_list(self, expression: str) -> list:
        """Evaluate expression and return a list of floats (NaN → None)."""
        series = self.eval(expression)
        return [nan_to_none(v) for v in series]

    def backtest(
        self,
        expression: str,
        annual_risk_free: float = 0.0,
        trading_days: int = 252,
    ) -> "BacktestResult":
        """
        Run a simple backtest for a given alpha factor expression.

        Methodology
        -----------
        1. Evaluate the alpha factor to get a signal series.
        2. Derive daily positions as the sign of the **lagged** signal (use
           yesterday's factor value to trade today), clipped to {-1, 0, +1}.
        3. Compute strategy returns = position × market daily returns.
        4. Fit an OLS regression of strategy returns on market returns to
           obtain Jensen's alpha (intercept) and beta (slope).
        5. Compute the annualised Sharpe ratio on strategy excess returns.

        Parameters
        ----------
        expression:       Alpha factor expression string.
        annual_risk_free: Annualised risk-free rate (e.g. 0.02 for 2 %).
                          Default 0.
        trading_days:     Number of trading days per year used for
                          annualisation.  Default 252.

        Returns
        -------
        BacktestResult dataclass with alpha, beta, sharpe_ratio, and
        supporting series.
        """
        factor = self.eval(expression)
        market_returns: pd.Series = self._df.get(
            "close", pd.Series(dtype=float)
        ).pct_change()

        # Positions: sign of previous day's factor, mapped to {-1, 0, +1}
        positions = np.sign(factor.shift(1))

        strategy_returns = (positions * market_returns).dropna()
        market_returns_aligned = market_returns.reindex(strategy_returns.index)

        if len(strategy_returns) < 2:
            return BacktestResult(
                alpha=None,
                beta=None,
                sharpe_ratio=None,
                annualized_return=None,
                annualized_volatility=None,
                dates=[],
                strategy_returns=[],
                cumulative_returns=[],
            )

        daily_rf = annual_risk_free / trading_days

        # OLS: strategy_returns = alpha_daily + beta * market_returns + ε
        mkt = market_returns_aligned.values.astype(float)
        strat = strategy_returns.values.astype(float)
        mask = np.isfinite(mkt) & np.isfinite(strat)
        mkt_clean, strat_clean = mkt[mask], strat[mask]

        if len(mkt_clean) < 2 or np.var(mkt_clean) == 0:
            beta_val = 0.0
            alpha_daily = float(np.mean(strat_clean))
        else:
            # beta = Cov(strat, mkt) / Var(mkt)
            beta_val = float(np.cov(strat_clean, mkt_clean)[0, 1] / np.var(mkt_clean))
            alpha_daily = float(np.mean(strat_clean) - beta_val * np.mean(mkt_clean))

        annualized_alpha = alpha_daily * trading_days
        annualized_return = float(np.mean(strat_clean)) * trading_days

        excess_returns = strat_clean - daily_rf
        vol = float(np.std(strat_clean, ddof=1))
        annualized_volatility = vol * math.sqrt(trading_days)
        sharpe = (
            float(np.mean(excess_returns)) / vol * math.sqrt(trading_days)
            if vol > 0
            else None
        )

        # Cumulative returns for the full (unmasked) strategy return series
        cum_returns = (1 + strategy_returns).cumprod() - 1

        df_index = self._df.get("date", pd.Series(dtype=str))
        if not df_index.empty:
            date_series = df_index.reindex(strategy_returns.index)
            dates_out = [str(d) for d in date_series.tolist()]
        else:
            dates_out = [str(i) for i in strategy_returns.index.tolist()]

        return BacktestResult(
            alpha=round(annualized_alpha, 6),
            beta=round(beta_val, 6),
            sharpe_ratio=round(sharpe, 6) if sharpe is not None else None,
            annualized_return=round(annualized_return, 6),
            annualized_volatility=round(annualized_volatility, 6),
            dates=dates_out,
            strategy_returns=[nan_to_none(v) for v in strategy_returns],
            cumulative_returns=[nan_to_none(v) for v in cum_returns],
        )


@dataclass
class BacktestResult:
    """Results returned by :meth:`AlphaEngine.backtest`."""

    alpha: Optional[float]
    """Jensen's alpha, annualised."""

    beta: Optional[float]
    """Systematic risk relative to the stock's own daily returns."""

    sharpe_ratio: Optional[float]
    """Annualised Sharpe ratio of the strategy (excess returns basis)."""

    annualized_return: Optional[float]
    """Annualised mean strategy return."""

    annualized_volatility: Optional[float]
    """Annualised volatility of strategy excess returns."""

    dates: list
    """Dates corresponding to each strategy return observation."""

    strategy_returns: list
    """Daily strategy returns (NaN → None)."""

    cumulative_returns: list
    """Cumulative strategy returns from the first trade day (NaN → None)."""
