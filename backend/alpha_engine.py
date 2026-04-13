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
from typing import Optional, Union

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
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

    def backtest(
        self,
        expression: str,
        annual_risk_free: float = 0.0,
        trading_days: int = 252,
    ) -> BacktestResult:
        """
        Backtest an alpha factor expression using the backtrader engine.

        The alpha signal is computed from *expression*, then passed to
        :func:`backend.backtest.run_backtest` which runs a full backtrader
        simulation and returns Jensen's alpha, beta, and Sharpe ratio.

        Parameters
        ----------
        expression:       Alpha factor expression string.
        annual_risk_free: Annualised risk-free rate (e.g. 0.02 for 2 %).
                          Default 0.
        trading_days:     Trading days per year used for annualisation.
                          Default 252.

        Returns
        -------
        BacktestResult
            Dataclass with ``alpha``, ``beta``, ``sharpe_ratio``,
            ``annualized_return``, ``annualized_volatility``, ``dates``,
            ``strategy_returns``, and ``cumulative_returns``.
        """
        signal = self.eval(expression)
        return run_backtest(
            self._df,
            signal,
            annual_risk_free=annual_risk_free,
            trading_days=trading_days,
        )

    def eval_to_list(self, expression: str) -> list:
        """Evaluate expression and return a list of floats (NaN → None)."""
        series = self.eval(expression)
        return [nan_to_none(v) for v in series]
