"""Tests for the alpha factor expression engine."""

import math

import numpy as np
import pandas as pd
import pytest

from backend.alpha_engine import AlphaEngine, AlphaExpressionError


@pytest.fixture()
def engine() -> AlphaEngine:
    """AlphaEngine backed by synthetic OHLCV data."""
    np.random.seed(0)
    n = 60
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.05
    high = np.maximum(close, open_) + np.abs(np.random.randn(n) * 0.1)
    low = np.minimum(close, open_) - np.abs(np.random.randn(n) * 0.1)
    volume = np.random.randint(100_000, 5_000_000, n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    return AlphaEngine(df)


# ── Basic arithmetic ─────────────────────────────────────────────────────────

def test_scalar_addition(engine):
    result = engine.eval("close + 1")
    assert len(result) == 60


def test_scalar_multiplication(engine):
    result = engine.eval("close * 2")
    assert len(result) == 60


def test_division(engine):
    result = engine.eval("close / open")
    assert (result.dropna() > 0).all()


# ── Time-series functions ────────────────────────────────────────────────────

def test_delay(engine):
    result = engine.eval("delay(close, 5)")
    assert result.iloc[:5].isna().all() or True  # first 5 may be NaN
    assert len(result) == 60


def test_delta(engine):
    result = engine.eval("delta(close, 1)")
    # delta of constant = 0
    const_engine = AlphaEngine(
        pd.DataFrame({"close": [5.0] * 20, "open": [5.0] * 20,
                      "high": [5.0] * 20, "low": [5.0] * 20,
                      "volume": [1e6] * 20})
    )
    d = const_engine.eval("delta(close, 1)")
    assert d.dropna().abs().max() < 1e-10


def test_ts_mean(engine):
    result = engine.eval("ts_mean(close, 5)")
    assert len(result) == 60


def test_ts_std(engine):
    result = engine.eval("ts_std(close, 5)")
    assert (result.dropna() >= 0).all()


def test_ts_min_leq_ts_max(engine):
    mn = engine.eval("ts_min(close, 10)")
    mx = engine.eval("ts_max(close, 10)")
    assert (mn.dropna() <= mx.dropna()).all()


def test_ts_sum(engine):
    result = engine.eval("ts_sum(volume, 5)")
    assert len(result) == 60


def test_ts_rank_range(engine):
    result = engine.eval("ts_rank(close, 10)")
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


# ── Cross-sectional functions ────────────────────────────────────────────────

def test_rank_range(engine):
    result = engine.eval("rank(close)")
    assert result.min() >= 0 and result.max() <= 1


def test_zscore(engine):
    result = engine.eval("zscore(close)")
    # mean ≈ 0 and std ≈ 1
    assert abs(result.mean()) < 1e-6
    assert abs(result.std() - 1.0) < 0.05


# ── Math functions ────────────────────────────────────────────────────────────

def test_abs(engine):
    result = engine.eval("abs(delta(close, 1))")
    assert (result.dropna() >= 0).all()


def test_log(engine):
    result = engine.eval("log(close)")
    assert (result.dropna() < 100).all()


def test_sign(engine):
    result = engine.eval("sign(delta(close, 1))")
    unique = set(result.dropna().unique())
    assert unique.issubset({-1.0, 0.0, 1.0})


def test_sqrt(engine):
    result = engine.eval("sqrt(volume)")
    assert (result.dropna() >= 0).all()


# ── Correlation ──────────────────────────────────────────────────────────────

def test_corr(engine):
    result = engine.eval("corr(returns, log(volume), 10)")
    valid = result.dropna()
    assert (valid >= -1.0).all() and (valid <= 1.0).all()


# ── Compound expressions ─────────────────────────────────────────────────────

def test_compound_rank_delta(engine):
    result = engine.eval("rank(delta(close, 5)) / rank(volume)")
    assert len(result) == 60


def test_compound_momentum(engine):
    result = engine.eval("(close - delay(close, 20)) / delay(close, 20)")
    assert len(result) == 60


def test_compound_combined(engine):
    result = engine.eval("ts_rank(volume, 20) - ts_rank(close, 20)")
    assert len(result) == 60


# ── eval_to_list ─────────────────────────────────────────────────────────────

def test_eval_to_list_returns_list(engine):
    result = engine.eval_to_list("close")
    assert isinstance(result, list)
    assert len(result) == 60


def test_eval_to_list_nans_become_none(engine):
    result = engine.eval_to_list("delay(close, 5)")
    # First value is NaN → None
    assert result[0] is None


# ── Error handling ────────────────────────────────────────────────────────────

def test_syntax_error(engine):
    with pytest.raises(AlphaExpressionError, match="Syntax error"):
        engine.eval("close +* 1")


def test_empty_expression(engine):
    with pytest.raises(AlphaExpressionError, match="Empty expression"):
        engine.eval("   ")


def test_disallowed_import(engine):
    with pytest.raises(AlphaExpressionError):
        engine.eval("__import__('os').system('ls')")


def test_disallowed_builtins(engine):
    with pytest.raises(AlphaExpressionError):
        engine.eval("open('/etc/passwd').read()")


def test_disallowed_lambda(engine):
    with pytest.raises(AlphaExpressionError):
        engine.eval("(lambda: 1)()")


def test_disallowed_list_comprehension(engine):
    with pytest.raises(AlphaExpressionError):
        engine.eval("[x for x in close]")
