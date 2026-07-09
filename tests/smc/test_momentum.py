"""
TDD tests for smc/momentum.py -- RSI + Bollinger Bands confirmation filter.
"""
import pandas as pd
import numpy as np
from smc.momentum import MomentumIndicators


def _df_trending_up(n=60, start=1.0, step=0.001):
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "open": closes, "high": [c + 0.0005 for c in closes],
        "low": [c - 0.0005 for c in closes], "close": closes,
    })


def _df_trending_down(n=60, start=1.0, step=0.001):
    closes = [start - i * step for i in range(n)]
    return pd.DataFrame({
        "open": closes, "high": [c + 0.0005 for c in closes],
        "low": [c - 0.0005 for c in closes], "close": closes,
    })


def _df_flat(n=60, price=1.0):
    closes = [price] * n
    return pd.DataFrame({
        "open": closes, "high": [price] * n, "low": [price] * n, "close": closes,
    })


# ── RSI ──────────────────────────────────────────────────────────────────

def test_rsi_neutral_with_insufficient_data():
    mi = MomentumIndicators(_df_flat(n=5))
    assert mi.rsi() == 50.0


def test_rsi_high_on_strong_uptrend():
    mi = MomentumIndicators(_df_trending_up())
    assert mi.rsi() >= 70


def test_rsi_low_on_strong_downtrend():
    mi = MomentumIndicators(_df_trending_down())
    assert mi.rsi() <= 30


def test_rsi_neutral_on_flat_price():
    mi = MomentumIndicators(_df_flat())
    assert 40 <= mi.rsi() <= 60


# ── Bollinger Bands ──────────────────────────────────────────────────────

def test_bollinger_bands_upper_above_lower():
    mi = MomentumIndicators(_df_trending_up())
    upper, mid, lower = mi.bollinger_bands()
    assert upper > mid > lower


def test_bollinger_bands_fallback_insufficient_data():
    mi = MomentumIndicators(_df_flat(n=5))
    upper, mid, lower = mi.bollinger_bands()
    assert upper == mid == lower


# ── score_for_signal ───────────────────────────────────────────────────

def test_long_penalized_when_overbought():
    mi = MomentumIndicators(_df_trending_up())
    result = mi.score_for_signal("LONG")
    assert result.pts_adjustment < 0
    assert "sobrecomprado" in result.reason or "Bollinger" in result.reason


def test_short_penalized_when_oversold():
    mi = MomentumIndicators(_df_trending_down())
    result = mi.score_for_signal("SHORT")
    assert result.pts_adjustment < 0


def test_long_not_penalized_when_neutral():
    mi = MomentumIndicators(_df_flat())
    result = mi.score_for_signal("LONG")
    assert result.pts_adjustment == 0


def test_short_not_penalized_on_uptrend():
    # Selling into an uptrend isn't "oversold" -- no RSI/BB penalty from this
    # filter (D1-trend-filter elsewhere already blocks countertrend entries)
    mi = MomentumIndicators(_df_trending_up())
    result = mi.score_for_signal("SHORT")
    assert result.pts_adjustment == 0
