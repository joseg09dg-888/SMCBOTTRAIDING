"""
TDD tests for smc/bill_williams.py -- Alligator + Awesome Oscillator.
"""
import pandas as pd
from smc.bill_williams import BillWilliamsIndicators


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


# ── Alligator ────────────────────────────────────────────────────────────

def test_alligator_fallback_insufficient_data():
    bw = BillWilliamsIndicators(_df_flat(n=10))
    jaw, teeth, lips = bw.alligator()
    assert jaw == teeth == lips


def test_alligator_lines_diverge_on_strong_trend():
    bw = BillWilliamsIndicators(_df_trending_up(n=80))
    jaw, teeth, lips = bw.alligator()
    # On a clean trend the fast line (lips) should be ahead of the slow one (jaw)
    assert lips > jaw


# ── Awesome Oscillator ───────────────────────────────────────────────────

def test_ao_neutral_insufficient_data():
    bw = BillWilliamsIndicators(_df_flat(n=10))
    assert bw.awesome_oscillator() == 0.0


def test_ao_positive_on_uptrend():
    bw = BillWilliamsIndicators(_df_trending_up())
    assert bw.awesome_oscillator() > 0


def test_ao_negative_on_downtrend():
    bw = BillWilliamsIndicators(_df_trending_down())
    assert bw.awesome_oscillator() < 0


# ── score_for_signal ─────────────────────────────────────────────────────

def test_long_penalized_when_ao_contradicts():
    bw = BillWilliamsIndicators(_df_trending_down(n=80))
    result = bw.score_for_signal("LONG")
    assert result.pts_adjustment < 0


def test_short_penalized_when_ao_contradicts():
    bw = BillWilliamsIndicators(_df_trending_up(n=80))
    result = bw.score_for_signal("SHORT")
    assert result.pts_adjustment < 0


def test_long_not_penalized_when_ao_agrees():
    bw = BillWilliamsIndicators(_df_trending_up(n=80))
    result = bw.score_for_signal("LONG")
    assert result.pts_adjustment == 0


def test_flat_market_penalized_alligator_asleep():
    bw = BillWilliamsIndicators(_df_flat(n=80))
    result = bw.score_for_signal("LONG")
    assert result.pts_adjustment < 0
    assert "dormido" in result.reason
