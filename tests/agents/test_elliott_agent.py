"""Tests for ElliottFibonacciAgent."""
import numpy as np
import pandas as pd
import pytest
from agents.elliott_agent import ElliottFibonacciAgent, ElliottResult


@pytest.fixture
def agent():
    return ElliottFibonacciAgent()


def _make_df(n=80, trend=1):
    """Generate synthetic OHLCV DataFrame."""
    np.random.seed(42)
    closes = 40000 + np.cumsum(np.random.normal(trend * 50, 200, n))
    highs  = closes + np.abs(np.random.normal(0, 100, n))
    lows   = closes - np.abs(np.random.normal(0, 100, n))
    return pd.DataFrame({
        "open": closes * 0.999, "high": highs,
        "low": lows, "close": closes, "volume": np.ones(n) * 1000,
    })


# ── Fibonacci levels ───────────────────────────────────────────────────────────

def test_fib_levels_all_keys_present(agent):
    levels = agent.calculate_fib_levels(50000, 40000)
    for key in ("0.236", "0.382", "0.500", "0.618", "0.786", "1.000", "1.618"):
        assert key in levels


def test_fib_618_correct_value(agent):
    levels = agent.calculate_fib_levels(50000, 40000)
    expected = 40000 + (50000 - 40000) * 0.618
    assert abs(levels["0.618"] - expected) < 1.0


def test_fib_extension_level_above_high(agent):
    levels = agent.calculate_fib_levels(50000, 40000)
    assert levels["1.618"] > 50000


def test_fib_500_is_midpoint(agent):
    levels = agent.calculate_fib_levels(60000, 40000)
    assert abs(levels["0.500"] - 50000) < 1.0


# ── Elliott analysis ───────────────────────────────────────────────────────────

def test_analyze_returns_elliott_result(agent):
    result = agent.analyze(_make_df())
    assert isinstance(result, ElliottResult)


def test_wave_count_between_1_and_5(agent):
    result = agent.analyze(_make_df())
    assert 1 <= result.wave_count <= 5


def test_score_bonus_in_range(agent):
    result = agent.analyze(_make_df())
    assert 0 <= result.score_bonus <= 10


def test_confidence_between_0_and_1(agent):
    result = agent.analyze(_make_df())
    assert 0.0 <= result.confidence <= 1.0


def test_time_projection_positive(agent):
    result = agent.analyze(_make_df())
    assert result.time_projection_days > 0


def test_wave_type_valid(agent):
    result = agent.analyze(_make_df())
    assert result.wave_type in ("impulse", "corrective")


def test_current_wave_is_string(agent):
    result = agent.analyze(_make_df())
    assert isinstance(result.current_wave, str)
    assert "wave" in result.current_wave


def test_analyze_minimal_df(agent):
    df = _make_df(n=15)
    result = agent.analyze(df)
    assert result.score_bonus == 0  # insufficient data
    assert result.confidence == 0.3


def test_analyze_bullish_bias(agent):
    result = agent.analyze(_make_df(trend=1), bias="bullish")
    assert isinstance(result, ElliottResult)


def test_score_adjustment_returns_int(agent):
    pts = agent.score_adjustment(_make_df(), "bullish")
    assert isinstance(pts, int)
    assert 0 <= pts <= 10


def test_format_telegram_contains_symbol(agent):
    text = agent.format_telegram("BTCUSDT", _make_df())
    assert "BTCUSDT" in text
    assert "Elliott" in text or "Onda" in text or "wave" in text.lower()
