"""
Tests for agents/retail_psychology_agent.py
"""

import numpy as np
import pandas as pd
import pytest

from agents.retail_psychology_agent import (
    RetailPsychologyAgent,
    RetailPsychologySignal,
    PsychLevel,
    StopHuntSignal,
    LiquidationZone,
    PSYCH_LEVELS_MAP,
)


agent = RetailPsychologyAgent()


def make_flat_df(price: float = 50000.0, n: int = 60) -> pd.DataFrame:
    """Flat OHLC DataFrame."""
    return pd.DataFrame({
        "open":  [price] * n,
        "high":  [price * 1.001] * n,
        "low":   [price * 0.999] * n,
        "close": [price] * n,
    })


def make_uptrend_df(n: int = 60, start: float = 47000.0, end: float = 55000.0) -> pd.DataFrame:
    prices = np.linspace(start, end, n)
    return pd.DataFrame({
        "open":  prices * 0.999,
        "high":  prices * 1.002,
        "low":   prices * 0.998,
        "close": prices,
    })


def make_downtrend_df(n: int = 60, start: float = 55000.0, end: float = 47000.0) -> pd.DataFrame:
    prices = np.linspace(start, end, n)
    return pd.DataFrame({
        "open":  prices * 1.001,
        "high":  prices * 1.002,
        "low":   prices * 0.998,
        "close": prices,
    })


def make_stop_hunt_bull_df(level: float = 50000.0) -> pd.DataFrame:
    """
    3 candles:
    [0] approaches level
    [1] wicks BELOW level - threshold (sweep)
    [2] closes ABOVE level (recovery)
    """
    sweep_low  = level * (1 - 0.005)   # 0.5% below → > 0.3% threshold
    rows = [
        {"open": level + 50, "high": level + 100, "low": level - 50,  "close": level + 20},
        {"open": level - 10, "high": level + 20,  "low": sweep_low,   "close": level - 5},
        {"open": level + 10, "high": level + 200, "low": level + 5,   "close": level + 150},
    ]
    return pd.DataFrame(rows)


def make_stop_hunt_bear_df(level: float = 50000.0) -> pd.DataFrame:
    """
    [1] wicks ABOVE level + threshold
    [2] closes BELOW level
    """
    sweep_high = level * (1 + 0.005)
    rows = [
        {"open": level - 50, "high": level + 50,  "low": level - 100, "close": level - 20},
        {"open": level + 10, "high": sweep_high,  "low": level - 20,  "close": level + 5},
        {"open": level - 10, "high": level - 5,   "low": level - 200, "close": level - 150},
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Psychological levels tests
# ------------------------------------------------------------------

class TestPsychologicalLevels:

    def test_psych_levels_btc_has_round_numbers(self):
        """BTC psychological levels must be multiples of 10,000."""
        raw = PSYCH_LEVELS_MAP["BTCUSDT"]
        for level in raw:
            assert level % 10000 == 0, f"{level} is not a multiple of 10,000"

    def test_nearest_level_correctly_identified(self):
        """The nearest level is the one with minimum distance."""
        above, below, nearest = agent.get_psychological_levels("BTCUSDT", 50500.0)
        all_levels = above + below
        if all_levels and nearest:
            min_dist = min(l.distance_pct for l in all_levels)
            assert abs(nearest.distance_pct - min_dist) < 1e-6

    def test_levels_above_are_above_price(self):
        """All 'above' levels must have price > current_price."""
        above, below, _ = agent.get_psychological_levels("BTCUSDT", 50000.0)
        for lvl in above:
            assert lvl.price > 50000.0
            assert lvl.above_or_below == "above"

    def test_levels_below_are_below_price(self):
        """All 'below' levels must have price < current_price."""
        above, below, _ = agent.get_psychological_levels("BTCUSDT", 50000.0)
        for lvl in below:
            assert lvl.price < 50000.0
            assert lvl.above_or_below == "below"

    def test_n_above_n_below_limits(self):
        """Returns at most n_above and n_below levels."""
        above, below, _ = agent.get_psychological_levels("BTCUSDT", 50000.0, n_above=2, n_below=2)
        assert len(above) <= 2
        assert len(below) <= 2


# ------------------------------------------------------------------
# Stop hunt tests
# ------------------------------------------------------------------

class TestStopHunt:

    def test_stop_hunt_bull_sweep_detected(self):
        """Bull stop hunt: wick below level, then close above → detected."""
        df = make_stop_hunt_bull_df(level=50000.0)
        result = agent.detect_stop_hunt(df, "BTCUSDT")
        assert isinstance(result, StopHuntSignal)
        assert result.detected is True
        assert result.direction == "bull_hunt"
        assert result.score_bonus == 15

    def test_stop_hunt_bear_sweep_detected(self):
        """Bear stop hunt: wick above level, then close below → detected."""
        df = make_stop_hunt_bear_df(level=50000.0)
        result = agent.detect_stop_hunt(df, "BTCUSDT")
        assert isinstance(result, StopHuntSignal)
        assert result.detected is True
        assert result.direction == "bear_hunt"
        assert result.score_bonus == 15

    def test_stop_hunt_not_detected_normal_move(self):
        """Normal move without sweep → not detected."""
        df = make_flat_df(50000.0, 10)
        result = agent.detect_stop_hunt(df, "BTCUSDT")
        assert result.detected is False
        assert result.score_bonus == 0

    def test_stop_hunt_insufficient_data(self):
        """Less than 3 candles → no hunt."""
        df = make_flat_df(50000.0, 2)
        result = agent.detect_stop_hunt(df, "BTCUSDT")
        assert result.detected is False

    def test_stop_hunt_score_bonus_capped_at_15(self):
        """score_bonus must be 0 or 15."""
        df = make_stop_hunt_bull_df(50000.0)
        result = agent.detect_stop_hunt(df, "BTCUSDT")
        assert result.score_bonus in (0, 15)


# ------------------------------------------------------------------
# Liquidation zones tests
# ------------------------------------------------------------------

class TestLiquidationZones:

    def test_liquidation_zones_synthetic_fallback(self):
        """When Coinglass is unavailable, synthetic zones are returned."""
        # Pass a symbol that definitely won't have real data
        zones = agent.get_liquidation_zones("FAKE_SYMBOL_XYZ", 50000.0)
        assert isinstance(zones, list)
        assert len(zones) >= 2   # at least 1 above + 1 below

    def test_synthetic_zones_have_correct_sides(self):
        """Synthetic zones have correct side labels."""
        zones = agent.get_liquidation_zones("BTCUSDT", 50000.0)
        sides = {z.side for z in zones}
        assert "long_liquidations" in sides
        assert "short_liquidations" in sides

    def test_liquidation_zones_distance_positive(self):
        """All distance_pct values must be positive."""
        zones = agent.get_liquidation_zones("BTCUSDT", 50000.0)
        for z in zones:
            assert z.distance_pct >= 0


# ------------------------------------------------------------------
# Retail sentiment tests
# ------------------------------------------------------------------

class TestRetailSentiment:

    def test_retail_sentiment_uptrend_majority_long(self):
        """Strong uptrend → retail is mostly long (> 60%)."""
        df = make_uptrend_df(60, 47000, 55000)
        pct = agent.estimate_retail_sentiment(df)
        assert pct > 60.0, f"Expected > 60% long in uptrend, got {pct}"

    def test_retail_sentiment_downtrend_majority_short(self):
        """Strong downtrend → retail is mostly short (< 40% long)."""
        df = make_downtrend_df(60, 55000, 47000)
        pct = agent.estimate_retail_sentiment(df)
        assert pct < 40.0, f"Expected < 40% long in downtrend, got {pct}"

    def test_retail_sentiment_flat_is_neutral(self):
        """Flat market → sentiment ≈ 50%."""
        df = make_flat_df(50000.0, 60)
        pct = agent.estimate_retail_sentiment(df)
        assert 40.0 <= pct <= 60.0

    def test_retail_sentiment_range(self):
        """Sentiment must always be in [0, 100]."""
        for df in (make_flat_df(), make_uptrend_df(), make_downtrend_df()):
            pct = agent.estimate_retail_sentiment(df)
            assert 0.0 <= pct <= 100.0


# ------------------------------------------------------------------
# Combined signal tests
# ------------------------------------------------------------------

class TestGetSignal:

    def test_contrarian_bias_opposite_of_retail(self):
        """When retail is mostly long, contrarian bias is bearish."""
        df = make_uptrend_df(60, 47000, 55000)
        sig = agent.get_signal("BTCUSDT", df)
        assert isinstance(sig, RetailPsychologySignal)
        if sig.retail_long_pct > 60:
            assert sig.contrarian_bias == "bearish"
        elif sig.retail_long_pct < 40:
            assert sig.contrarian_bias == "bullish"
        else:
            assert sig.contrarian_bias == "neutral"

    def test_score_adjustment_stop_hunt_gives_15pts(self):
        """Stop hunt event adds +15 to score."""
        df = make_stop_hunt_bull_df(50000.0)
        score = agent.score_adjustment("BTCUSDT", df, "bullish")
        # At minimum, the stop hunt bonus of 15 should be included
        assert score >= 15

    def test_score_bonus_capped_at_35(self):
        """total_bonus must not exceed 35."""
        df = make_stop_hunt_bull_df(50000.0)
        sig = agent.get_signal("BTCUSDT", df)
        assert sig.total_bonus <= 35

    def test_score_bonus_non_negative(self):
        """total_bonus must be >= 0."""
        df = make_flat_df(50000.0, 60)
        sig = agent.get_signal("BTCUSDT", df)
        assert sig.total_bonus >= 0

    def test_format_telegram_has_psych_level(self):
        """format_telegram output mentions a psychological level."""
        df = make_flat_df(50000.0, 60)
        text = agent.format_telegram("BTCUSDT", df)
        assert "BTCUSDT" in text
        # Should mention either psych level or retail info
        assert any(kw in text for kw in ("psych", "Psych", "level", "Level", "Retail", "retail"))

    def test_format_telegram_contains_symbol(self):
        """format_telegram always includes the symbol name."""
        df = make_flat_df(50000.0, 60)
        text = agent.format_telegram("ETHUSDT", df)
        assert "ETHUSDT" in text

    def test_get_signal_fields_present(self):
        """RetailPsychologySignal has all required fields."""
        df = make_flat_df(50000.0, 60)
        sig = agent.get_signal("BTCUSDT", df)
        assert hasattr(sig, "current_price")
        assert hasattr(sig, "stop_hunt")
        assert hasattr(sig, "retail_long_pct")
        assert hasattr(sig, "contrarian_bias")
        assert hasattr(sig, "liquidation_zones")
        assert hasattr(sig, "total_bonus")
        assert hasattr(sig, "summary")
