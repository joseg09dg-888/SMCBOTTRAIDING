"""Tests for MarketMicrostructureAgent."""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from agents.microstructure_agent import (
    MarketMicrostructureAgent,
    MicrostructureSignal,
    SessionWindow,
    PsychologicalLevel,
    SESSIONS,
)


def make_utc(year: int, month: int, day: int, hour: int, minute: int = 0, weekday_override: int = None) -> datetime:
    """Helper: create a timezone-aware UTC datetime."""
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return dt


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return MarketMicrostructureAgent()


# ── Session tests ──────────────────────────────────────────────────────────

def test_london_overlap_gives_max_bonus(agent):
    """12:00-16:00 UTC is the London-NY overlap with +15 bonus."""
    # Tuesday 14:00 UTC (overlap, Tuesday bonus +5 but session bonus is 15, total capped at 15)
    dt = datetime(2025, 1, 14, 14, 0, tzinfo=timezone.utc)  # Tuesday
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.current_session.name == "overlap_LN"
    assert signal.session_bonus == 15
    assert signal.total_bonus == 15  # capped at 15


def test_london_session_gives_10_bonus(agent):
    """08:00-12:00 UTC is London session with +10 bonus."""
    dt = datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc)  # Tuesday 10:00
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.current_session.name == "london"
    assert signal.session_bonus == 10


def test_dead_session_blocks_trade(agent):
    """21:00-23:59 UTC dead session: should_trade=False, total_bonus=0."""
    dt = datetime(2025, 1, 15, 22, 0, tzinfo=timezone.utc)  # Wednesday 22:00
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.should_trade is False
    assert signal.total_bonus == 0
    assert signal.current_session.name == "dead"


def test_friday_night_blocks_trade(agent):
    """Friday after 20:00 UTC must block trading regardless of session."""
    # Friday = weekday 4
    dt = datetime(2025, 1, 17, 20, 30, tzinfo=timezone.utc)  # Friday 20:30 UTC
    blocked, reason = agent.is_trade_blocked(dt)
    assert blocked is True
    assert "Friday" in reason

    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.should_trade is False


def test_not_blocked_on_friday_before_20(agent):
    """Friday before 20:00 UTC should not be blocked."""
    dt = datetime(2025, 1, 17, 14, 0, tzinfo=timezone.utc)  # Friday 14:00
    blocked, _ = agent.is_trade_blocked(dt)
    assert blocked is False


def test_not_blocked_on_thursday_night(agent):
    """Thursday 21:00 is dead session but not Friday night blocked."""
    dt = datetime(2025, 1, 16, 21, 30, tzinfo=timezone.utc)  # Thursday 21:30
    # Dead session blocks it via should_trade=False, but is_trade_blocked also returns True
    blocked, reason = agent.is_trade_blocked(dt)
    assert blocked is True  # dead session is also blocked
    assert "Dead session" in reason


# ── Psychological levels ───────────────────────────────────────────────────

def test_psychological_levels_btc_has_round_numbers(agent):
    """BTC psychological levels include round thousands."""
    levels = agent.get_psychological_levels("BTCUSDT", 50000)
    prices = [lvl.price for lvl in levels]
    assert 50000 in prices
    assert 60000 in prices
    assert 40000 in prices


def test_psychological_levels_distance_calculated(agent):
    """Distance percentage should be 0 when price is exactly at a level."""
    levels = agent.get_psychological_levels("BTCUSDT", 50000)
    exact = next((lvl for lvl in levels if lvl.price == 50000), None)
    assert exact is not None
    assert exact.distance_pct == pytest.approx(0.0, abs=0.001)
    assert exact.stop_hunt_likely is True  # within 0.5%


def test_psychological_levels_far_price_not_stop_hunt(agent):
    """A level far from current price should not flag stop_hunt_likely."""
    levels = agent.get_psychological_levels("BTCUSDT", 50000)
    far = next((lvl for lvl in levels if lvl.price == 100000), None)
    assert far is not None
    assert far.stop_hunt_likely is False


def test_unknown_symbol_returns_empty_levels(agent):
    """Unknown symbol has no predefined levels."""
    levels = agent.get_psychological_levels("XYZUSDT", 100)
    assert levels == []


# ── Stop hunt detection ────────────────────────────────────────────────────

def test_stop_hunt_detected_when_spike_and_return(agent):
    """Detects stop hunt when candle spiked past level and price returned."""
    # BTC level at 50000; candle high went to 50200 (0.4% above), current price 49800
    detected = agent.detect_stop_hunt(
        symbol="BTCUSDT",
        current_price=49800,
        prev_candle_high=50200,   # spiked 0.4% above 50000
        prev_candle_low=49500,
    )
    assert detected is True


def test_stop_hunt_detected_below_level(agent):
    """Detects stop hunt when candle spiked below a level and recovered."""
    # BTC level at 50000; candle low went to 49800 (0.4% below), current price 50200
    detected = agent.detect_stop_hunt(
        symbol="BTCUSDT",
        current_price=50200,
        prev_candle_high=50500,
        prev_candle_low=49800,   # spiked 0.4% below 50000
    )
    assert detected is True


def test_stop_hunt_not_detected_normal_move(agent):
    """Normal price movement not near a psychological level is not a stop hunt."""
    # Price at 48000, far from 50000 level, no spike
    detected = agent.detect_stop_hunt(
        symbol="BTCUSDT",
        current_price=48000,
        prev_candle_high=48200,
        prev_candle_low=47800,
    )
    assert detected is False


def test_stop_hunt_not_detected_small_spike(agent):
    """Spike < 0.3% threshold should not trigger stop hunt."""
    # BTC at 50001; candle high is 50100 (only 0.2% above 50000), price still above
    detected = agent.detect_stop_hunt(
        symbol="BTCUSDT",
        current_price=50050,
        prev_candle_high=50100,  # 0.2% above 50000, below 0.3% threshold
        prev_candle_low=49900,
    )
    assert detected is False


# ── Day of week ────────────────────────────────────────────────────────────

def test_tuesday_gives_day_bonus(agent):
    """Tuesday should give +5 day bonus."""
    dt = datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc)  # Tuesday
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.day_of_week == "Tuesday"
    assert signal.day_bonus == 5


def test_wednesday_gives_day_bonus(agent):
    """Wednesday should give +5 day bonus."""
    dt = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)  # Wednesday
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.day_of_week == "Wednesday"
    assert signal.day_bonus == 5


def test_thursday_gives_day_bonus(agent):
    """Thursday should give +5 day bonus."""
    dt = datetime(2025, 1, 16, 10, 0, tzinfo=timezone.utc)  # Thursday
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.day_of_week == "Thursday"
    assert signal.day_bonus == 5


def test_monday_gives_zero_day_bonus(agent):
    """Monday should give 0 day bonus."""
    dt = datetime(2025, 1, 13, 10, 0, tzinfo=timezone.utc)  # Monday
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.day_of_week == "Monday"
    assert signal.day_bonus == 0


def test_friday_gives_zero_day_bonus(agent):
    """Friday should give 0 day bonus."""
    dt = datetime(2025, 1, 17, 10, 0, tzinfo=timezone.utc)  # Friday 10:00 (not blocked)
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert signal.day_of_week == "Friday"
    assert signal.day_bonus == 0


# ── Full signal and integration ────────────────────────────────────────────

def test_get_signal_returns_microstructure_signal(agent):
    """get_signal returns a MicrostructureSignal with all required fields."""
    dt = datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc)
    signal = agent.get_signal("BTCUSDT", 50000, utc_now=dt)
    assert isinstance(signal, MicrostructureSignal)
    assert isinstance(signal.current_session, SessionWindow)
    assert isinstance(signal.day_of_week, str)
    assert isinstance(signal.psychological_levels, list)
    assert isinstance(signal.should_trade, bool)
    assert isinstance(signal.total_bonus, int)
    assert isinstance(signal.summary, str)


def test_score_adjustment_capped_at_15(agent):
    """Total bonus must never exceed 15."""
    # Tuesday + overlap_LN (12:00-16:00) would be 15+5=20 without cap
    dt = datetime(2025, 1, 14, 14, 0, tzinfo=timezone.utc)  # Tuesday 14:00 overlap
    score = agent.score_adjustment("BTCUSDT", 50000, utc_now=dt)
    assert score <= 15


def test_score_adjustment_zero_in_dead_session(agent):
    """score_adjustment returns 0 in dead session."""
    dt = datetime(2025, 1, 14, 22, 0, tzinfo=timezone.utc)  # 22:00 dead
    score = agent.score_adjustment("BTCUSDT", 50000, utc_now=dt)
    assert score == 0


def test_score_adjustment_non_negative(agent):
    """score_adjustment is always non-negative (0 to 15)."""
    dt = datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc)
    score = agent.score_adjustment("BTCUSDT", 50000, utc_now=dt)
    assert 0 <= score <= 15


# ── Telegram format ────────────────────────────────────────────────────────

def test_format_telegram_has_session_name(agent):
    """Telegram output must contain the current session name."""
    dt = datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc)  # London
    text = agent.format_telegram("BTCUSDT", 50000, utc_now=dt)
    assert "london" in text.lower()
    assert "BTCUSDT" in text


def test_format_telegram_has_bonus(agent):
    """Telegram output must mention the microstructure bonus."""
    dt = datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc)
    text = agent.format_telegram("BTCUSDT", 50000, utc_now=dt)
    assert "/15" in text


def test_format_telegram_blocked_message(agent):
    """Telegram output for dead session should indicate trade not allowed."""
    dt = datetime(2025, 1, 14, 22, 0, tzinfo=timezone.utc)  # dead session
    text = agent.format_telegram("BTCUSDT", 50000, utc_now=dt)
    assert "NO" in text or "BLOCKED" in text.upper() or "❌" in text


# ── Asia and pre-london sessions ────────────────────────────────────────────

def test_asia_session_detected(agent):
    """Hours 0-6 should be classified as Asia session."""
    dt = datetime(2025, 1, 14, 3, 0, tzinfo=timezone.utc)
    session = agent.get_current_session(dt)
    assert session.name == "asia"
    assert session.score_bonus == 5


def test_pre_london_session_detected(agent):
    """Hour 7 should be pre-london session."""
    dt = datetime(2025, 1, 14, 7, 30, tzinfo=timezone.utc)
    session = agent.get_current_session(dt)
    assert session.name == "pre_london"
    assert session.score_bonus == 8
