"""
TDD tests for smc/liquidity_sweep.py -- ICT Silver Bullet all-or-nothing setup.
"""
from datetime import datetime, timezone
import pandas as pd
from smc.liquidity_sweep import (
    detect_sweep, check_setup, in_kill_zone, in_active_kill_zone, KILL_ZONES_UTC,
)


def _base_range_df(n=25, low=1.0, high=1.01):
    """Flat range -- no sweep, nothing to detect."""
    mid = (low + high) / 2
    return pd.DataFrame({
        "open": [mid] * n, "high": [high] * n, "low": [low] * n, "close": [mid] * n,
    })


def _df_with_bullish_sweep():
    """20 flat range bars, then a bar that pierces the low and closes back
    inside, then an up-displacement bar creating a bullish FVG."""
    df = _base_range_df(n=20, low=1.0000, high=1.0100)
    sweep_bar = pd.DataFrame({
        "open": [1.0050], "high": [1.0060], "low": [0.9950], "close": [1.0055],
    })
    fvg_candle1 = pd.DataFrame({
        "open": [1.0055], "high": [1.0100], "low": [1.0050], "close": [1.0095],
    })
    fvg_candle2 = pd.DataFrame({
        "open": [1.0150], "high": [1.0250], "low": [1.0140], "close": [1.0240],
    })  # displacement candle -- its low (1.0140) will be gap_high vs candle0 high
    fvg_candle3 = pd.DataFrame({
        "open": [1.0240], "high": [1.0300], "low": [1.0200], "close": [1.0280],
    })
    return pd.concat([df, sweep_bar, fvg_candle1, fvg_candle2, fvg_candle3], ignore_index=True)


def _df_no_sweep_no_fvg():
    return _base_range_df(n=25)


# ── detect_sweep ─────────────────────────────────────────────────────────

def test_detect_sweep_none_on_flat_range():
    assert detect_sweep(_df_no_sweep_no_fvg()) is None


def test_detect_sweep_none_insufficient_data():
    assert detect_sweep(_base_range_df(n=5), lookback=20) is None


def test_detect_bullish_sweep():
    df = _base_range_df(n=21, low=1.0000, high=1.0100)
    sweep_bar = pd.DataFrame({
        "open": [1.0050], "high": [1.0060], "low": [0.9950], "close": [1.0055],
    })
    df = pd.concat([df, sweep_bar], ignore_index=True)
    sweep = detect_sweep(df)
    assert sweep is not None
    assert sweep.direction == "bullish"
    assert sweep.swept_level == 1.0000


def test_detect_bearish_sweep():
    df = _base_range_df(n=21, low=1.0000, high=1.0100)
    sweep_bar = pd.DataFrame({
        "open": [1.0050], "high": [1.0150], "low": [1.0040], "close": [1.0045],
    })
    df = pd.concat([df, sweep_bar], ignore_index=True)
    sweep = detect_sweep(df)
    assert sweep is not None
    assert sweep.direction == "bearish"
    assert sweep.swept_level == 1.0100


# ── kill zones ───────────────────────────────────────────────────────────

def test_in_kill_zone_true_during_ny_am():
    dt = datetime(2026, 7, 9, 14, 30, tzinfo=timezone.utc)
    assert in_kill_zone(dt) is True
    assert in_active_kill_zone(dt) is True


def test_in_kill_zone_false_outside_windows():
    dt = datetime(2026, 7, 9, 10, 0, tzinfo=timezone.utc)
    assert in_kill_zone(dt) is False


def test_ny_pm_is_a_kill_zone_but_not_the_active_one():
    dt = datetime(2026, 7, 9, 18, 30, tzinfo=timezone.utc)
    assert in_kill_zone(dt) is True
    assert in_active_kill_zone(dt) is False


# ── check_setup (full confluence) ───────────────────────────────────────

def test_check_setup_none_without_sweep():
    result = check_setup(_df_no_sweep_no_fvg())
    assert result is None


def test_check_setup_full_confluence_in_kill_zone():
    df = _df_with_bullish_sweep()
    as_of = datetime(2026, 7, 9, 14, 30, tzinfo=timezone.utc)
    result = check_setup(df, as_of=as_of)
    assert result is not None
    assert result.direction == "bullish"
    assert result.valid is True
    assert result.in_kill_zone is True


def test_check_setup_valid_confluence_outside_kill_zone_is_invalid():
    df = _df_with_bullish_sweep()
    as_of = datetime(2026, 7, 9, 10, 0, tzinfo=timezone.utc)  # outside any kill zone
    result = check_setup(df, as_of=as_of)
    assert result is not None
    assert result.valid is False
    assert result.in_kill_zone is False


def test_check_setup_stop_loss_beyond_swept_level():
    df = _df_with_bullish_sweep()
    as_of = datetime(2026, 7, 9, 14, 30, tzinfo=timezone.utc)
    result = check_setup(df, as_of=as_of)
    assert result.stop_loss == 1.0000
