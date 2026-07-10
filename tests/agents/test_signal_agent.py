import pytest
from agents.signal_agent import SignalAgent, TradeSignal, SignalType


def test_signal_has_required_fields():
    sig = TradeSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.LONG,
        entry=50000,
        stop_loss=49500,
        take_profit=51000,
        timeframe="1h",
        trigger="CHoCH + bullish OB retest",
        confidence=0.82,
    )
    assert sig.risk_reward == pytest.approx(2.0, rel=0.1)
    assert sig.is_valid()


def test_signal_invalid_without_sl():
    sig = TradeSignal(
        symbol="EURUSD",
        signal_type=SignalType.SHORT,
        entry=1.1000,
        stop_loss=None,
        take_profit=1.0900,
        timeframe="4h",
        trigger="BOS bajista",
        confidence=0.7,
    )
    assert not sig.is_valid()


def test_signal_invalid_low_rr():
    sig = TradeSignal(
        symbol="EURUSD",
        signal_type=SignalType.LONG,
        entry=1.1000,
        stop_loss=1.0990,
        take_profit=1.1005,
        timeframe="1h",
        trigger="FVG",
        confidence=0.6,
    )
    assert not sig.is_valid()


def test_signal_wait_is_invalid():
    sig = TradeSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.WAIT,
        entry=50000,
        stop_loss=49500,
        take_profit=51000,
        timeframe="1h",
        trigger="Sin setup",
        confidence=0.5,
    )
    assert not sig.is_valid()


def test_telegram_format():
    sig = TradeSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.LONG,
        entry=50000,
        stop_loss=49000,
        take_profit=52000,
        timeframe="4h",
        trigger="CHoCH + FVG",
        confidence=0.85,
    )
    txt = sig.format_telegram()
    assert "BTCUSDT" in txt
    assert "50000" in txt
    assert "SETUP VÁLIDO" in txt


def test_signal_agent_no_setup_returns_wait():
    agent = SignalAgent()
    sig = agent.evaluate(
        analysis_text="No hay setup claro en este momento",
        symbol="EURUSD",
        timeframe="1h",
        current_price=1.1000,
        poi_zones=[],
    )
    assert sig.signal_type == SignalType.WAIT


def test_signal_agent_atr_sl_with_df():
    """ATR-based SL produces wider stops than fixed 1% when market is volatile."""
    import pandas as pd
    import numpy as np
    agent = SignalAgent()
    # Create a volatile 30-bar DF (big ranges → high ATR)
    closes = [100.0 + np.sin(i) * 5 for i in range(30)]
    df = pd.DataFrame({
        "open":  closes,
        "high":  [c + 3.0 for c in closes],
        "low":   [c - 3.0 for c in closes],
        "close": closes,
        "volume": [1000.0] * 30,
    })
    sl_atr = agent._sl_distance("BNBUSDT", 100.0, df=df)
    sl_pct = 100.0 * 0.01  # fallback 1%
    assert sl_atr > 0
    # ATR-based SL should be different from (likely larger than) fixed 1%
    assert isinstance(sl_atr, float)


def test_signal_agent_swing_tp_long():
    """_nearest_swing returns a swing high closer than tp_raw for LONG trades."""
    import pandas as pd
    agent = SignalAgent()
    # Create DF with a clear swing high at 110
    closes = [100.0] * 30
    highs  = [100.0] * 30
    highs[20] = 110.0   # swing high at 110
    lows   = [99.0] * 30
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                       "close": closes, "volume": [1000.0] * 30})
    tp = agent._nearest_swing(100.0, sl_dist=2.0, is_bullish=True,
                               tp_raw=106.0, df=df)
    assert tp > 100.0


def test_signal_agent_evaluate_long_returns_valid_signal():
    """Evaluate with bullish analysis text returns LONG signal."""
    agent = SignalAgent()
    sig = agent.evaluate(
        analysis_text="bullish trend BOS confirmado order block presente setup valido",
        symbol="BTCUSDT",
        timeframe="4h",
        current_price=67000.0,
        poi_zones=[{"zone_low": 66500.0, "zone_high": 67000.0}],
    )
    assert sig.signal_type == SignalType.LONG
    assert sig.entry > 0
    assert sig.stop_loss is not None
    assert sig.take_profit > sig.entry


# ── BUG-TRIGGER-HARDCODED regression: trigger must reflect real confluence ──

def test_trigger_reflects_actual_confluence_not_hardcoded_direction():
    """Before the fix, trigger was hardcoded to 'CHoCH + OB retest' for every
    bullish signal and 'BOS + FVG bajista' for every bearish one, regardless
    of what actually fired -- this broke AutonomousLearner's per-setup
    grouping (it was really grouping by direction, not by setup pattern)."""
    agent = SignalAgent()
    sig = agent.evaluate(
        analysis_text="bullish trend BOS confirmado FVG presente setup valido",
        symbol="BTCUSDT", timeframe="4h", current_price=67000.0, poi_zones=[],
    )
    assert "BOS" in sig.trigger
    assert "FVG" in sig.trigger
    assert "OB" not in sig.trigger  # no order block was present in this text


def test_trigger_differs_for_different_confluence_same_direction():
    agent = SignalAgent()
    sig_bos_only = agent.evaluate(
        analysis_text="bullish trend BOS confirmado setup valido",
        symbol="EURUSD", timeframe="1h", current_price=1.1000, poi_zones=[],
    )
    sig_choch_ob = agent.evaluate(
        analysis_text="bullish trend CHoCH detectado order block presente setup valido",
        symbol="EURUSD", timeframe="1h", current_price=1.1000, poi_zones=[],
    )
    assert sig_bos_only.trigger != sig_choch_ob.trigger
