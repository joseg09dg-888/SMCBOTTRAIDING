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
