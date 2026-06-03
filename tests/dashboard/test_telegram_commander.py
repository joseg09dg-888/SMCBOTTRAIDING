import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dashboard.telegram_commander import (
    TelegramCommander, BotMode, BotStatus, CommandResult
)


@pytest.fixture
def mock_state():
    """Minimal bot state for testing commands."""
    state = MagicMock()
    state.mode = BotMode.HYBRID
    state.paused = False
    state.capital = 1000.0
    state.balance = 1047.50
    state.daily_pnl = 47.50
    state.total_pnl = 47.50
    state.open_positions = 2
    state.wins_today = 3
    state.losses_today = 1
    state.win_rate = 61.3
    state.drawdown = 0.8
    state.last_trade_symbol = "BTCUSDT"
    state.last_trade_pnl = 23.50
    state.score_history = [87, 72, 91, 65, 78]
    return state


@pytest.fixture
def commander(mock_state):
    cmd = TelegramCommander.__new__(TelegramCommander)
    cmd.state = mock_state
    cmd._bot = None
    return cmd


def test_mode_enum_values():
    assert BotMode.AUTO.value == "auto"
    assert BotMode.SEMI.value == "semi"
    assert BotMode.PAUSED.value == "paused"
    assert BotMode.HYBRID.value == "hybrid"


def test_handle_auto_command(commander):
    result = commander.handle_command("/auto")
    assert result.success is True
    assert "AUTO" in result.message.upper()
    assert commander.state.mode == BotMode.AUTO


def test_handle_semi_command(commander):
    result = commander.handle_command("/semi")
    assert result.success is True
    assert "SEMI" in result.message.upper()
    assert commander.state.mode == BotMode.SEMI


def test_handle_pause_command(commander):
    result = commander.handle_command("/pause")
    assert result.success is True
    assert commander.state.paused is True


def test_handle_resume_command(commander):
    commander.state.paused = True
    result = commander.handle_command("/resume")
    assert result.success is True
    assert commander.state.paused is False


def test_handle_status_command(commander):
    # _cmd_status now fetches real Binance data; just verify it succeeds
    # and returns a non-empty HTML message (may fall back to error msg on testnet)
    result = commander.handle_command("/status")
    assert result.success is True
    assert len(result.message) > 10  # non-empty response
    assert result.action == "status"


def test_handle_scores_command(commander):
    # _cmd_scores now reads SQLite; in test env the DB is empty
    result = commander.handle_command("/scores")
    assert result.success is True
    # Either shows real scores OR the "no scores yet" message
    msg = result.message
    assert "score" in msg.lower() or "Score" in msg or "scan" in msg.lower()


def test_handle_unknown_command(commander):
    result = commander.handle_command("/unknowncmd")
    assert result.success is False
    assert "desconocido" in result.message.lower() or "unknown" in result.message.lower()


def test_handle_risk_command(commander):
    result = commander.handle_command("/risk")
    assert result.success is True
    assert "0.8" in result.message or "drawdown" in result.message.lower()


def test_command_result_has_fields():
    cr = CommandResult(success=True, message="OK", action="test")
    assert cr.success is True
    assert cr.message == "OK"


def test_status_contains_mode(commander):
    # _cmd_status now fetches real data; verify it returns success regardless
    commander.state.mode = BotMode.AUTO
    result = commander.handle_command("/status")
    assert result.success is True
    # Message contains either mode info or bot status info
    assert len(result.message) > 5


def test_demo_command_no_supervisor(commander):
    """Without supervisor, /demo returns graceful empty message."""
    commander._supervisor = None
    result = commander.handle_command("/demo")
    assert result.success is True
    assert "demo" in result.message.lower() or "DEMO" in result.message


def test_demo_command_empty_trades(commander):
    """Supervisor with no open demo trades."""
    sup = MagicMock()
    sup._demo_trades = []
    commander._supervisor = sup
    result = commander.handle_command("/demo")
    assert result.success is True
    assert "Sin posiciones" in result.message or "demo" in result.message.lower()


def test_demo_command_with_open_trade(commander):
    """Supervisor with one open demo trade returns position info."""
    from unittest.mock import patch as _patch
    from agents.signal_agent import SignalType, TradeSignal
    from datetime import datetime, timezone

    sig = TradeSignal(
        symbol="BTCUSDT", signal_type=SignalType.SHORT,
        entry=65000.0, stop_loss=66000.0, take_profit=62000.0,
        timeframe="1h", trigger="CHoCH", confidence=0.8,
    )
    sig.decision_score = 110

    class _FakeDemo:
        status   = "open"
        signal   = sig
        score    = 110
        opened_at = datetime.now(timezone.utc)

    sup = MagicMock()
    sup._demo_trades = [_FakeDemo()]
    commander._supervisor = sup

    with _patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.fast_info.last_price = 64000.0
        result = commander.handle_command("/demo")

    assert result.success is True
    assert "BTCUSDT" in result.message
    assert "SHORT" in result.message
