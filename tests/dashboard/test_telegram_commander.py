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
    result = commander.handle_command("/status")
    assert result.success is True
    msg = result.message
    assert "1,047" in msg or "1047" in msg or "47.50" in msg  # balance or P&L
    assert "2" in msg  # open positions


def test_handle_scores_command(commander):
    result = commander.handle_command("/scores")
    assert result.success is True
    assert "87" in result.message
    assert "91" in result.message


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
    commander.state.mode = BotMode.AUTO
    result = commander.handle_command("/status")
    assert "AUTO" in result.message.upper()
