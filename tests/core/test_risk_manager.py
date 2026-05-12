import pytest
from core.risk_manager import RiskManager
from core.config import Config


@pytest.fixture
def rm():
    cfg = Config()
    cfg.max_risk_per_trade = 0.005
    cfg.max_daily_loss = 0.05
    cfg.max_monthly_loss = 0.15
    cfg.max_open_positions = 3
    return RiskManager(cfg, capital=1000.0)


def test_position_size_half_percent(rm):
    size = rm.calculate_position_size(entry=1.1000, stop_loss=1.0950, pip_value=0.0001)
    assert 0 < size <= 1.0, f"Size fuera de rango: {size}"


def test_blocks_trade_over_max_positions(rm):
    rm.open_positions = 3
    allowed, reason = rm.can_open_trade()
    assert not allowed
    assert "posiciones" in reason.lower()


def test_blocks_trade_over_daily_loss(rm):
    rm.daily_pnl = -51.0
    allowed, reason = rm.can_open_trade()
    assert not allowed
    assert "diaria" in reason.lower()


def test_blocks_trade_over_monthly_loss(rm):
    rm.monthly_pnl = -151.0
    allowed, reason = rm.can_open_trade()
    assert not allowed
    assert "mensual" in reason.lower()


def test_allows_trade_when_clear(rm):
    allowed, reason = rm.can_open_trade()
    assert allowed
    assert reason == "OK"


def test_mandatory_stop_loss(rm):
    with pytest.raises(ValueError, match="Stop Loss es obligatorio"):
        rm.validate_trade(entry=1.1000, stop_loss=None, take_profit=1.1100)


def test_rr_minimum_two(rm):
    result = rm.validate_trade(entry=1.1000, stop_loss=1.0950, take_profit=1.1100)
    assert result["risk_reward"] >= 2.0


def test_record_trade_updates_pnl(rm):
    rm.record_trade(pnl=10.0)
    assert rm.daily_pnl == 10.0
    assert rm.monthly_pnl == 10.0
