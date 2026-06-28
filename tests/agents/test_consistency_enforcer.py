import pytest
from agents.consistency_enforcer import ConsistencyEnforcer, EnforceResult


ce = ConsistencyEnforcer()


def test_ok_when_month_no_profit():
    r = ce.check(today_pnl=500, monthly_pnl=0)
    assert r.should_block_new is False
    assert r.is_conservative is False

def test_ok_when_day_below_25pct():
    r = ce.check(today_pnl=200, monthly_pnl=1_000)
    # 200/1000 = 20% — below 25% conservative threshold
    assert r.is_conservative is False
    assert r.should_block_new is False

def test_conservative_at_25pct():
    r = ce.check(today_pnl=300, monthly_pnl=1_000)
    # 300/1000 = 30% — exactly at block threshold, conservative triggers first
    # Actually 30% is block, 25% is conservative
    assert r.should_block_new is True  # 30% = block

def test_conservative_between_25_and_30():
    r = ce.check(today_pnl=260, monthly_pnl=1_000)
    # 260/1000 = 26% — conservative but not block
    assert r.is_conservative is True
    assert r.should_block_new is False

def test_block_at_30pct():
    r = ce.check(today_pnl=300, monthly_pnl=1_000)
    # 300/1000 = 30%
    assert r.should_block_new is True

def test_block_above_30pct():
    r = ce.check(today_pnl=400, monthly_pnl=1_000)
    assert r.should_block_new is True

def test_pct_calculation():
    r = ce.check(today_pnl=200, monthly_pnl=1_000)
    assert r.today_pct_of_monthly == pytest.approx(0.20)

def test_max_allowed_today_formula():
    # monthly_sin_hoy = 1000 - 200 = 800
    # max_allowed = 0.30 * 800 / 0.70 = 342.86
    r = ce.check(today_pnl=200, monthly_pnl=1_000)
    assert r.max_allowed_today == pytest.approx(0.30 * 800 / 0.70, rel=0.01)

def test_no_block_on_loss_day():
    r = ce.check(today_pnl=-200, monthly_pnl=1_000)
    assert r.should_block_new is False

def test_reason_ok():
    r = ce.check(today_pnl=100, monthly_pnl=1_000)
    assert "OK" in r.reason

def test_reason_conservative():
    r = ce.check(today_pnl=260, monthly_pnl=1_000)
    assert "AVISO" in r.reason or "conservador" in r.reason.lower()

def test_reason_block():
    r = ce.check(today_pnl=400, monthly_pnl=1_000)
    assert "BLOQUEO" in r.reason

def test_format_telegram_ok():
    r = ce.check(today_pnl=100, monthly_pnl=1_000)
    msg = ce.format_telegram(r)
    assert "CONSISTENCIA" in msg
    assert "🟢" in msg

def test_format_telegram_block():
    r = ce.check(today_pnl=400, monthly_pnl=1_000)
    msg = ce.format_telegram(r)
    assert "🔴" in msg

def test_format_telegram_warning():
    r = ce.check(today_pnl=260, monthly_pnl=1_000)
    msg = ce.format_telegram(r)
    assert "🟡" in msg

def test_enforce_result_is_dataclass():
    r = ce.check(100, 1_000)
    assert isinstance(r, EnforceResult)
    assert hasattr(r, "should_block_new")
    assert hasattr(r, "max_allowed_today")

def test_monthly_pnl_negative_no_block():
    r = ce.check(today_pnl=100, monthly_pnl=-500)
    assert r.should_block_new is False

def test_zero_today_always_ok():
    r = ce.check(today_pnl=0, monthly_pnl=1_000)
    assert r.should_block_new is False
    assert r.today_pct_of_monthly == pytest.approx(0.0)
