import pytest
from agents.axi_select_guard import AxiSelectGuard, GuardResult


def make_guard(day_start: float = 10_000.0) -> AxiSelectGuard:
    g = AxiSelectGuard()
    g._day_start_balance = day_start
    g._day_start_date = "2026-06-29"
    return g


def test_ok_when_no_loss():
    g = make_guard(10_000)
    r = g.check(10_050)
    assert r.should_close is False
    assert r.warning_level is False
    assert r.daily_pnl_usd == pytest.approx(50)

def test_warning_at_3pct():
    g = make_guard(10_000)
    r = g.check(9_700)   # -3%
    assert r.warning_level is True
    assert r.should_close is False

def test_emergency_at_4pct():
    g = make_guard(10_000)
    r = g.check(9_600)   # -4%
    assert r.should_close is True

def test_emergency_at_5pct():
    g = make_guard(10_000)
    r = g.check(9_400)   # -6% — also emergency
    assert r.should_close is True

def test_pct_calculation():
    g = make_guard(10_000)
    r = g.check(9_700)
    assert r.daily_pnl_pct == pytest.approx(-0.03)

def test_limit_usd_correct():
    g = make_guard(10_000)
    r = g.check(10_000)
    assert r.limit_usd == pytest.approx(-400.0)
    assert r.warning_usd == pytest.approx(-300.0)

def test_set_day_start_once_per_day():
    g = AxiSelectGuard()
    g.set_day_start(1_000)
    g.set_day_start(2_000)   # same day — should not override
    assert g._day_start_balance == pytest.approx(1_000)

def test_with_axi_capital_override():
    g = make_guard(10_000)
    # day start was 10K but Axi assigned 50K capital
    r = g.check(9_900, capital_assigned=50_000)
    # loss is $100 vs $50K capital = 0.2% — OK
    assert r.should_close is False
    assert r.limit_usd == pytest.approx(-2_000)  # 4% of $50K

def test_reason_contains_pct():
    g = make_guard(10_000)
    r = g.check(9_600)
    assert "4" in r.reason or "EMERGENCY" in r.reason

def test_format_telegram_ok():
    g = make_guard(10_000)
    r = g.check(10_050)
    msg = g.format_telegram(r)
    assert "AXI GUARD" in msg
    assert "OK" in msg

def test_format_telegram_emergency():
    g = make_guard(10_000)
    r = g.check(9_400)
    msg = g.format_telegram(r)
    assert "EMERGENCY" in msg or "🚨" in msg

def test_format_telegram_warning():
    g = make_guard(10_000)
    r = g.check(9_700)
    msg = g.format_telegram(r)
    assert "WARNING" in msg or "⚠️" in msg

def test_guard_result_fields():
    g = make_guard(50_000)
    r = g.check(48_000)
    assert isinstance(r, GuardResult)
    assert r.capital == pytest.approx(50_000)

def test_no_false_emergency_on_profit():
    g = make_guard(10_000)
    r = g.check(15_000)   # +50% profit
    assert r.should_close is False
    assert r.warning_level is False

def test_exactly_at_warning_boundary():
    g = make_guard(10_000)
    r = g.check(9_700.01)  # just above -3%
    assert r.warning_level is False

def test_exactly_at_limit_boundary():
    g = make_guard(10_000)
    r = g.check(9_600.0)   # exactly -4%
    assert r.should_close is True
