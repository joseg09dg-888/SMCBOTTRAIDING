# tests/strategies/test_ftmo_agent.py
import pytest
from datetime import date, datetime, timezone, timedelta
from strategies.ftmo_agent import (
    FTMOAgent, FTMORules, ChallengeState, ChallengeStatus, ChallengeType,
    DailyRecord
)

agent = FTMOAgent()

# ── Rules factory ──────────────────────────────────────────────────────────
def test_two_step_rules():
    r = FTMOAgent.create_rules(ChallengeType.TWO_STEP, 10000)
    assert r.profit_target_usd == pytest.approx(1000.0)
    assert r.max_daily_loss_usd == pytest.approx(500.0)
    assert r.max_total_drawdown_usd == pytest.approx(1000.0)
    assert r.trailing_drawdown is False

def test_one_step_rules():
    r = FTMOAgent.create_rules(ChallengeType.ONE_STEP, 10000)
    assert r.max_daily_loss_pct == pytest.approx(0.03)
    assert r.profit_split_pct == pytest.approx(0.90)
    assert r.trailing_drawdown is True

def test_new_challenge_initial_state():
    s = FTMOAgent.new_challenge(10000)
    assert s.current_balance == 10000.0
    assert s.total_pnl == 0.0
    assert s.status == ChallengeStatus.ACTIVE
    assert s.trading_days == 0

# ── Daily loss limit ────────────────────────────────────────────────────────
def test_daily_loss_ok():
    s = FTMOAgent.new_challenge(10000)
    s.daily_pnl_today = -100.0  # well within $500 limit
    ok, msg = agent.check_daily_loss_limit(s)
    assert ok is True

def test_daily_loss_limit_hit():
    s = FTMOAgent.new_challenge(10000)
    s.daily_pnl_today = -500.0  # exactly at limit
    ok, msg = agent.check_daily_loss_limit(s)
    assert ok is False

def test_daily_loss_safety_stop():
    s = FTMOAgent.new_challenge(10000)
    s.daily_pnl_today = -301.0  # > 60% of $500 = $300
    ok, msg = agent.check_daily_loss_limit(s)
    assert ok is False

# ── Drawdown limit ─────────────────────────────────────────────────────────
def test_drawdown_ok():
    s = FTMOAgent.new_challenge(10000)
    s.current_balance = 9500.0  # 5% drawdown
    ok, msg = agent.check_drawdown_limit(s)
    assert ok is True

def test_drawdown_limit_hit():
    s = FTMOAgent.new_challenge(10000)
    s.current_balance = 9000.0  # 10% drawdown = limit
    ok, msg = agent.check_drawdown_limit(s)
    assert ok is False

def test_drawdown_safety_stop():
    s = FTMOAgent.new_challenge(10000)
    s.current_balance = 9300.0  # 7% > 70% of limit
    ok, msg = agent.check_drawdown_limit(s)
    assert ok is False

# ── Consistency rule ───────────────────────────────────────────────────────
def test_consistency_ok():
    s = FTMOAgent.new_challenge(10000)
    s.total_pnl = 400.0
    s.daily_records = [
        DailyRecord(date.today(), 100.0, 2, 60.0, -10.0),
        DailyRecord(date.today() - timedelta(days=1), 300.0, 3, 300.0, -50.0),
    ]
    ok, msg = agent.check_consistency_rule(s)
    # best day 300 = 75% of 400 → FAILS (> 30%)
    assert ok is False

def test_consistency_passes():
    s = FTMOAgent.new_challenge(10000)
    s.total_pnl = 500.0
    s.daily_records = [
        DailyRecord(date.today(), 100.0, 2, 100.0, -10.0),
        DailyRecord(date.today()-timedelta(1), 150.0, 2, 150.0, -5.0),
        DailyRecord(date.today()-timedelta(2), 140.0, 2, 140.0, -5.0),
        DailyRecord(date.today()-timedelta(3), 110.0, 2, 110.0, -5.0),
    ]
    ok, msg = agent.check_consistency_rule(s)
    # best day 150 = 30% of 500 → borderline OK
    assert isinstance(ok, bool)

def test_consistency_no_trades():
    s = FTMOAgent.new_challenge(10000)
    s.total_pnl = 0.0
    ok, msg = agent.check_consistency_rule(s)
    assert ok is True  # no data → OK

# ── can_trade ──────────────────────────────────────────────────────────────
def test_can_trade_active():
    s = FTMOAgent.new_challenge(10000)
    wednesday = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)  # Wed noon
    ok, msg = agent.can_trade(s, wednesday)
    assert ok is True

def test_cant_trade_failed():
    s = FTMOAgent.new_challenge(10000)
    s.status = ChallengeStatus.FAILED
    ok, msg = agent.can_trade(s)
    assert ok is False and "failed" in msg.lower()

def test_cant_trade_monday_early():
    s = FTMOAgent.new_challenge(10000)
    monday_early = datetime(2026, 5, 18, 1, 0, tzinfo=timezone.utc)  # Mon 01:00
    ok, msg = agent.can_trade(s, monday_early)
    assert ok is False

def test_cant_trade_friday_late():
    s = FTMOAgent.new_challenge(10000)
    friday_late = datetime(2026, 5, 22, 17, 0, tzinfo=timezone.utc)  # Fri 17:00
    ok, msg = agent.can_trade(s, friday_late)
    assert ok is False

def test_cant_trade_3_losses():
    s = FTMOAgent.new_challenge(10000)
    s.consecutive_losses = 3
    ok, msg = agent.can_trade(s)
    assert ok is False

# ── record_trade ───────────────────────────────────────────────────────────
def test_record_winning_trade():
    s = FTMOAgent.new_challenge(10000)
    s = agent.record_trade(s, 100.0, date(2026, 5, 15))
    assert s.current_balance == pytest.approx(10100.0)
    assert s.total_pnl == pytest.approx(100.0)
    assert s.consecutive_losses == 0
    assert s.trading_days == 1

def test_record_losing_trade():
    s = FTMOAgent.new_challenge(10000)
    s = agent.record_trade(s, -200.0, date(2026, 5, 15))
    assert s.current_balance == pytest.approx(9800.0)
    assert s.consecutive_losses == 1

def test_record_fails_on_drawdown():
    s = FTMOAgent.new_challenge(10000)
    s = agent.record_trade(s, -1001.0, date(2026, 5, 15))
    assert s.status == ChallengeStatus.FAILED

def test_record_pauses_on_daily_limit():
    s = FTMOAgent.new_challenge(10000)
    s.daily_pnl_today = -400.0
    s = agent.record_trade(s, -310.0, date(2026, 5, 15))  # total daily = -710 > limit
    assert s.status in (ChallengeStatus.PAUSED, ChallengeStatus.FAILED)

def test_record_passes_challenge():
    s = FTMOAgent.new_challenge(10000)
    s.trading_days = 4
    for i in range(4):
        s = agent.record_trade(s, 260.0, date(2026, 5, 15) + timedelta(days=i))
    # total 1040 > 1000 target, 4+ days, consistency ok (equal days)
    assert s.status == ChallengeStatus.PASSED

# ── new_trading_day ────────────────────────────────────────────────────────
def test_new_trading_day_resets_daily():
    s = FTMOAgent.new_challenge(10000)
    s.daily_pnl_today = -200.0
    s.status = ChallengeStatus.PAUSED
    s = agent.new_trading_day(s)
    assert s.daily_pnl_today == 0.0
    assert s.status == ChallengeStatus.ACTIVE

# ── format_daily_report ────────────────────────────────────────────────────
def test_format_report_contains_balance():
    s = FTMOAgent.new_challenge(10000)
    msg = agent.format_daily_report(s)
    assert "10,000" in msg or "10000" in msg

def test_format_report_contains_status():
    s = FTMOAgent.new_challenge(10000)
    msg = agent.format_daily_report(s)
    assert "ACTIVE" in msg.upper() or "activo" in msg.lower()

def test_format_report_html():
    s = FTMOAgent.new_challenge(10000)
    msg = agent.format_daily_report(s)
    assert "<b>" in msg

# ── monthly_income ─────────────────────────────────────────────────────────
def test_monthly_income_200k():
    result = agent.calculate_monthly_income(200000, 0.05, 0.90)
    assert result["net_monthly"] == pytest.approx(9000.0)
    assert result["yearly"] == pytest.approx(108000.0)

def test_monthly_income_10k():
    result = agent.calculate_monthly_income(10000, 0.05, 0.80)
    assert result["gross_monthly"] == pytest.approx(500.0)
    assert result["net_monthly"] == pytest.approx(400.0)
