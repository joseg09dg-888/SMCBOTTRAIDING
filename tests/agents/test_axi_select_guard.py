import json
import pytest
from agents.axi_select_guard import AxiSelectGuard, GuardResult, STATE_FILE


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Every test runs in an empty tmp dir so AxiSelectGuard's disk
    persistence never reads/writes the real memory/axi_select_state.json."""
    monkeypatch.chdir(tmp_path)


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


# ── BUG-AXI-GUARD-RESTART regression tests ─────────────────────────────
# A pm2 restart mid-day used to reset day_start_balance to the current
# (post-loss) balance and silently clear paused_today, letting the bot
# burn through another -4% before the guard fired again.

def test_day_start_survives_process_restart():
    g1 = AxiSelectGuard()
    g1.set_day_start(10_000.0)

    g2 = AxiSelectGuard()  # simulates a fresh process after pm2 restart
    assert g2._day_start_balance == pytest.approx(10_000.0)
    assert g2._day_start_date == g1._day_start_date

def test_paused_today_survives_process_restart():
    g1 = AxiSelectGuard()
    g1.set_day_start(10_000.0)
    g1.check(9_500.0)  # -5% -> emergency close, should persist paused
    assert g1.paused_today is True

    g2 = AxiSelectGuard()  # fresh process
    assert g2.paused_today is True

def test_restart_does_not_reset_baseline_to_post_loss_balance():
    g1 = AxiSelectGuard()
    g1.set_day_start(10_000.0)
    g1.check(9_700.0)  # down 3% (warning, not yet closed)

    g2 = AxiSelectGuard()  # restart happens here at $9,700 balance
    g2.set_day_start(9_700.0)  # supervisor calls this every cycle
    r = g2.check(9_700.0)
    # baseline must still be the original $10,000, not the post-restart $9,700
    assert r.daily_pnl_usd == pytest.approx(-300.0)
    assert r.limit_usd == pytest.approx(-400.0)

def test_new_day_resets_paused_and_persists():
    g1 = AxiSelectGuard()
    g1._day_start_date = "2026-01-01"  # force "yesterday"
    g1._day_start_balance = 10_000.0
    g1._paused_today = True
    g1._save()

    g2 = AxiSelectGuard()
    assert g2.paused_today is True  # still yesterday's pause on load
    g2.set_day_start(9_000.0)  # today's first cycle -> new day detected
    assert g2.paused_today is False
    assert g2._day_start_balance == pytest.approx(9_000.0)

    g3 = AxiSelectGuard()  # confirm the reset itself was persisted
    assert g3.paused_today is False
    assert g3._day_start_balance == pytest.approx(9_000.0)

def test_persistence_coexists_with_tracker_state(tmp_path):
    import os
    os.makedirs("memory", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"month": "2026-07", "capital": 97_000.0, "records": []}, f)

    g = AxiSelectGuard()
    g.set_day_start(97_000.0)
    g.check(93_000.0)  # -4.1% -> emergency

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    assert state["capital"] == pytest.approx(97_000.0)  # tracker keys untouched
    assert state["guard_paused_today"] is True
