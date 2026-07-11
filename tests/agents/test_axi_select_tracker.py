import pytest
import os
import json
import tempfile
from unittest.mock import patch
from agents.axi_select_tracker import AxiSelectTracker, TrackResult


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """BUG-AXI-STATE-TORN-WRITE: record_day()/set_capital() persist to
    memory/axi_select_state.json. Without this isolation those writes
    land on the real production state file every time the suite runs."""
    monkeypatch.chdir(tmp_path)


def make_tracker(capital: float = 10_000.0, month: str = "2026-06") -> AxiSelectTracker:
    t = AxiSelectTracker()
    t._state = {
        "month": month,
        "records": [],
        "capital": capital,
        "initial_month_balance": capital,
    }
    return t


def test_empty_month_zero_pnl():
    t = make_tracker(10_000)
    r = t.get_status()
    assert r.monthly_pnl == pytest.approx(0.0)
    assert r.monthly_pct == pytest.approx(0.0)

def test_record_day_adds_record():
    t = make_tracker(10_000)
    with patch("agents.axi_select_tracker.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "2026-06-29"
        t.record_day(300.0)
    assert len(t._state["records"]) == 1
    assert t._state["records"][0]["pnl"] == pytest.approx(300.0)

def test_monthly_pct_calculation():
    t = make_tracker(10_000)
    t._state["records"] = [
        {"date": "2026-06-02", "pnl": 200, "pnl_pct": 0.02},
        {"date": "2026-06-03", "pnl": 300, "pnl_pct": 0.03},
    ]
    r = t.get_status()
    assert r.monthly_pnl == pytest.approx(500.0)
    assert r.monthly_pct == pytest.approx(0.05)

def test_on_track_when_above_5pct():
    t = make_tracker(10_000)
    t._state["records"] = [{"date": f"2026-06-{d:02d}", "pnl": 50, "pnl_pct": 0.005}
                            for d in range(1, 12)]
    r = t.get_status()
    assert r.on_track is True

def test_not_on_track_when_behind():
    t = make_tracker(10_000)
    t._state["records"] = [{"date": "2026-06-01", "pnl": 10, "pnl_pct": 0.001}]
    r = t.get_status()
    assert r.on_track is False

def test_days_remaining_correct():
    t = make_tracker(10_000)
    t._state["records"] = [{"date": f"2026-06-{d:02d}", "pnl": 10, "pnl_pct": 0.001}
                            for d in range(1, 6)]  # 5 days traded
    r = t.get_status()
    assert r.days_traded == 5
    assert r.days_remaining == 17  # 22 - 5

def test_daily_avg_needed():
    t = make_tracker(10_000)
    # 0% progress, 22 days remaining → need $500/month → ~$22.7/day
    r = t.get_status()
    assert r.daily_avg_needed == pytest.approx(500.0 / 22, rel=0.01)

def test_daily_avg_needed_partial():
    t = make_tracker(10_000)
    t._state["records"] = [{"date": "2026-06-01", "pnl": 250, "pnl_pct": 0.025}]
    r = t.get_status()
    # need $250 more in 21 days
    assert r.daily_avg_needed == pytest.approx(250.0 / 21, rel=0.01)

def test_stage_pre_seed():
    t = make_tracker(500)
    r = t.get_status()
    assert "PRE" in r.stage_name.upper()

def test_stage_incubation():
    t = make_tracker(50_500)
    r = t.get_status()
    assert "50K" in r.stage_name or "INCUBATION" in r.stage_name

def test_stage_pro_m():
    t = make_tracker(2_000_500)
    r = t.get_status()
    assert "2M" in r.stage_name or "PRO" in r.stage_name

def test_set_capital_resets_month():
    t = make_tracker(10_000, month="2026-05")
    with patch("agents.axi_select_tracker.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "2026-06"
        t.set_capital(50_000)
    assert t._state["capital"] == pytest.approx(50_000)

def test_format_telegram_contains_key_fields():
    t = make_tracker(50_000)
    t._state["records"] = [{"date": "2026-06-01", "pnl": 1000, "pnl_pct": 0.02}]
    msg = t.format_telegram()
    assert "AXI SELECT" in msg
    assert "50,000" in msg or "50000" in msg
    assert "%" in msg

def test_format_telegram_on_track_icon():
    t = make_tracker(10_000)
    # fully met 5%
    t._state["records"] = [{"date": f"2026-06-{d:02d}", "pnl": 50, "pnl_pct": 0.005}
                            for d in range(1, 12)]
    msg = t.format_telegram()
    assert "✅" in msg

def test_track_result_is_dataclass():
    t = make_tracker(10_000)
    r = t.get_status()
    assert isinstance(r, TrackResult)
    assert hasattr(r, "monthly_pnl")
    assert hasattr(r, "on_track")
