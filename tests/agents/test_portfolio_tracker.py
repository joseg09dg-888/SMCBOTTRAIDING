"""Tests for PortfolioTracker — Plan financiero 70-20-10."""
import json
import os
import tempfile
import pytest
from unittest.mock import patch


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    state_file = str(tmp_path / "portfolio_state.json")
    monkeypatch.setattr(
        "agents.portfolio_tracker.STATE_FILE", state_file
    )
    return state_file


def make_tracker(tmp_state):
    from agents.portfolio_tracker import PortfolioTracker
    return PortfolioTracker()


class TestPortfolioTrackerInit:
    def test_loads_default_state(self, tmp_state):
        t = make_tracker(tmp_state)
        s = t._state
        assert s["own_capital_total"] == 500.0
        assert s["bot_earnings_total"] == 0.0
        assert s["axi_funded_capital"] == 500.0

    def test_loads_existing_state(self, tmp_state):
        data = {
            "bot_earnings_total": 1000.0,
            "debts_paid": 500.0,
            "expenses_covered": 0.0,
            "bucket_70_invested": 0.0,
            "bucket_20_invested": 0.0,
            "bucket_10_capital": 600.0,
            "own_capital_total": 600.0,
            "axi_funded_capital": 500.0,
            "last_updated": "2026-01-01T00:00:00+00:00",
        }
        with open(tmp_state, "w") as f:
            json.dump(data, f)
        t = make_tracker(tmp_state)
        assert t._state["bot_earnings_total"] == 1000.0
        assert t._state["debts_paid"] == 500.0


class TestRecordEarnings:
    def test_first_payment_goes_to_debt(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(1000.0)
        assert t._state["debts_paid"] == 1000.0
        assert t._state["expenses_covered"] == 0.0
        assert t._state["bucket_70_invested"] == 0.0

    def test_second_payment_continues_debt(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(1000.0)
        t.record_earnings(500.0)
        assert t._state["debts_paid"] == 1500.0

    def test_overflow_to_expenses_after_debt(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(3000.0)
        assert t._state["debts_paid"] == 2000.0
        assert t._state["expenses_covered"] == 1000.0

    def test_full_debt_and_buffer_then_allocate(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(10000.0)
        assert t._state["debts_paid"] == 2000.0
        assert t._state["expenses_covered"] == 6000.0
        remaining = 2000.0
        assert abs(t._state["bucket_70_invested"] - remaining * 0.70) < 0.01
        assert abs(t._state["bucket_20_invested"] - remaining * 0.20) < 0.01

    def test_capital_propio_grows_by_10pct(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(10000.0)
        remaining = 2000.0
        expected = 500.0 + remaining * 0.10
        assert abs(t._state["own_capital_total"] - expected) < 0.01

    def test_axi_capital_updated(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(500.0, axi_capital=10500.0)
        assert t._state["axi_funded_capital"] == 10500.0

    def test_state_persisted_to_file(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(5000.0)
        with open(tmp_state) as f:
            saved = json.load(f)
        assert saved["bot_earnings_total"] == 5000.0

    def test_multiple_small_payments(self, tmp_state):
        t = make_tracker(tmp_state)
        for _ in range(10):
            t.record_earnings(500.0)
        assert t._state["debts_paid"] == 2000.0
        assert t._state["expenses_covered"] == 3000.0

    def test_no_bucket_allocation_before_debt_clear(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(1000.0)
        assert t._state["bucket_70_invested"] == 0.0
        assert t._state["bucket_20_invested"] == 0.0


class TestNextMilestone:
    def test_first_milestone_is_100k(self, tmp_state):
        t = make_tracker(tmp_state)
        target, label = t.get_next_milestone()
        assert target == 100_000

    def test_milestone_advances(self, tmp_state):
        t = make_tracker(tmp_state)
        t._state["own_capital_total"] = 150_000
        target, label = t.get_next_milestone()
        assert target == 500_000

    def test_million_milestone(self, tmp_state):
        t = make_tracker(tmp_state)
        t._state["own_capital_total"] = 600_000
        target, _ = t.get_next_milestone()
        assert target == 1_000_000

    def test_final_milestone_at_5m(self, tmp_state):
        t = make_tracker(tmp_state)
        t._state["own_capital_total"] = 2_000_000
        target, _ = t.get_next_milestone()
        assert target == 5_000_000


class TestFormatTelegram:
    def test_returns_string(self, tmp_state):
        t = make_tracker(tmp_state)
        msg = t.format_telegram()
        assert isinstance(msg, str)
        assert len(msg) > 50

    def test_contains_buckets(self, tmp_state):
        t = make_tracker(tmp_state)
        msg = t.format_telegram()
        assert "70%" in msg
        assert "20%" in msg
        assert "10%" in msg

    def test_shows_debt_pending(self, tmp_state):
        t = make_tracker(tmp_state)
        msg = t.format_telegram()
        assert "Deuda" in msg or "deuda" in msg.lower()

    def test_shows_axi_capital(self, tmp_state):
        t = make_tracker(tmp_state)
        msg = t.format_telegram()
        assert "AXI" in msg.upper()

    def test_projection_with_income(self, tmp_state):
        t = make_tracker(tmp_state)
        t._state["own_capital_total"] = 50_000
        msg = t.format_telegram(axi_monthly_income=5000.0)
        assert "meses" in msg

    def test_html_safe(self, tmp_state):
        t = make_tracker(tmp_state)
        msg = t.format_telegram()
        assert "<b>" in msg

    def test_action_now_changes_by_state(self, tmp_state):
        t = make_tracker(tmp_state)
        t.record_earnings(10000.0)
        msg = t.format_telegram()
        assert "AHORA" in msg
