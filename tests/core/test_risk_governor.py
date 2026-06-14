# tests/core/test_risk_governor.py
import os
import json
import pytest
from datetime import datetime, timedelta, timezone
from core.risk_governor import RiskGovernor


SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "AUDUSD", "USDCAD", "NAS100", "US30"]


@pytest.fixture
def state_path(tmp_path):
    return str(tmp_path / "risk_governor_state.json")


def _wins(n):
    return [{"profit": 10.0} for _ in range(n)]


def _losses(n):
    return [{"profit": -10.0} for _ in range(n)]


class TestRiskGovernorBasics:
    def test_fresh_state_all_active_no_suspension(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path)
        assert gov.active_symbols() == SYMBOLS
        assert gov.risk_multiplier() == 1.0
        assert gov.suspended_symbols() == {}

    def test_initial_suspended_seeds_state(self, state_path):
        gov = RiskGovernor(
            SYMBOLS, state_path=state_path,
            initial_suspended={"USDJPY": "WR 6.2% historico", "GBPJPY": "WR 17.6% historico"},
        )
        assert "USDJPY" not in gov.active_symbols()
        assert "GBPJPY" not in gov.active_symbols()
        assert "EURUSD" in gov.active_symbols()
        assert os.path.exists(state_path)

    def test_state_persists_across_instances(self, state_path):
        gov1 = RiskGovernor(SYMBOLS, state_path=state_path, initial_suspended={"USDJPY": "bad"})
        gov2 = RiskGovernor(SYMBOLS, state_path=state_path)
        assert "USDJPY" not in gov2.active_symbols()


class TestSymbolSuspension:
    def test_low_win_rate_suspends_symbol(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25)
        symbol_deals = {"USDJPY": _wins(1) + _losses(7)}  # WR = 12.5%
        changes = gov.evaluate(symbol_deals, drawdown_pct=0.0)
        assert any(c["symbol"] == "USDJPY" for c in changes["suspended"])
        assert "USDJPY" not in gov.active_symbols()

    def test_decent_win_rate_stays_active(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25)
        symbol_deals = {"EURUSD": _wins(4) + _losses(4)}  # WR = 50%
        changes = gov.evaluate(symbol_deals, drawdown_pct=0.0)
        assert changes["suspended"] == []
        assert "EURUSD" in gov.active_symbols()

    def test_insufficient_samples_not_suspended(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25)
        symbol_deals = {"NAS100": _losses(3)}  # only 3 trades, all losses
        changes = gov.evaluate(symbol_deals, drawdown_pct=0.0)
        assert changes["suspended"] == []
        assert "NAS100" in gov.active_symbols()

    def test_only_last_min_trades_considered(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25)
        # 20 old losses then 8 recent wins -> rolling window should be the 8 wins
        symbol_deals = {"GBPUSD": _losses(20) + _wins(8)}
        changes = gov.evaluate(symbol_deals, drawdown_pct=0.0)
        assert changes["suspended"] == []
        assert "GBPUSD" in gov.active_symbols()

    def test_reactivation_after_cooldown(self, state_path):
        gov = RiskGovernor(
            SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25, cooldown_hours=1,
        )
        # Manually seed a suspension that's already 2 hours old
        gov._state["suspended"]["USDJPY"] = {
            "reason": "test",
            "since": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        }
        gov.save_state()
        changes = gov.evaluate({}, drawdown_pct=0.0)
        assert any(c["symbol"] == "USDJPY" for c in changes["reactivated"])
        assert "USDJPY" in gov.active_symbols()

    def test_not_yet_cooldown_stays_suspended(self, state_path):
        gov = RiskGovernor(
            SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25, cooldown_hours=168,
        )
        gov._state["suspended"]["USDJPY"] = {
            "reason": "test",
            "since": datetime.now(timezone.utc).isoformat(),
        }
        gov.save_state()
        changes = gov.evaluate({}, drawdown_pct=0.0)
        assert changes["reactivated"] == []
        assert "USDJPY" not in gov.active_symbols()


class TestRiskMultiplier:
    def test_low_drawdown_keeps_multiplier_at_one(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, dd_tiers=((0.04, 0.25), (0.02, 0.5)))
        changes = gov.evaluate({}, drawdown_pct=0.0176)  # current real DD
        assert changes["risk_multiplier"] is None
        assert gov.risk_multiplier() == 1.0

    def test_moderate_drawdown_cuts_multiplier_in_half(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, dd_tiers=((0.04, 0.25), (0.02, 0.5)))
        changes = gov.evaluate({}, drawdown_pct=0.025)
        assert changes["risk_multiplier"]["to"] == 0.5
        assert gov.risk_multiplier() == 0.5

    def test_severe_drawdown_cuts_multiplier_to_quarter(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, dd_tiers=((0.04, 0.25), (0.02, 0.5)))
        changes = gov.evaluate({}, drawdown_pct=0.05)
        assert changes["risk_multiplier"]["to"] == 0.25
        assert gov.risk_multiplier() == 0.25

    def test_recovery_steps_up_one_tier_at_a_time(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, dd_tiers=((0.04, 0.25), (0.02, 0.5)))
        gov.evaluate({}, drawdown_pct=0.05)   # -> 0.25
        assert gov.risk_multiplier() == 0.25
        gov.evaluate({}, drawdown_pct=0.0)    # recovered fully, but steps to 0.5 first
        assert gov.risk_multiplier() == 0.5
        gov.evaluate({}, drawdown_pct=0.0)    # next cycle steps to 1.0
        assert gov.risk_multiplier() == 1.0

    def test_immediate_cut_on_worsening_drawdown(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, dd_tiers=((0.04, 0.25), (0.02, 0.5)))
        gov.evaluate({}, drawdown_pct=0.0)
        gov.evaluate({}, drawdown_pct=0.06)   # straight to most severe tier, no gradual cut
        assert gov.risk_multiplier() == 0.25


class TestReporting:
    def test_status_line_includes_active_and_suspended(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, initial_suspended={"USDJPY": "bad"})
        line = gov.status_line()
        assert "USDJPY" in line
        assert "EURUSD" in line

    def test_format_report_mentions_new_suspensions(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25)
        changes = gov.evaluate({"GBPJPY": _losses(8)}, drawdown_pct=0.0)
        report = gov.format_report(changes, balance=98240.08, drawdown_pct=0.0176)
        assert "GBPJPY" in report
        assert "RISK GOVERNOR" in report

    def test_has_changes(self, state_path):
        gov = RiskGovernor(SYMBOLS, state_path=state_path, min_trades=8, suspend_wr=0.25)
        no_change = gov.evaluate({"EURUSD": _wins(8)}, drawdown_pct=0.0)
        assert RiskGovernor.has_changes(no_change) is False
        change = gov.evaluate({"GBPJPY": _losses(8)}, drawdown_pct=0.0)
        assert RiskGovernor.has_changes(change) is True


class TestFetchHelperShape:
    def test_module_exposes_fetch_function(self):
        from core.risk_governor import fetch_recent_deals_by_symbol
        assert callable(fetch_recent_deals_by_symbol)
