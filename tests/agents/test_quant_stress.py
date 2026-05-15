"""
tests/agents/test_quant_stress.py
TDD tests for agents/quant_stress.py — Historical Stress Testing
"""

import pytest
from agents.quant_stress import StressTester, StressScenario, StressResult, StressReport


def test_scenarios_list_count():
    assert len(StressTester.SCENARIOS) >= 8


def test_run_scenario_returns_result():
    tester = StressTester()
    scenario = StressTester.SCENARIOS[0]  # Black Monday
    result = tester.run_scenario(scenario, equity=1000.0)
    assert isinstance(result, StressResult)


def test_run_scenario_loss_positive():
    tester = StressTester()
    scenario = StressTester.SCENARIOS[0]
    result = tester.run_scenario(scenario, equity=1000.0)
    assert result.portfolio_loss_usd >= 0
    assert result.portfolio_loss_pct >= 0


def test_run_scenario_luna_very_high_loss():
    tester = StressTester()
    luna = next(s for s in StressTester.SCENARIOS if "Luna" in s.name)
    result = tester.run_scenario(luna, equity=1000.0)
    # Luna lost 99% — even with 10% in position, loss is notable
    assert result.portfolio_loss_pct > 0


def test_circuit_breaker_triggered():
    tester = StressTester()
    # Large crash -> circuit breaker
    scenario = StressScenario("test", "2026-01-01", -30.0, 1, 90, "all", 5.0)
    result = tester.run_scenario(scenario, equity=1000.0, open_positions=[{"size": 500}])
    assert result.circuit_breaker_triggered is True


def test_survived_small_crash():
    tester = StressTester()
    scenario = StressScenario("small", "2026-01-01", -2.0, 1, 5, "equity", 1.5)
    result = tester.run_scenario(scenario, equity=1000.0)
    assert result.survived is True


def test_risk_level_safe():
    tester = StressTester()
    scenario = StressScenario("tiny", "2026-01-01", -1.0, 1, 1, "equity", 1.0)
    result = tester.run_scenario(scenario, equity=10000.0)
    assert result.risk_level == "SAFE"


def test_risk_level_ruin():
    tester = StressTester()
    scenario = StressScenario("huge", "2026-01-01", -50.0, 1, 1, "crypto", 15.0)
    result = tester.run_scenario(scenario, equity=1000.0, open_positions=[{"size": 1000}])
    assert result.risk_level in ("DANGER", "RUIN")


def test_run_all_scenarios_returns_report():
    tester = StressTester()
    report = tester.run_all_scenarios(equity=10000.0)
    assert isinstance(report, StressReport)
    assert report.scenarios_run == len(StressTester.SCENARIOS)


def test_run_all_scenarios_counts():
    tester = StressTester()
    report = tester.run_all_scenarios(equity=10000.0)
    assert report.passed + report.failed == report.scenarios_run


def test_survival_rate_range():
    tester = StressTester()
    report = tester.run_all_scenarios(equity=10000.0)
    assert 0.0 <= report.survival_rate <= 1.0


def test_recommendations_not_empty():
    tester = StressTester()
    report = tester.run_all_scenarios(equity=1000.0)
    assert len(report.recommendations) > 0


def test_filter_by_asset_class():
    tester = StressTester()
    report = tester.run_all_scenarios(equity=10000.0, filter_asset_class="crypto")
    assert report.scenarios_run < len(StressTester.SCENARIOS)


def test_get_scenario_by_name():
    tester = StressTester()
    s = tester.get_scenario_by_name("luna")
    assert s is not None
    assert "Luna" in s.name or "luna" in s.name.lower()


def test_get_scenario_unknown():
    assert StressTester().get_scenario_by_name("doesnotexist") is None


def test_estimate_max_loss():
    pct = StressTester.estimate_max_loss_pct(-20.0, leverage=1.0, position_pct=0.1)
    assert pct == pytest.approx(0.02)


def test_estimate_max_loss_clamped():
    pct = StressTester.estimate_max_loss_pct(-200.0, leverage=5.0, position_pct=1.0)
    assert pct <= 1.0
