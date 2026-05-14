# tests/core/test_agent_health_check.py
"""
TDD tests for core.agent_health_check
Written BEFORE implementation — these must fail first, then pass.
"""
import pytest
from datetime import datetime, timezone

from core.agent_health_check import (
    AgentStatus,
    HealthReport,
    AgentHealthCheck,
    AGENT_REGISTRY,
)


# ─── HealthReport property tests ──────────────────────────────────────────────

def _make_status(name: str, healthy: bool) -> AgentStatus:
    return AgentStatus(
        name=name,
        module="agents.signal_agent",
        class_name="SignalAgent",
        is_initialized=healthy,
        has_valid_output=healthy,
        signals_generated=0,
        accuracy=None,
        last_check=datetime.now(timezone.utc),
        error=None if healthy else "ImportError: no module",
    )


def test_health_report_all_healthy_property():
    """HealthReport with 21/21 healthy → all_healthy = True"""
    statuses = [_make_status(f"Agent{i}", True) for i in range(21)]
    report = HealthReport(
        timestamp=datetime.now(timezone.utc),
        total_agents=21,
        healthy_agents=21,
        failed_agents=0,
        statuses=statuses,
    )
    assert report.all_healthy is True


def test_health_report_not_all_healthy():
    """HealthReport with 20/21 healthy → all_healthy = False"""
    statuses = [_make_status(f"Agent{i}", i < 20) for i in range(21)]
    report = HealthReport(
        timestamp=datetime.now(timezone.utc),
        total_agents=21,
        healthy_agents=20,
        failed_agents=1,
        statuses=statuses,
    )
    assert report.all_healthy is False


def test_health_report_failed_count():
    """failed_agents == total_agents - healthy_agents"""
    statuses = [_make_status(f"Agent{i}", i < 18) for i in range(21)]
    report = HealthReport(
        timestamp=datetime.now(timezone.utc),
        total_agents=21,
        healthy_agents=18,
        failed_agents=3,
        statuses=statuses,
    )
    assert report.failed_agents == report.total_agents - report.healthy_agents


# ─── check_agent tests ─────────────────────────────────────────────────────────

def test_check_agent_valid_module():
    """check_agent with agents.lunar_agent / LunarCycleAgent → is_initialized=True"""
    checker = AgentHealthCheck()
    status = checker.check_agent({
        "name": "Lunar Agent",
        "module": "agents.lunar_agent",
        "class": "LunarCycleAgent",
    })
    assert status.is_initialized is True
    assert status.error is None


def test_check_agent_invalid_module():
    """check_agent with non-existent module → is_initialized=False, error not None"""
    checker = AgentHealthCheck()
    status = checker.check_agent({
        "name": "Fake Agent",
        "module": "agents.nonexistent_totally_fake",
        "class": "FakeClass",
    })
    assert status.is_initialized is False
    assert status.error is not None
    assert len(status.error) > 0


def test_check_agent_invalid_class():
    """Valid module but non-existent class → is_initialized=False"""
    checker = AgentHealthCheck()
    status = checker.check_agent({
        "name": "Bad Class Agent",
        "module": "agents.lunar_agent",
        "class": "NonExistentClass999",
    })
    assert status.is_initialized is False
    assert status.error is not None


# ─── run_full_check tests ──────────────────────────────────────────────────────

def test_run_full_check_returns_report():
    """run_full_check() returns HealthReport with total_agents == len(AGENT_REGISTRY)"""
    checker = AgentHealthCheck()
    report = checker.run_full_check()
    assert isinstance(report, HealthReport)
    assert report.total_agents == len(AGENT_REGISTRY)


def test_run_full_check_counts_healthy():
    """healthy_agents + failed_agents == total_agents"""
    checker = AgentHealthCheck()
    report = checker.run_full_check()
    assert report.healthy_agents + report.failed_agents == report.total_agents


def test_run_full_check_has_all_statuses():
    """report.statuses has the same count as AGENT_REGISTRY"""
    checker = AgentHealthCheck()
    report = checker.run_full_check()
    assert len(report.statuses) == len(AGENT_REGISTRY)


# ─── format_telegram tests ─────────────────────────────────────────────────────

def test_format_telegram_contains_health_check():
    """format_telegram() contains 'HEALTH CHECK'"""
    statuses = [_make_status(f"Agent{i}", True) for i in range(21)]
    report = HealthReport(
        timestamp=datetime.now(timezone.utc),
        total_agents=21,
        healthy_agents=21,
        failed_agents=0,
        statuses=statuses,
    )
    text = report.format_telegram()
    assert "HEALTH CHECK" in text


def test_format_telegram_contains_all_agents():
    """format_telegram() shows at least 20 agent lines"""
    statuses = [_make_status(f"Agent{i}", True) for i in range(21)]
    report = HealthReport(
        timestamp=datetime.now(timezone.utc),
        total_agents=21,
        healthy_agents=21,
        failed_agents=0,
        statuses=statuses,
    )
    text = report.format_telegram()
    # Count lines containing ✅ or ❌ (agent status lines)
    agent_lines = [line for line in text.splitlines() if "✅" in line or "❌" in line]
    assert len(agent_lines) >= 20


def test_format_telegram_shows_summary_line():
    """format_telegram() contains '/21 agentes' in the summary section"""
    statuses = [_make_status(f"Agent{i}", True) for i in range(21)]
    report = HealthReport(
        timestamp=datetime.now(timezone.utc),
        total_agents=21,
        healthy_agents=21,
        failed_agents=0,
        statuses=statuses,
    )
    text = report.format_telegram()
    assert "/21 agentes" in text


# ─── format_short_status tests ────────────────────────────────────────────────

def test_format_short_status_ok():
    """AgentStatus with is_initialized=True → '✅' in result"""
    checker = AgentHealthCheck()
    status = _make_status("Signal Agent", True)
    result = checker.format_short_status(status)
    assert "✅" in result


def test_format_short_status_error():
    """AgentStatus with is_initialized=False → '❌' in result"""
    checker = AgentHealthCheck()
    status = _make_status("Bad Agent", False)
    result = checker.format_short_status(status)
    assert "❌" in result


# ─── AGENT_REGISTRY tests ─────────────────────────────────────────────────────

def test_agent_registry_has_21_entries():
    """AGENT_REGISTRY has exactly 21 entries"""
    assert len(AGENT_REGISTRY) == 21


def test_agent_registry_all_have_required_keys():
    """Every entry in AGENT_REGISTRY has 'name', 'module', 'class' keys"""
    for entry in AGENT_REGISTRY:
        assert "name" in entry, f"Missing 'name' in {entry}"
        assert "module" in entry, f"Missing 'module' in {entry}"
        assert "class" in entry, f"Missing 'class' in {entry}"
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["module"], str) and entry["module"]
        assert isinstance(entry["class"], str) and entry["class"]
