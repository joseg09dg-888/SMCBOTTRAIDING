"""Integration tests for shared context cross-agent communication."""
import pytest
from core.agent_memory import AgentMemoryManager


@pytest.fixture
def mem(tmp_path):
    m = AgentMemoryManager(
        db_path=str(tmp_path / "agent_memory.db"),
        agents_dir=str(tmp_path / "agents"),
        shared_ctx_path=str(tmp_path / "shared_context.json"),
        vector_dir=str(tmp_path / "vector_store"),
    )
    yield m
    m.close()


def test_fed_alert_visible_to_all_agents(mem):
    """FED agent writes alert → other agents can read it."""
    mem.broadcast_alert("fed_sentiment_agent", "FOMC in 24 hours")
    alerts = mem.get_shared_context("active_alerts", [])
    assert any("FOMC" in a["message"] for a in alerts)


def test_market_condition_cross_agent(mem):
    """Geopolitical agent sets condition → risk manager reads it."""
    mem.set_shared_context("market_condition", "high_risk")
    assert mem.get_shared_context("market_condition") == "high_risk"


def test_shared_context_persists_to_disk(tmp_path):
    """Write in one instance → read in new instance."""
    db = str(tmp_path / "agent_memory.db")
    agents = str(tmp_path / "agents")
    ctx = str(tmp_path / "shared_context.json")
    vec = str(tmp_path / "vector_store")

    m1 = AgentMemoryManager(db, agents, ctx, vec)
    m1.set_shared_context("test_key", "persisted_value")
    m1.close()

    m2 = AgentMemoryManager(db, agents, ctx, vec)
    assert m2.get_shared_context("test_key") == "persisted_value"
    m2.close()


def test_active_alerts_from_multiple_agents(mem):
    """Multiple agents can all broadcast alerts."""
    mem.broadcast_alert("onchain_agent", "Whale movement detected")
    mem.broadcast_alert("geopolitical_agent", "Conflict escalation")
    mem.broadcast_alert("fed_sentiment_agent", "Powell speech today")
    alerts = mem.get_shared_context("active_alerts", [])
    assert len(alerts) == 3
    sources = {a["from"] for a in alerts}
    assert "onchain_agent" in sources


def test_recent_avg_score_update(mem):
    mem.set_shared_context("recent_avg_score", 78)
    assert mem.get_shared_context("recent_avg_score") == 78


def test_best_agents_week_update(mem):
    mem.set_shared_context("best_agents_week", ["onchain_agent", "microstructure_agent"])
    agents = mem.get_shared_context("best_agents_week")
    assert "onchain_agent" in agents
