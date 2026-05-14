"""Tests for AgentMemoryManager — all in-memory, no file I/O side effects."""
import json
import pytest
from unittest.mock import patch
from core.agent_memory import AgentMemoryManager, AGENT_NAMES


@pytest.fixture
def mem(tmp_path):
    """Fresh AgentMemoryManager backed by tmp dirs."""
    m = AgentMemoryManager(
        db_path=str(tmp_path / "agent_memory.db"),
        agents_dir=str(tmp_path / "agents"),
        shared_ctx_path=str(tmp_path / "shared_context.json"),
        vector_dir=str(tmp_path / "vector_store"),
    )
    yield m
    m.close()


# ── Layer 1: Short-term ────────────────────────────────────────────────────────

def test_set_and_get_short_term(mem):
    mem.set_short_term("lunar_agent", "last_phase", "llena")
    assert mem.get_short_term("lunar_agent", "last_phase") == "llena"


def test_get_short_term_default(mem):
    assert mem.get_short_term("smc_agent", "missing_key", "default") == "default"


def test_clear_short_term_single_agent(mem):
    mem.set_short_term("smc_agent", "foo", "bar")
    mem.clear_short_term("smc_agent")
    assert mem.get_short_term("smc_agent", "foo") is None


def test_clear_all_short_term(mem):
    mem.set_short_term("lunar_agent", "x", 1)
    mem.set_short_term("smc_agent", "y", 2)
    mem.clear_short_term()
    assert mem.get_short_term("lunar_agent", "x") is None
    assert mem.get_short_term("smc_agent", "y") is None


# ── Layer 2: SQLite ────────────────────────────────────────────────────────────

def test_record_and_count_signal(mem):
    mem.record_signal("lunar_agent", "BTCUSDT", "entry", "bullish", 75)
    stats = mem.get_agent_stats("lunar_agent")
    assert stats["total_signals_db"] == 1


def test_record_trade_stored(mem):
    mem.record_trade("BTCUSDT", 60000, 62000, 200.0, ["smc_agent", "lunar_agent"], True)
    trades = mem.get_recent_trades(days=7)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTCUSDT"
    assert trades[0]["won"] == 1


def test_record_losing_trade(mem):
    mem.record_trade("ETHUSDT", 3000, 2900, -50.0, ["smc_agent"], False)
    trades = mem.get_recent_trades(days=7)
    assert trades[0]["won"] == 0
    assert trades[0]["pnl"] == -50.0


def test_record_decision(mem):
    mem.record_decision("fed_sentiment_agent", "XAUUSD", "bullish", "dovish FED", 10)
    stats = mem.get_agent_stats("fed_sentiment_agent")
    assert stats["avg_score_contribution"] == 10.0


def test_record_pattern(mem):
    mem.record_pattern("BOS_breakout", "BTCUSDT", "1h", won=True)
    mem.record_pattern("BOS_breakout", "BTCUSDT", "1h", won=True)
    mem.record_pattern("BOS_breakout", "BTCUSDT", "1h", won=False)
    patterns = mem.get_pattern_stats("BOS_breakout")
    assert patterns[0]["wins"] == 2
    assert patterns[0]["losses"] == 1


def test_multiple_signals_same_agent(mem):
    for i in range(5):
        mem.record_signal("elliott_agent", "BTCUSDT", "entry", "bullish", 70 + i)
    stats = mem.get_agent_stats("elliott_agent")
    assert stats["total_signals_db"] == 5


# ── Layer 3: Vector store ──────────────────────────────────────────────────────

def test_store_and_search_knowledge(mem):
    mem.store_knowledge("smc_agent", "Order blocks at 60000 acted as strong support")
    results = mem.search_knowledge("order blocks")
    assert len(results) >= 1
    assert any("60000" in r["text"] or "order" in r["text"].lower() for r in results)


def test_knowledge_count_increases(mem):
    initial = mem.get_knowledge_count()
    mem.store_knowledge("lunar_agent", "Full moon correlated with BTC rally in 2023")
    assert mem.get_knowledge_count() >= initial + 1


def test_search_no_results_empty_list(mem):
    results = mem.search_knowledge("xyzzy_nonexistent_query_12345")
    assert isinstance(results, list)


# ── Layer 4: Per-agent JSON ────────────────────────────────────────────────────

def test_all_21_agents_initialized(mem):
    stats = mem.get_all_agent_stats()
    assert len(stats) == 21
    for name in AGENT_NAMES:
        assert name in stats


def test_update_agent_correct_signal(mem):
    mem.update_agent_data("chaos_agent", signal_correct=True)
    assert mem.get_agent_accuracy("chaos_agent") == 1.0


def test_update_agent_wrong_signal(mem):
    mem.update_agent_data("chaos_agent", signal_correct=True)
    mem.update_agent_data("chaos_agent", signal_correct=False)
    assert mem.get_agent_accuracy("chaos_agent") == 0.5


def test_accuracy_starts_at_zero(mem):
    assert mem.get_agent_accuracy("institutional_flow_agent") == 0.0


def test_update_agent_weight(mem):
    mem.update_agent_weight("lunar_agent", 1.5)
    assert mem.get_agent_weight("lunar_agent") == 1.5


def test_weight_clamped_max(mem):
    mem.update_agent_weight("lunar_agent", 99.9)
    assert mem.get_agent_weight("lunar_agent") == 3.0


def test_weight_clamped_min(mem):
    mem.update_agent_weight("lunar_agent", -5.0)
    assert mem.get_agent_weight("lunar_agent") == 0.0


def test_add_lesson_stored(mem):
    mem.add_lesson("microstructure_agent", "Stop hunts work best 08:00-10:00 UTC")
    data = mem._agent_data["microstructure_agent"]
    lessons = data["lessons_learned"]
    assert len(lessons) == 1
    assert "08:00" in lessons[0]["lesson"]


def test_lessons_capped_at_20(mem):
    for i in range(25):
        mem.add_lesson("smc_agent", f"Lesson {i}")
    assert len(mem._agent_data["smc_agent"]["lessons_learned"]) == 20


def test_best_conditions_tracked(mem):
    mem.update_agent_data("onchain_agent", True, condition="extreme_fear")
    data = mem._agent_data["onchain_agent"]
    assert "extreme_fear" in data["best_conditions"]


def test_worst_conditions_tracked(mem):
    mem.update_agent_data("onchain_agent", False, condition="greed_peak")
    data = mem._agent_data["onchain_agent"]
    assert "greed_peak" in data["worst_conditions"]


# ── Shared context ─────────────────────────────────────────────────────────────

def test_set_and_get_shared_context(mem):
    mem.set_shared_context("market_condition", "bull")
    assert mem.get_shared_context("market_condition") == "bull"


def test_shared_context_default(mem):
    assert mem.get_shared_context("nonexistent_key", "fallback") == "fallback"


def test_broadcast_alert_stored(mem):
    mem.broadcast_alert("fed_sentiment_agent", "FOMC meeting tomorrow — reduce risk")
    alerts = mem.get_shared_context("active_alerts", [])
    assert len(alerts) == 1
    assert "FOMC" in alerts[0]["message"]


def test_broadcast_alert_capped_at_10(mem):
    for i in range(15):
        mem.broadcast_alert("smc_agent", f"Alert {i}")
    alerts = mem.get_shared_context("active_alerts", [])
    assert len(alerts) == 10


def test_get_full_shared_context_is_dict(mem):
    ctx = mem.get_full_shared_context()
    assert isinstance(ctx, dict)
    assert "market_condition" in ctx


# ── Memory summary ─────────────────────────────────────────────────────────────

def test_memory_summary_returns_string(mem):
    summary = mem.memory_summary()
    assert isinstance(summary, str)
    assert "MEMORIA" in summary


def test_memory_summary_contains_counts(mem):
    mem.record_trade("BTCUSDT", 60000, 61000, 100.0, ["smc_agent"], True)
    summary = mem.memory_summary()
    assert "1" in summary  # at least 1 trade
