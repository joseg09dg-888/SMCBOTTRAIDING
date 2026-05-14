"""Tests for LearningEngine."""
import pytest
from core.agent_memory import AgentMemoryManager
from core.learning_engine import LearningEngine


@pytest.fixture
def mem(tmp_path):
    m = AgentMemoryManager(
        db_path=str(tmp_path / "db.db"),
        agents_dir=str(tmp_path / "agents"),
        shared_ctx_path=str(tmp_path / "ctx.json"),
        vector_dir=str(tmp_path / "vs"),
    )
    yield m
    m.close()


@pytest.fixture
def engine(mem):
    return LearningEngine(mem)


# ── daily_review ───────────────────────────────────────────────────────────────

def test_daily_review_returns_dict(engine):
    result = engine.daily_review()
    assert isinstance(result, dict)
    assert "trades_today" in result
    assert "agent_performance" in result


def test_daily_review_no_trades_zero_wins(engine):
    result = engine.daily_review()
    assert result["trades_today"] == 0
    assert result["wins_today"] == 0
    assert result["win_rate"] == 0.0


def test_daily_review_counts_trades(engine, mem):
    mem.record_trade("BTCUSDT", 60000, 62000, 200.0, [], True)
    mem.record_trade("ETHUSDT", 3000, 2900, -50.0, [], False)
    result = engine.daily_review()
    assert result["trades_today"] == 2
    assert result["wins_today"] == 1
    assert result["losses_today"] == 1


def test_daily_review_calculates_pnl(engine, mem):
    mem.record_trade("BTCUSDT", 60000, 62000, 150.0, [], True)
    mem.record_trade("ETHUSDT", 3000, 2900, -50.0, [], False)
    result = engine.daily_review()
    assert abs(result["daily_pnl"] - 100.0) < 0.01


# ── Weight adjustment ──────────────────────────────────────────────────────────

def test_high_accuracy_increases_weight(engine, mem):
    # 8/10 correct = 80% accuracy -> increase
    for _ in range(8):
        mem.update_agent_data("onchain_agent", True)
    for _ in range(2):
        mem.update_agent_data("onchain_agent", False)
    initial_weight = mem.get_agent_weight("onchain_agent")
    engine.daily_review()
    new_weight = mem.get_agent_weight("onchain_agent")
    assert new_weight > initial_weight


def test_low_accuracy_reduces_weight(engine, mem):
    # 2/10 correct = 20% -> reduce
    for _ in range(2):
        mem.update_agent_data("lunar_agent", True)
    for _ in range(8):
        mem.update_agent_data("lunar_agent", False)
    initial_weight = mem.get_agent_weight("lunar_agent")
    engine.daily_review()
    new_weight = mem.get_agent_weight("lunar_agent")
    assert new_weight < initial_weight


def test_medium_accuracy_maintains_weight(engine, mem):
    # 6/10 = 60% -> maintain
    for _ in range(6):
        mem.update_agent_data("geopolitical_agent", True)
    for _ in range(4):
        mem.update_agent_data("geopolitical_agent", False)
    initial_weight = mem.get_agent_weight("geopolitical_agent")
    engine.daily_review()
    new_weight = mem.get_agent_weight("geopolitical_agent")
    assert abs(new_weight - initial_weight) < 0.001


def test_very_low_accuracy_adds_lesson(engine, mem):
    # 1/10 = 10% -> reduce + lesson
    for _ in range(1):
        mem.update_agent_data("elliott_agent", True)
    for _ in range(9):
        mem.update_agent_data("elliott_agent", False)
    engine.daily_review()
    lessons = mem._agent_data["elliott_agent"]["lessons_learned"]
    assert len(lessons) >= 1


# ── evaluate_agent_signal ──────────────────────────────────────────────────────

def test_evaluate_correct_signal(engine, mem):
    engine.evaluate_agent_signal("smc_agent", True, "trending_bull")
    assert mem.get_agent_accuracy("smc_agent") == 1.0


def test_evaluate_incorrect_signal(engine, mem):
    engine.evaluate_agent_signal("smc_agent", True)
    engine.evaluate_agent_signal("smc_agent", False)
    assert mem.get_agent_accuracy("smc_agent") == 0.5


# ── Leaderboard ────────────────────────────────────────────────────────────────

def test_get_top_agents_empty_when_no_signals(engine):
    top = engine.get_top_agents(3)
    assert top == []


def test_get_top_agents_ranked_by_accuracy(engine, mem):
    # Give agent A 9/10, agent B 5/10
    for _ in range(9): mem.update_agent_data("onchain_agent", True)
    mem.update_agent_data("onchain_agent", False)
    for _ in range(5): mem.update_agent_data("chaos_agent", True)
    for _ in range(5): mem.update_agent_data("chaos_agent", False)
    top = engine.get_top_agents(2)
    assert top[0][0] == "onchain_agent"
    assert top[0][1] > top[1][1]


def test_get_underperforming_empty_when_accurate(engine, mem):
    for _ in range(10):
        mem.update_agent_data("smc_agent", True)
    under = engine.get_underperforming_agents()
    assert not any(a == "smc_agent" for a, _ in under)


def test_get_underperforming_detects_bad_agent(engine, mem):
    for _ in range(2): mem.update_agent_data("lunar_agent", True)
    for _ in range(10): mem.update_agent_data("lunar_agent", False)
    under = engine.get_underperforming_agents()
    assert any(a == "lunar_agent" for a, _ in under)


# ── generate_daily_report ──────────────────────────────────────────────────────

def test_daily_report_returns_string(engine):
    report = engine.generate_daily_report()
    assert isinstance(report, str)
    assert "REPORTE" in report


def test_daily_report_contains_trade_counts(engine, mem):
    mem.record_trade("BTCUSDT", 60000, 62000, 100.0, [], True)
    review = engine.daily_review()
    report = engine.generate_daily_report(review)
    assert "1" in report  # at least mentions the trade count


def test_daily_report_contains_win_rate(engine, mem):
    mem.record_trade("BTCUSDT", 60000, 62000, 100.0, [], True)
    mem.record_trade("ETHUSDT", 3000, 2900, -30.0, [], False)
    review = engine.daily_review()
    report = engine.generate_daily_report(review)
    assert "50%" in report or "Win" in report


def test_daily_report_has_learning_section(engine):
    report = engine.generate_daily_report()
    assert "Aprendizaje" in report or "learning" in report.lower()


def test_weight_action_new_for_few_signals(engine, mem):
    # Only 1 signal -> action should be "new"
    mem.update_agent_data("geopolitical_agent", True)
    review = engine.daily_review()
    action = review["agent_performance"]["geopolitical_agent"]["action"]
    assert action == "new"
