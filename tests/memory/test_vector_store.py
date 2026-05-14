"""Tests for Layer 3 vector store (JSON fallback path)."""
import pytest
from core.agent_memory import AgentMemoryManager


@pytest.fixture
def mem(tmp_path):
    # ChromaDB may or may not be installed — tests use JSON fallback path
    m = AgentMemoryManager(
        db_path=str(tmp_path / "db.db"),
        agents_dir=str(tmp_path / "agents"),
        shared_ctx_path=str(tmp_path / "ctx.json"),
        vector_dir=str(tmp_path / "vector_store"),
    )
    yield m
    m.close()


def test_store_knowledge_no_error(mem):
    """store_knowledge should never raise."""
    mem.store_knowledge("smc_agent", "Order block at 60000 BTCUSDT acted as support")


def test_search_returns_list(mem):
    mem.store_knowledge("onchain_agent", "Fear and greed at 12 — extreme fear")
    results = mem.search_knowledge("fear greed")
    assert isinstance(results, list)


def test_search_finds_stored_text(mem):
    mem.store_knowledge("lunar_agent", "Full moon on November 15 — BTC rallied 8%")
    results = mem.search_knowledge("full moon")
    assert any("moon" in r["text"].lower() for r in results)


def test_multiple_entries_all_searchable(mem):
    entries = [
        ("smc_agent",      "BOS breakout confirmed at London open"),
        ("elliott_agent",  "Wave 3 extension hit 1.618 Fibonacci"),
        ("onchain_agent",  "MVRV ratio below 1 — buy zone"),
    ]
    for agent, text in entries:
        mem.store_knowledge(agent, text)

    assert len(mem.search_knowledge("BOS breakout")) >= 1
    assert len(mem.search_knowledge("Fibonacci")) >= 1
    assert len(mem.search_knowledge("MVRV")) >= 1


def test_knowledge_count_increments(mem):
    c0 = mem.get_knowledge_count()
    mem.store_knowledge("chaos_agent", "Hurst exponent 0.71 — trending market")
    c1 = mem.get_knowledge_count()
    assert c1 == c0 + 1


def test_search_with_no_match_returns_empty(mem):
    results = mem.search_knowledge("zzz_no_match_xyz_9999")
    assert results == []


def test_add_lesson_stored_in_vector_kb(mem):
    mem.add_lesson("microstructure_agent",
                   "Stop hunts at $60K BTC work 80% of the time")
    results = mem.search_knowledge("stop hunt")
    assert any("stop" in r["text"].lower() for r in results)


def test_vector_store_metadata_has_agent(mem):
    mem.store_knowledge("fed_sentiment_agent", "Dovish pivot expected Q2 2025",
                        {"topic": "fed"})
    results = mem.search_knowledge("dovish")
    if results:
        assert "agent" in results[0].get("meta", {})
