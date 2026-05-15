"""
tests/agents/test_quant_intel.py
TDD tests for agents/quant_intel.py — Collective Intelligence
"""

import pytest
from agents.quant_intel import CollectiveIntelligence, MarketConsensus, AcademicPaper, InsiderActivity


def test_academic_knowledge_not_empty():
    assert len(CollectiveIntelligence.ACADEMIC_KNOWLEDGE) >= 5


def test_get_consensus_returns_consensus():
    ci = CollectiveIntelligence()
    c = ci.get_consensus_bias("BTCUSDT")
    assert isinstance(c, MarketConsensus)
    assert 0.0 <= c.bullish_pct <= 1.0
    assert 0.0 <= c.bearish_pct <= 1.0
    assert 0.0 <= c.neutral_pct <= 1.0


def test_consensus_probabilities_sum_to_one():
    ci = CollectiveIntelligence()
    c = ci.get_consensus_bias("ETHUSDT")
    total = c.bullish_pct + c.bearish_pct + c.neutral_pct
    assert abs(total - 1.0) < 0.01


def test_consensus_deterministic():
    ci = CollectiveIntelligence()
    c1 = ci.get_consensus_bias("BTCUSDT")
    c2 = ci.get_consensus_bias("BTCUSDT")
    assert c1.bullish_pct == c2.bullish_pct


def test_get_relevant_papers_filters():
    ci = CollectiveIntelligence()
    papers = ci.get_relevant_papers(min_relevance=0.8)
    for p in papers:
        assert p.relevance_score >= 0.8


def test_get_relevant_papers_all():
    ci = CollectiveIntelligence()
    papers = ci.get_relevant_papers(strategy_type="all", min_relevance=0.0)
    assert len(papers) == len(CollectiveIntelligence.ACADEMIC_KNOWLEDGE)


def test_get_relevant_papers_returns_list():
    ci = CollectiveIntelligence()
    papers = ci.get_relevant_papers()
    assert isinstance(papers, list)


def test_strategy_edge_with_momentum_tag():
    ci = CollectiveIntelligence()
    edge = ci.get_strategy_edge_from_papers(["momentum", "BOS"])
    assert edge > 0.0


def test_strategy_edge_no_match():
    ci = CollectiveIntelligence()
    edge = ci.get_strategy_edge_from_papers(["unknown_tag_xyz"])
    assert edge == 0.0


def test_get_insider_activity_returns():
    ci = CollectiveIntelligence()
    ia = ci.get_insider_activity("BTCUSDT")
    assert isinstance(ia, InsiderActivity)
    assert ia.action in ("buy", "sell", "neutral")
    assert -5 <= ia.pts <= 5


def test_get_insider_deterministic():
    ci = CollectiveIntelligence()
    ia1 = ci.get_insider_activity("AAPL")
    ia2 = ci.get_insider_activity("AAPL")
    assert ia1.pts == ia2.pts


def test_calculate_collective_score_range():
    ci = CollectiveIntelligence()
    score = ci.calculate_collective_score("BTCUSDT", ["OB", "FVG", "momentum"])
    assert -10 <= score <= 10


def test_calculate_collective_score_returns_int():
    ci = CollectiveIntelligence()
    score = ci.calculate_collective_score("ETHUSDT")
    assert isinstance(score, int)


def test_paper_edge_high_for_funding():
    ci = CollectiveIntelligence()
    edge = ci.get_strategy_edge_from_papers(["funding", "extreme"])
    assert edge >= 0.08  # funding rate paper has expected_edge_pct = 0.12
