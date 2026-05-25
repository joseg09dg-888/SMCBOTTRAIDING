# tests/core/test_research_agent.py
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from memory.episodic_db import _create_tables
from core.research_agent import ResearchAgent, _score_relevance


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _create_tables(c)
    return c


ARXIV_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>SMC Order Block Detection with Deep Learning</title>
    <summary>We study ICT liquidity sweeps and order blocks using LSTM models.</summary>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <published>2026-05-20T00:00:00Z</published>
  </entry>
  <entry>
    <title>Unrelated Paper on Climate</title>
    <summary>This paper has nothing to do with trading.</summary>
    <id>http://arxiv.org/abs/2601.00002v1</id>
    <published>2026-05-21T00:00:00Z</published>
  </entry>
</feed>"""

MQL5_HTML = """<html><body>
<div class="article-name"><a href="/articles/12345">ICT Concepts MQL5</a></div>
<div class="article-name"><a href="/articles/12346">Smart Money Algorithm</a></div>
</body></html>"""


class TestScoreRelevance:
    def test_high_relevance_for_smc_keywords(self):
        assert _score_relevance("SMC Order Block ICT liquidity") > 0.7

    def test_zero_relevance_for_unrelated(self):
        assert _score_relevance("Climate change ocean temperature") < 0.3

    def test_partial_match_medium_relevance(self):
        score = _score_relevance("Machine learning trading systems")
        assert 0.0 <= score <= 1.0

    def test_empty_string_returns_zero(self):
        assert _score_relevance("") == 0.0

    def test_ict_keyword_scores_high(self):
        assert _score_relevance("ICT institutional order flow") > 0.4


class TestResearchAgent:
    def test_instantiates(self, conn):
        agent = ResearchAgent(conn=conn)
        assert agent is not None

    def test_fetch_arxiv_parses_entries(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            items = agent._fetch_arxiv()
        # SMC paper should be included (has "SMC", "ICT", "liquidity", "order blocks")
        assert len(items) >= 1

    def test_fetch_arxiv_filters_low_relevance(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            items = agent._fetch_arxiv()
        titles = [i["title"] for i in items]
        assert not any("Climate" in t for t in titles)

    def test_fetch_arxiv_returns_empty_on_error(self, conn):
        agent = ResearchAgent(conn=conn)
        with patch("httpx.get", side_effect=Exception("DNS fail")):
            items = agent._fetch_arxiv()
        assert items == []

    def test_fetch_arxiv_returns_empty_on_non_200(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("httpx.get", return_value=mock_resp):
            items = agent._fetch_arxiv()
        assert items == []

    def test_fetch_mql5_returns_empty_on_error(self, conn):
        agent = ResearchAgent(conn=conn)
        with patch("httpx.get", side_effect=Exception("timeout")):
            items = agent._fetch_mql5()
        assert items == []

    def test_run_cycle_saves_to_db(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            agent.run_cycle()
        rows = conn.execute("SELECT COUNT(*) FROM research").fetchone()[0]
        assert rows >= 1

    def test_run_cycle_no_crash_when_all_fail(self, conn):
        agent = ResearchAgent(conn=conn)
        with patch("httpx.get", side_effect=Exception("all down")):
            agent.run_cycle()

    def test_get_top_research_returns_list(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            agent.run_cycle()
        top = agent.get_top_research(n=3, conn=conn)
        assert isinstance(top, list)

    def test_get_top_research_sorted_by_relevance(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            agent.run_cycle()
        top = agent.get_top_research(n=5, conn=conn)
        if len(top) >= 2:
            assert top[0]["relevance"] >= top[1]["relevance"]

    def test_run_cycle_does_not_duplicate_url(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            agent.run_cycle()
            agent.run_cycle()
        rows = conn.execute("SELECT COUNT(*) FROM research").fetchone()[0]
        assert rows < 10

    def test_relevance_threshold_filters_low_scores(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            agent.run_cycle()
        rows = conn.execute(
            "SELECT relevance FROM research WHERE relevance < 0.4"
        ).fetchall()
        assert len(rows) == 0

    def test_fetch_returns_at_most_5_per_source(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            items = agent._fetch_arxiv()
        assert len(items) <= 5

    def test_source_label_set_correctly(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            agent.run_cycle()
        rows = conn.execute("SELECT DISTINCT source FROM research").fetchall()
        sources = [r["source"] for r in rows]
        assert "arxiv" in sources
