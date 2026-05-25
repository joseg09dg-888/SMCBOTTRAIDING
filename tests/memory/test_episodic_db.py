# tests/memory/test_episodic_db.py
import pytest
import sqlite3
from datetime import datetime, timezone

from memory.episodic_db import _create_tables, record_episode, update_episode_result, \
    query_similar_episodes, get_setup_stats, get_session_stats, \
    save_lesson, save_research, save_report, get_goals, update_goal, seed_goals


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _create_tables(c)
    return c


def _ep(symbol="USDJPY", setup="CHoCH+OB", regime="trending", session="london"):
    return {
        "ts": "2026-05-25T10:00:00Z",
        "symbol": symbol,
        "timeframe": "H4",
        "direction": "BUY",
        "entry": 155.50,
        "sl": 155.20,
        "tp": 156.00,
        "ticket": 12345,
        "score": 72,
        "setup_type": setup,
        "regime": regime,
        "session": session,
        "reasoning": '{"decision":"LONG"}',
        "macro_ctx": None,
    }


class TestRecordEpisode:
    def test_returns_positive_id(self, conn):
        eid = record_episode(_ep(), conn=conn)
        assert eid > 0

    def test_stores_all_fields(self, conn):
        eid = record_episode(_ep(), conn=conn)
        row = conn.execute("SELECT * FROM episodes WHERE id=?", (eid,)).fetchone()
        assert row["symbol"] == "USDJPY"
        assert row["direction"] == "BUY"
        assert row["score"] == 72
        assert row["result"] is None

    def test_multiple_episodes_increment_id(self, conn):
        id1 = record_episode(_ep(), conn=conn)
        id2 = record_episode(_ep(symbol="EURUSD"), conn=conn)
        assert id2 > id1

    def test_missing_optional_fields_ok(self, conn):
        ep = {"ts": "2026-05-25T10:00:00Z", "symbol": "GBPUSD",
              "timeframe": "H1", "direction": "SELL",
              "entry": 1.2700, "sl": 0.0, "tp": 0.0}
        eid = record_episode(ep, conn=conn)
        assert eid > 0


class TestUpdateEpisodeResult:
    def test_sets_result_to_win(self, conn):
        eid = record_episode(_ep(), conn=conn)
        update_episode_result(eid, exit_price=156.00, pnl=50.0,
                              result="WIN", lesson="CHoCH+OB works in trending", conn=conn)
        row = conn.execute("SELECT result, pnl, lesson FROM episodes WHERE id=?", (eid,)).fetchone()
        assert row["result"] == "WIN"
        assert row["pnl"] == 50.0
        assert "CHoCH" in row["lesson"]

    def test_sets_result_to_loss(self, conn):
        eid = record_episode(_ep(), conn=conn)
        update_episode_result(eid, exit_price=155.20, pnl=-30.0,
                              result="LOSS", lesson=None, conn=conn)
        row = conn.execute("SELECT result FROM episodes WHERE id=?", (eid,)).fetchone()
        assert row["result"] == "LOSS"

    def test_nonexistent_id_does_not_raise(self, conn):
        update_episode_result(9999, exit_price=1.0, pnl=0.0,
                              result="WIN", lesson=None, conn=conn)


class TestQuerySimilarEpisodes:
    def test_returns_similar_by_setup_and_regime(self, conn):
        for _ in range(3):
            eid = record_episode(_ep(), conn=conn)
            update_episode_result(eid, 156.0, 50.0, "WIN", None, conn=conn)
        record_episode(_ep(setup="FVG", regime="ranging"), conn=conn)
        results = query_similar_episodes("USDJPY", "CHoCH+OB", "trending", n=10, conn=conn)
        assert len(results) == 3

    def test_returns_at_most_n(self, conn):
        for _ in range(10):
            eid = record_episode(_ep(), conn=conn)
            update_episode_result(eid, 156.0, 50.0, "WIN", None, conn=conn)
        results = query_similar_episodes("USDJPY", "CHoCH+OB", "trending", n=5, conn=conn)
        assert len(results) <= 5

    def test_excludes_open_episodes(self, conn):
        record_episode(_ep(), conn=conn)  # result=None (OPEN)
        results = query_similar_episodes("USDJPY", "CHoCH+OB", "trending", n=10, conn=conn)
        assert len(results) == 0

    def test_returns_empty_for_unknown_setup(self, conn):
        results = query_similar_episodes("XAUUSD", "UNKNOWN", "trending", n=10, conn=conn)
        assert results == []


class TestGetSetupStats:
    def test_computes_win_rate_per_setup(self, conn):
        for i in range(6):
            eid = record_episode(_ep(setup="CHoCH+OB"), conn=conn)
            update_episode_result(eid, 156.0, 50.0, "WIN" if i < 4 else "LOSS", None, conn=conn)
        stats = get_setup_stats(conn=conn)
        assert "CHoCH+OB" in stats
        assert abs(stats["CHoCH+OB"]["win_rate"] - 66.67) < 1.0
        assert stats["CHoCH+OB"]["sample_size"] == 6

    def test_returns_empty_dict_if_no_closed_episodes(self, conn):
        record_episode(_ep(), conn=conn)  # OPEN
        assert get_setup_stats(conn=conn) == {}


class TestGetSessionStats:
    def test_computes_win_rate_per_session(self, conn):
        for _ in range(3):
            eid = record_episode(_ep(session="london"), conn=conn)
            update_episode_result(eid, 156.0, 50.0, "WIN", None, conn=conn)
        stats = get_session_stats(conn=conn)
        assert stats["london"]["win_rate"] == 100.0

    def test_multiple_sessions_independent(self, conn):
        for sess in ["london", "ny", "asia"]:
            eid = record_episode(_ep(session=sess), conn=conn)
            update_episode_result(eid, 156.0, 50.0, "WIN", None, conn=conn)
        stats = get_session_stats(conn=conn)
        assert len(stats) == 3


class TestSaveLesson:
    def test_saves_lesson_to_db(self, conn):
        save_lesson({"setup_type": "CHoCH+OB", "regime": "trending",
                     "session": "london", "win_rate": 0.77,
                     "sample_size": 9, "weight_adj": 1.20, "notes": "good"}, conn=conn)
        row = conn.execute("SELECT * FROM lessons").fetchone()
        assert row["setup_type"] == "CHoCH+OB"
        assert row["weight_adj"] == 1.20


class TestSaveResearch:
    def test_saves_research_item(self, conn):
        save_research({"source": "arxiv", "title": "SMC Alpha",
                       "summary": "ICT order blocks study",
                       "url": "https://arxiv.org/abs/2601.0001",
                       "relevance": 0.85}, conn=conn)
        row = conn.execute("SELECT * FROM research").fetchone()
        assert row["source"] == "arxiv"
        assert row["relevance"] == 0.85
        assert row["applied"] == 0


class TestSaveReport:
    def test_saves_report(self, conn):
        save_report({"date": "2026-05-25", "trades_total": 5, "trades_win": 3,
                     "trades_loss": 2, "pnl_day": 120.0, "win_rate": 60.0,
                     "best_setup": "CHoCH+OB", "worst_setup": "FVG",
                     "lessons_text": "Trending works", "plan_tomorrow": "Focus H4",
                     "goals_snapshot": "{}", "report_text": "Good day"}, conn=conn)
        row = conn.execute("SELECT * FROM reports").fetchone()
        assert row["date"] == "2026-05-25"
        assert row["pnl_day"] == 120.0


class TestGoals:
    def test_seed_goals_inserts_5_defaults(self, conn):
        seed_goals(conn=conn)
        rows = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
        assert rows == 5

    def test_seed_goals_idempotent(self, conn):
        seed_goals(conn=conn)
        seed_goals(conn=conn)
        rows = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
        assert rows == 5

    def test_get_goals_returns_list(self, conn):
        seed_goals(conn=conn)
        goals = get_goals(conn=conn)
        assert len(goals) == 5
        assert all("goal_id" in g for g in goals)

    def test_update_goal_sets_current(self, conn):
        seed_goals(conn=conn)
        update_goal("winrate_65", 62.5, conn=conn)
        row = conn.execute("SELECT current, progress_pct FROM goals WHERE goal_id=?",
                           ("winrate_65",)).fetchone()
        assert row["current"] == 62.5
        assert row["progress_pct"] == pytest.approx(62.5 / 65 * 100, abs=0.1)

    def test_update_nonexistent_goal_does_not_raise(self, conn):
        update_goal("nonexistent", 42.0, conn=conn)


class TestWALMode:
    def test_wal_pragma_set_on_file_db(self, tmp_path):
        from memory.episodic_db import get_db
        db_file = str(tmp_path / "test.db")
        c = get_db(path=db_file)
        mode = c.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        c.close()
