# tests/core/test_goals_manager.py
import pytest
import sqlite3
from memory.episodic_db import _create_tables, record_episode, update_episode_result, seed_goals
from core.goals_manager import GoalsManager


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _create_tables(c)
    seed_goals(conn=c)
    return c


def _add_closed(conn, wins, losses):
    for i in range(wins + losses):
        eid = record_episode({
            "ts": "2026-05-25T10:00:00Z", "symbol": "USDJPY",
            "timeframe": "H4", "direction": "BUY", "entry": 155.5,
            "score": 70, "setup_type": "CHoCH+OB",
        }, conn=conn)
        update_episode_result(eid, 156.0,
                              50.0 if i < wins else -30.0,
                              "WIN" if i < wins else "LOSS",
                              None, conn=conn)


class TestGoalsManager:
    def test_instantiates(self, conn):
        gm = GoalsManager(conn=conn)
        assert gm is not None

    def test_evaluate_returns_dict(self, conn):
        _add_closed(conn, 10, 5)
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert isinstance(result, dict)

    def test_win_rate_metric_computed(self, conn):
        _add_closed(conn, 7, 3)
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert "win_rate_pct_100" in result
        assert abs(result["win_rate_pct_100"] - 70.0) < 1.0

    def test_goals_updated_in_db(self, conn):
        _add_closed(conn, 7, 3)
        gm = GoalsManager(conn=conn)
        gm.evaluate()
        row = conn.execute(
            "SELECT current FROM goals WHERE goal_id='winrate_65'"
        ).fetchone()
        assert row is not None
        assert abs(row["current"] - 70.0) < 1.0

    def test_progress_pct_computed(self, conn):
        _add_closed(conn, 7, 3)
        gm = GoalsManager(conn=conn)
        gm.evaluate()
        row = conn.execute(
            "SELECT progress_pct FROM goals WHERE goal_id='winrate_65'"
        ).fetchone()
        assert row["progress_pct"] > 100.0

    def test_axi_edge_score_metric(self, conn):
        _add_closed(conn, 7, 3)
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert "axi_edge_score" in result
        assert result["axi_edge_score"] >= 0

    def test_zero_trades_returns_zero_metrics(self, conn):
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert result.get("win_rate_pct_100", 0) == 0.0

    def test_format_goals_summary_returns_str(self, conn):
        gm = GoalsManager(conn=conn)
        summary = gm.format_goals_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_format_goals_includes_all_goals(self, conn):
        gm = GoalsManager(conn=conn)
        summary = gm.format_goals_summary()
        assert len(summary) > 10

    def test_evaluate_no_crash_with_empty_db(self, conn):
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert isinstance(result, dict)

    def test_win_rate_pct_100_uses_last_100_closed(self, conn):
        _add_closed(conn, 60, 40)
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert abs(result["win_rate_pct_100"] - 60.0) < 2.0

    def test_multiple_evaluate_calls_update_db(self, conn):
        _add_closed(conn, 7, 3)
        gm = GoalsManager(conn=conn)
        gm.evaluate()
        _add_closed(conn, 1, 0)
        gm.evaluate()
        row = conn.execute(
            "SELECT current FROM goals WHERE goal_id='winrate_65'"
        ).fetchone()
        assert row["current"] > 0

    def test_funded_usd_metric_default_zero(self, conn):
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert result.get("funded_usd", 0) == 0.0

    def test_challenge_passed_default_zero(self, conn):
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()
        assert result.get("challenge_passed", 0) == 0.0

    def test_get_goals_snapshot_json(self, conn):
        gm = GoalsManager(conn=conn)
        snapshot = gm.get_goals_snapshot()
        import json
        data = json.loads(snapshot)
        assert isinstance(data, list)
        assert len(data) == 5
