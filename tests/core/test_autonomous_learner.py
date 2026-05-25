# tests/core/test_autonomous_learner.py
import pytest
import sqlite3
from memory.episodic_db import _create_tables, record_episode, update_episode_result
from core.autonomous_learner import AutonomousLearner


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _create_tables(c)
    return c


def _add_episodes(conn, setup, regime, session, wins, losses):
    for i in range(wins + losses):
        eid = record_episode({
            "ts": "2026-05-25T10:00:00Z",
            "symbol": "USDJPY", "timeframe": "H4",
            "direction": "BUY", "entry": 155.5,
            "setup_type": setup, "regime": regime, "session": session,
        }, conn=conn)
        result = "WIN" if i < wins else "LOSS"
        update_episode_result(eid, 156.0, 50.0 if result == "WIN" else -30.0,
                              result, None, conn=conn)


class TestAutonomousLearner:
    def test_instantiates(self, conn):
        learner = AutonomousLearner(conn=conn)
        assert learner is not None

    def test_run_analysis_returns_dict(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 5, 2)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        assert isinstance(result, dict)

    def test_high_win_rate_gets_boost(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("CHoCH+OB", "trending", "london")
        assert result[key]["weight_adj"] == 1.20

    def test_mid_win_rate_stays_neutral(self, conn):
        _add_episodes(conn, "FVG", "ranging", "ny", 3, 2)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("FVG", "ranging", "ny")
        assert result[key]["weight_adj"] == 1.00

    def test_low_win_rate_gets_penalty(self, conn):
        _add_episodes(conn, "OB", "high_vol", "asia", 2, 5)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("OB", "high_vol", "asia")
        assert result[key]["weight_adj"] == 0.80

    def test_very_low_win_rate_near_disabled(self, conn):
        _add_episodes(conn, "BOS", "ranging", "ny", 2, 10)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("BOS", "ranging", "ny")
        assert result[key]["weight_adj"] == 0.50

    def test_groups_below_5_samples_skipped(self, conn):
        _add_episodes(conn, "RARE", "trending", "london", 3, 1)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        assert ("RARE", "trending", "london") not in result

    def test_saves_lessons_to_db(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        rows = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        assert rows >= 1

    def test_get_weight_adj_returns_float(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        adj = learner.get_weight_adj("CHoCH+OB", "trending", "london")
        assert isinstance(adj, float)
        assert adj == 1.20

    def test_get_weight_adj_defaults_to_1_if_unknown(self, conn):
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        adj = learner.get_weight_adj("UNKNOWN", "trending", "london")
        assert adj == 1.0

    def test_65_pct_exactly_is_neutral(self, conn):
        _add_episodes(conn, "FVG", "trending", "london", 13, 7)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("FVG", "trending", "london")
        assert result[key]["weight_adj"] == 1.00

    def test_66_pct_is_boost(self, conn):
        _add_episodes(conn, "FVG", "trending", "london", 4, 2)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("FVG", "trending", "london")
        assert result[key]["weight_adj"] == 1.20

    def test_50_pct_is_neutral(self, conn):
        _add_episodes(conn, "FVG", "ranging", "asia", 5, 5)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("FVG", "ranging", "asia")
        assert result[key]["weight_adj"] == 1.00

    def test_35_pct_with_10_samples_near_disable(self, conn):
        _add_episodes(conn, "WEAK", "high_vol", "asia", 3, 7)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("WEAK", "high_vol", "asia")
        assert result[key]["weight_adj"] == 0.50

    def test_empty_db_returns_empty_dict(self, conn):
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        assert result == {}

    def test_run_analysis_updates_weight_cache(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        assert len(learner._weights) > 0

    def test_all_open_episodes_skipped(self, conn):
        for _ in range(10):
            record_episode({
                "ts": "2026-05-25T10:00:00Z", "symbol": "USDJPY",
                "timeframe": "H4", "direction": "BUY", "entry": 155.5,
                "setup_type": "CHoCH+OB", "regime": "trending",
            }, conn=conn)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        assert result == {}

    def test_mixed_setups_analyzed_independently(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        _add_episodes(conn, "FVG", "ranging", "ny", 2, 5)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        assert result[("CHoCH+OB", "trending", "london")]["weight_adj"] == 1.20
        assert result[("FVG", "ranging", "ny")]["weight_adj"] == 0.80

    def test_effective_threshold_decreases_on_boost(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        base_threshold = 30
        eff = learner.effective_threshold(base_threshold, "CHoCH+OB", "trending", "london")
        assert eff < base_threshold

    def test_effective_threshold_increases_on_penalty(self, conn):
        _add_episodes(conn, "FVG", "ranging", "ny", 2, 5)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        base_threshold = 30
        eff = learner.effective_threshold(base_threshold, "FVG", "ranging", "ny")
        assert eff > base_threshold

    def test_effective_threshold_unchanged_on_neutral(self, conn):
        _add_episodes(conn, "FVG", "ranging", "ny", 3, 2)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        base = 30
        eff = learner.effective_threshold(base, "FVG", "ranging", "ny")
        assert eff == base

    def test_print_summary_no_crash(self, conn, capsys):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        captured = capsys.readouterr()
        assert "[LEARN]" in captured.out

    def test_run_analysis_twice_updates_lessons(self, conn):
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 7, 2)
        learner = AutonomousLearner(conn=conn)
        learner.run_analysis()
        _add_episodes(conn, "CHoCH+OB", "trending", "london", 2, 1)
        learner.run_analysis()
        rows = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        assert rows >= 2

    def test_weight_adj_1_0_for_exact_neutral_boundary(self, conn):
        _add_episodes(conn, "FVG", "trending", "london", 10, 10)
        learner = AutonomousLearner(conn=conn)
        result = learner.run_analysis()
        key = ("FVG", "trending", "london")
        assert result[key]["weight_adj"] == 1.00
