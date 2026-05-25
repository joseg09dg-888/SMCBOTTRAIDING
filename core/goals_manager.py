# core/goals_manager.py
import json
import sqlite3
from memory.episodic_db import get_db, get_goals, update_goal


class GoalsManager:
    """
    Evaluates autonomous bot goals daily from episodic DB.
    Updates progress for: win_rate_pct_100, axi_edge_score,
    challenge_passed, funded_usd.
    """

    def __init__(self, conn: sqlite3.Connection = None):
        self._conn = conn

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn or get_db()

    def _compute_metrics(self, conn: sqlite3.Connection) -> dict:
        rows = conn.execute(
            """SELECT result FROM episodes
               WHERE result IN ('WIN','LOSS')
               ORDER BY id DESC LIMIT 100"""
        ).fetchall()
        total  = len(rows)
        wins   = sum(1 for r in rows if r["result"] == "WIN")
        win_rate = (wins / total * 100) if total > 0 else 0.0

        edge = self._compute_axi_edge(win_rate, total)
        return {
            "win_rate_pct_100":  win_rate,
            "axi_edge_score":    edge,
            "challenge_passed":  0.0,
            "funded_usd":        0.0,
        }

    def _compute_axi_edge(self, win_rate: float, total: int) -> float:
        if total == 0:
            return 0.0
        pf = max(0.0, win_rate / max(100.0 - win_rate, 1.0))
        return round((win_rate * 0.6 + pf * 0.4) * 50 / 100, 1)

    def evaluate(self) -> dict:
        conn = self._get_conn()
        metrics = self._compute_metrics(conn)
        for goal_id, metric in [
            ("winrate_65",    "win_rate_pct_100"),
            ("edge_score_50", "axi_edge_score"),
            ("axi_challenge", "challenge_passed"),
            ("funded_5k",     "funded_usd"),
            ("funded_1m",     "funded_usd"),
        ]:
            val = metrics.get(metric, 0.0)
            update_goal(goal_id, val, conn=conn)
        return metrics

    def format_goals_summary(self) -> str:
        conn = self._get_conn()
        goals = get_goals(conn=conn)
        lines = []
        for g in goals:
            pct  = g.get("progress_pct", 0)
            bar  = int(min(pct, 100) / 10)
            prog = "#" * bar + "." * (10 - bar)
            lines.append(
                f"{g['description']}: [{prog}] {pct:.0f}%"
            )
        return "\n".join(lines) if lines else "No goals set."

    def get_goals_snapshot(self) -> str:
        conn = self._get_conn()
        goals = get_goals(conn=conn)
        return json.dumps(goals, default=str)
