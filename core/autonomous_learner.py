# core/autonomous_learner.py
import sqlite3
from typing import Optional
from memory.episodic_db import get_db, save_lesson


class AutonomousLearner:
    """
    Every 60 minutes, analyze real trade history and auto-adjust
    DecisionFilter weights per (setup_type, regime, session) group.

    Weight rules:
      win_rate > 65%  -> weight_adj = 1.20  (lower threshold)
      50-65%          -> weight_adj = 1.00  (neutral)
      < 50%           -> weight_adj = 0.90  (slight raise)
      < 35% + 20+     -> weight_adj = 0.80  (moderate raise)

    Groups with fewer than 20 samples are skipped (avoid noise from small samples).
    """

    def __init__(self, conn: sqlite3.Connection = None):
        self._conn = conn
        self._weights: dict = {}

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn or get_db()

    def run_analysis(self) -> dict:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT setup_type, regime, session, result
               FROM episodes
               WHERE result IN ('WIN','LOSS')
               ORDER BY id DESC LIMIT 500"""
        ).fetchall()

        groups: dict = {}
        for r in rows:
            key = (r["setup_type"] or "unknown",
                   r["regime"]     or "unknown",
                   r["session"]    or "unknown")
            if key not in groups:
                groups[key] = {"wins": 0, "total": 0}
            groups[key]["total"] += 1
            if r["result"] == "WIN":
                groups[key]["wins"] += 1

        result = {}
        for key, data in groups.items():
            total = data["total"]
            if total < 20:
                continue
            win_rate = data["wins"] / total
            if win_rate > 0.65:
                adj = 1.20
            elif win_rate >= 0.50:
                adj = 1.00
            elif win_rate >= 0.35:
                adj = 0.90
            else:
                adj = 0.80

            result[key] = {"weight_adj": adj, "win_rate": win_rate * 100,
                           "sample_size": total}
            setup, regime, session = key
            save_lesson({
                "setup_type": setup, "regime": regime, "session": session,
                "win_rate": win_rate * 100, "sample_size": total,
                "weight_adj": adj,
                "notes": f"auto-adjusted {setup}/{regime}: {win_rate*100:.1f}%",
            }, conn=conn)
            print(
                f"[LEARN] {setup}+{regime}+{session}: "
                f"{data['wins']}/{total} WIN ({win_rate*100:.0f}%) "
                f"-> weight {adj:+.0%}",
                flush=True,
            )

        self._weights = result
        return result

    def get_weight_adj(self, setup_type: str, regime: str, session: str) -> float:
        key = (setup_type or "unknown", regime or "unknown", session or "unknown")
        return self._weights.get(key, {}).get("weight_adj", 1.0)

    def effective_threshold(self, base_threshold: int,
                             setup_type: str, regime: str, session: str) -> int:
        adj = self.get_weight_adj(setup_type, regime, session)
        if adj == 1.0:
            return base_threshold
        new_thr = int(base_threshold / adj)
        # Cap: learner can only raise threshold by +5 or lower by -15
        max_thr = base_threshold + 5
        min_thr = max(25, base_threshold - 15)
        return max(min_thr, min(max_thr, new_thr))
