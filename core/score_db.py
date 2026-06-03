"""SQLite persistence for scores and demo trades."""
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memory" / "scores.db"


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT,
            symbol    TEXT,
            timeframe TEXT,
            score     INTEGER,
            direction TEXT,
            entry     REAL,
            sl        REAL,
            tp        REAL,
            executed  INTEGER DEFAULT 0,
            outcome   TEXT DEFAULT NULL,
            pnl_pct   REAL DEFAULT NULL
        )
    """)
    # Safe migration: add columns if they don't exist yet
    for col, definition in [("outcome", "TEXT DEFAULT NULL"), ("pnl_pct", "REAL DEFAULT NULL")]:
        try:
            conn.execute(f"ALTER TABLE scores ADD COLUMN {col} {definition}")
        except Exception:
            pass
    conn.commit()
    return conn


def save_score(symbol: str, timeframe: str, score: int,
               direction: str, entry: float = 0.0,
               sl: float = 0.0, tp: float = 0.0,
               executed: bool = False) -> int:
    """Insert a new score row. Returns the row id."""
    try:
        conn = _conn()
        cur = conn.execute(
            "INSERT INTO scores VALUES (NULL,?,?,?,?,?,?,?,?,?,NULL,NULL)",
            (datetime.now(timezone.utc).isoformat(),
             symbol, timeframe, score, direction,
             entry, sl, tp, 1 if executed else 0)
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id or 0
    except Exception:
        return 0


def update_score_outcome(symbol: str, entry: float, outcome: str, pnl_pct: float = 0.0):
    """Set WIN/LOSS outcome on the most recent executed row for this symbol+entry."""
    try:
        conn = _conn()
        conn.execute(
            """UPDATE scores SET outcome=?, pnl_pct=?
               WHERE id=(
                 SELECT id FROM scores
                 WHERE symbol=? AND ABS(entry-?)<0.001 AND executed=1 AND outcome IS NULL
                 ORDER BY id DESC LIMIT 1
               )""",
            (outcome, pnl_pct, symbol, entry)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_recent_scores(n: int = 10) -> list:
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT ts,symbol,timeframe,score,direction,entry,executed "
            "FROM scores ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def get_stats() -> dict:
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT score, direction, executed, outcome, pnl_pct FROM scores"
        ).fetchall()
        conn.close()
        total    = len(rows)
        executed = sum(1 for r in rows if r[2])

        # Real outcomes when available
        with_outcome = [r for r in rows if r[2] and r[3] is not None]
        if with_outcome:
            wins     = sum(1 for r in with_outcome if r[3] == "WIN")
            losses   = sum(1 for r in with_outcome if r[3] == "LOSS")
            win_rate = (wins / len(with_outcome) * 100) if with_outcome else 0.0
            pnl_vals = [r[4] for r in with_outcome if r[4] is not None]
            avg_pnl  = sum(pnl_vals) / len(pnl_vals) if pnl_vals else 0.0
        else:
            # Fallback: score >= 60 as proxy until real outcomes accumulate
            wins     = sum(1 for r in rows if r[0] >= 60 and r[2])
            losses   = max(0, executed - wins)
            win_rate = (wins / executed * 100) if executed > 0 else 0.0
            avg_pnl  = 0.0

        profit_factor = wins / max(losses, 1)

        return {
            "total": total,
            "executed": executed,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "high_score": wins,
            "profit_factor": profit_factor,
            "avg_pnl_pct": avg_pnl,
            "has_real_outcomes": len(with_outcome) > 0,
        }
    except Exception:
        return {
            "total": 0, "executed": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "high_score": 0,
            "profit_factor": 0.0, "avg_pnl_pct": 0.0,
            "has_real_outcomes": False,
        }
