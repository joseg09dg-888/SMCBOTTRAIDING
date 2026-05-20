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
            executed  INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def save_score(symbol: str, timeframe: str, score: int,
               direction: str, entry: float = 0.0,
               sl: float = 0.0, tp: float = 0.0,
               executed: bool = False):
    try:
        conn = _conn()
        conn.execute(
            "INSERT INTO scores VALUES (NULL,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(),
             symbol, timeframe, score, direction,
             entry, sl, tp, 1 if executed else 0)
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
            "SELECT score, direction, executed FROM scores"
        ).fetchall()
        conn.close()
        total    = len(rows)
        executed = sum(1 for r in rows if r[2])
        high     = sum(1 for r in rows if r[0] >= 60 and r[2])
        win_rate = (high / executed * 100) if executed > 0 else 0.0
        return {"total": total, "executed": executed,
                "win_rate": win_rate, "high_score": high}
    except Exception:
        return {"total": 0, "executed": 0, "win_rate": 0.0, "high_score": 0}
