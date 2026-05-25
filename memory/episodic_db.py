# memory/episodic_db.py
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

DB_PATH = Path(__file__).parent / "episodes.db"

INITIAL_GOALS = [
    {"goal_id": "edge_score_50",  "description": "Alcanzar Edge Score 50 en Axi Select",
     "metric": "axi_edge_score",  "target": 50.0,  "horizon": "short"},
    {"goal_id": "winrate_65",     "description": "Win rate > 65% en 100 trades",
     "metric": "win_rate_pct_100","target": 65.0,  "horizon": "medium"},
    {"goal_id": "axi_challenge",  "description": "Pasar challenge Axi $5K",
     "metric": "challenge_passed","target": 1.0,   "horizon": "medium"},
    {"goal_id": "funded_5k",      "description": "Cuenta fondeada $5,000 Axi",
     "metric": "funded_usd",      "target": 5000.0,"horizon": "long"},
    {"goal_id": "funded_1m",      "description": "Llegar a $1,000,000 fondeado",
     "metric": "funded_usd",      "target": 1000000.0, "horizon": "ultimate"},
]


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            timeframe   TEXT NOT NULL DEFAULT '',
            direction   TEXT NOT NULL,
            entry       REAL NOT NULL DEFAULT 0,
            sl          REAL,
            tp          REAL,
            ticket      INTEGER,
            score       INTEGER,
            setup_type  TEXT,
            regime      TEXT,
            session     TEXT,
            reasoning   TEXT,
            macro_ctx   TEXT,
            exit_price  REAL,
            pnl         REAL,
            result      TEXT,
            lesson      TEXT
        );

        CREATE TABLE IF NOT EXISTS lessons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            setup_type  TEXT NOT NULL,
            regime      TEXT,
            session     TEXT,
            win_rate    REAL,
            sample_size INTEGER,
            weight_adj  REAL,
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS goals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id      TEXT UNIQUE NOT NULL,
            description  TEXT NOT NULL,
            metric       TEXT NOT NULL,
            target       REAL NOT NULL,
            current      REAL DEFAULT 0,
            progress_pct REAL DEFAULT 0,
            horizon      TEXT,
            updated_ts   TEXT
        );

        CREATE TABLE IF NOT EXISTS research (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT NOT NULL,
            source    TEXT NOT NULL,
            title     TEXT,
            summary   TEXT,
            url       TEXT,
            applied   INTEGER DEFAULT 0,
            relevance REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            date           TEXT UNIQUE NOT NULL,
            trades_total   INTEGER DEFAULT 0,
            trades_win     INTEGER DEFAULT 0,
            trades_loss    INTEGER DEFAULT 0,
            pnl_day        REAL DEFAULT 0,
            win_rate       REAL DEFAULT 0,
            best_setup     TEXT,
            worst_setup    TEXT,
            lessons_text   TEXT,
            plan_tomorrow  TEXT,
            goals_snapshot TEXT,
            report_text    TEXT
        );
    """)
    conn.commit()


def get_db(path: str = None) -> sqlite3.Connection:
    db_path = path or str(DB_PATH)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    if db_path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_episode(ep: dict, conn: sqlite3.Connection = None) -> int:
    c = conn or get_db()
    cur = c.execute(
        """INSERT INTO episodes
           (ts, symbol, timeframe, direction, entry, sl, tp, ticket,
            score, setup_type, regime, session, reasoning, macro_ctx)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ep.get("ts", _now()), ep.get("symbol", ""), ep.get("timeframe", ""),
         ep.get("direction", ""), ep.get("entry", 0.0), ep.get("sl"),
         ep.get("tp"), ep.get("ticket"), ep.get("score"),
         ep.get("setup_type"), ep.get("regime"), ep.get("session"),
         ep.get("reasoning"), ep.get("macro_ctx")),
    )
    c.commit()
    return cur.lastrowid


def update_episode_result(episode_id: int, exit_price: float, pnl: float,
                           result: str, lesson: Optional[str],
                           conn: sqlite3.Connection = None):
    c = conn or get_db()
    c.execute(
        "UPDATE episodes SET exit_price=?, pnl=?, result=?, lesson=? WHERE id=?",
        (exit_price, pnl, result, lesson, episode_id),
    )
    c.commit()


def query_similar_episodes(symbol: str, setup_type: str, regime: str,
                            n: int = 10, conn: sqlite3.Connection = None) -> list:
    c = conn or get_db()
    rows = c.execute(
        """SELECT * FROM episodes
           WHERE symbol=? AND setup_type=? AND regime=? AND result IS NOT NULL
           ORDER BY id DESC LIMIT ?""",
        (symbol, setup_type, regime, n),
    ).fetchall()
    return [dict(r) for r in rows]


def get_setup_stats(conn: sqlite3.Connection = None) -> dict:
    c = conn or get_db()
    rows = c.execute(
        """SELECT setup_type, result FROM episodes WHERE result IS NOT NULL"""
    ).fetchall()
    data: dict = {}
    for r in rows:
        st = r["setup_type"] or "unknown"
        if st not in data:
            data[st] = {"wins": 0, "total": 0}
        data[st]["total"] += 1
        if r["result"] == "WIN":
            data[st]["wins"] += 1
    return {
        k: {"win_rate": v["wins"] / v["total"] * 100, "sample_size": v["total"]}
        for k, v in data.items()
    }


def get_session_stats(conn: sqlite3.Connection = None) -> dict:
    c = conn or get_db()
    rows = c.execute(
        """SELECT session, result FROM episodes WHERE result IS NOT NULL"""
    ).fetchall()
    data: dict = {}
    for r in rows:
        sess = r["session"] or "unknown"
        if sess not in data:
            data[sess] = {"wins": 0, "total": 0}
        data[sess]["total"] += 1
        if r["result"] == "WIN":
            data[sess]["wins"] += 1
    return {
        k: {"win_rate": v["wins"] / v["total"] * 100, "sample_size": v["total"]}
        for k, v in data.items()
    }


def save_lesson(lesson: dict, conn: sqlite3.Connection = None):
    c = conn or get_db()
    c.execute(
        """INSERT INTO lessons (ts, setup_type, regime, session, win_rate,
           sample_size, weight_adj, notes) VALUES (?,?,?,?,?,?,?,?)""",
        (_now(), lesson.get("setup_type", ""), lesson.get("regime"),
         lesson.get("session"), lesson.get("win_rate"), lesson.get("sample_size"),
         lesson.get("weight_adj"), lesson.get("notes")),
    )
    c.commit()


def save_research(item: dict, conn: sqlite3.Connection = None):
    c = conn or get_db()
    c.execute(
        """INSERT INTO research (ts, source, title, summary, url, relevance)
           VALUES (?,?,?,?,?,?)""",
        (_now(), item.get("source", ""), item.get("title"),
         item.get("summary"), item.get("url"), item.get("relevance", 0.0)),
    )
    c.commit()


def save_report(report: dict, conn: sqlite3.Connection = None):
    c = conn or get_db()
    c.execute(
        """INSERT OR REPLACE INTO reports
           (date, trades_total, trades_win, trades_loss, pnl_day, win_rate,
            best_setup, worst_setup, lessons_text, plan_tomorrow, goals_snapshot,
            report_text)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (report.get("date", ""), report.get("trades_total", 0),
         report.get("trades_win", 0), report.get("trades_loss", 0),
         report.get("pnl_day", 0.0), report.get("win_rate", 0.0),
         report.get("best_setup"), report.get("worst_setup"),
         report.get("lessons_text"), report.get("plan_tomorrow"),
         report.get("goals_snapshot"), report.get("report_text")),
    )
    c.commit()


def get_goals(conn: sqlite3.Connection = None) -> list:
    c = conn or get_db()
    rows = c.execute("SELECT * FROM goals ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def update_goal(goal_id: str, current_value: float, conn: sqlite3.Connection = None):
    c = conn or get_db()
    row = c.execute("SELECT target FROM goals WHERE goal_id=?", (goal_id,)).fetchone()
    if row is None:
        return
    target = row["target"]
    progress = (current_value / target * 100) if target > 0 else 0.0
    c.execute(
        """UPDATE goals SET current=?, progress_pct=?, updated_ts=? WHERE goal_id=?""",
        (current_value, progress, _now(), goal_id),
    )
    c.commit()


def seed_goals(conn: sqlite3.Connection = None):
    c = conn or get_db()
    for g in INITIAL_GOALS:
        c.execute(
            """INSERT OR IGNORE INTO goals
               (goal_id, description, metric, target, horizon, updated_ts)
               VALUES (?,?,?,?,?,?)""",
            (g["goal_id"], g["description"], g["metric"],
             g["target"], g["horizon"], _now()),
        )
    c.commit()
