# Autonomous Mode 24/7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the SMC bot into a self-improving 24/7 autonomous agent with episodic memory, continuous learning, research ingestion, autonomous goals, and nightly reports.

**Architecture:** Six parallel asyncio loops wired into `supervisor.py` via `asyncio.gather()`. All state persisted in SQLite WAL (`memory/episodes.db`). Claude API enhances reasoning before each trade, falling back to base score on failure. All modules are exception-isolated — a crash in one loop never stops others.

**Tech Stack:** Python 3.12, SQLite 3 (WAL mode), asyncio, httpx (already installed), anthropic SDK (already installed), MetaTrader5 (already installed).

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `memory/episodic_db.py` | **CREATE** | SQLite WAL foundation — all persistence for episodes, lessons, goals, research, reports |
| `core/autonomous_learner.py` | **CREATE** | Hourly trade history analysis → adjust DecisionFilter weights |
| `core/research_agent.py` | **CREATE** | Every 2h fetch arXiv + MQL5 articles, relevance score, save |
| `core/goals_manager.py` | **CREATE** | Seed 5 default goals, daily evaluation from episodes |
| `core/nightly_reporter.py` | **CREATE** | 22:00 UTC — generate + send Telegram summary |
| `agents/analysis_agent.py` | **MODIFY** | Add `reason_with_context()` Claude reasoning method |
| `core/supervisor.py` | **MODIFY** | Wire 4 new loops + episode recording in `_send_mt5_real_order` |
| `tests/memory/test_episodic_db.py` | **CREATE** | ~35 tests for all DB functions |
| `tests/core/test_autonomous_learner.py` | **CREATE** | ~25 tests for weight adjustment logic |
| `tests/core/test_research_agent.py` | **CREATE** | ~20 tests for fetch + relevance + fallback |
| `tests/core/test_goals_manager.py` | **CREATE** | ~20 tests for seeding + evaluation |
| `tests/core/test_nightly_reporter.py` | **CREATE** | ~20 tests for report generation |
| `tests/agents/test_reasoning_prompt.py` | **CREATE** | ~15 tests for prompt build + JSON parse + fallback |

---

## Task 1: Foundation — `memory/episodic_db.py`

**Files:**
- Create: `memory/episodic_db.py`
- Create: `tests/memory/test_episodic_db.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they all fail**

```
cd C:\Users\jose-\projects\trading_agent
.venv\Scripts\python -m pytest tests/memory/test_episodic_db.py -x -q 2>&1
```
Expected: `ModuleNotFoundError: No module named 'memory.episodic_db'`

- [ ] **Step 3: Implement `memory/episodic_db.py`**

```python
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
    if path != ":memory:":
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
```

- [ ] **Step 4: Run tests**

```
.venv\Scripts\python -m pytest tests/memory/test_episodic_db.py -v 2>&1
```
Expected: All 35 tests PASS.

- [ ] **Step 5: Commit**

```
git add memory/episodic_db.py tests/memory/test_episodic_db.py
git commit -m "feat: episodic_db — SQLite WAL foundation for autonomous mode"
```

---

## Task 2: `core/autonomous_learner.py`

**Files:**
- Create: `core/autonomous_learner.py`
- Create: `tests/core/test_autonomous_learner.py`

- [ ] **Step 1: Write failing tests**

```python
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
        # verify it logged something
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
```

- [ ] **Step 2: Run to verify failure**

```
.venv\Scripts\python -m pytest tests/core/test_autonomous_learner.py -x -q 2>&1
```
Expected: `ModuleNotFoundError: No module named 'core.autonomous_learner'`

- [ ] **Step 3: Implement `core/autonomous_learner.py`**

```python
# core/autonomous_learner.py
import sqlite3
from typing import Optional
from memory.episodic_db import get_db, get_setup_stats, save_lesson


class AutonomousLearner:
    """
    Every 60 minutes, analyze real trade history and auto-adjust
    DecisionFilter weights per (setup_type, regime, session) group.

    Weight rules:
      win_rate > 65%  → weight_adj = 1.20
      50-65%          → weight_adj = 1.00 (neutral)
      < 50%           → weight_adj = 0.80
      < 35% + 10+     → weight_adj = 0.50 (near-disable)

    Groups with fewer than 5 samples are skipped.
    """

    def __init__(self, conn: sqlite3.Connection = None):
        self._conn = conn
        self._weights: dict = {}

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn or get_db()

    def run_analysis(self) -> dict:
        conn = self._get_conn()
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
            if total < 5:
                continue
            win_rate = data["wins"] / total
            if win_rate > 0.65:
                adj = 1.20
            elif win_rate >= 0.50:
                adj = 1.00
            elif win_rate >= 0.35 or total < 10:
                adj = 0.80
            else:
                adj = 0.50

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
        return max(0, int(base_threshold / adj))
```

- [ ] **Step 4: Run tests**

```
.venv\Scripts\python -m pytest tests/core/test_autonomous_learner.py -v 2>&1
```
Expected: All 25 tests PASS.

- [ ] **Step 5: Commit**

```
git add core/autonomous_learner.py tests/core/test_autonomous_learner.py
git commit -m "feat: autonomous_learner — hourly weight adjustment from trade history"
```

---

## Task 3: `core/research_agent.py`

**Files:**
- Create: `core/research_agent.py`
- Create: `tests/core/test_research_agent.py`

- [ ] **Step 1: Write failing tests**

```python
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
        assert 0.3 <= score <= 0.8

    def test_empty_string_returns_zero(self):
        assert _score_relevance("") == 0.0

    def test_ict_keyword_scores_high(self):
        assert _score_relevance("ICT institutional order flow") > 0.7


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
        assert any("SMC" in item["title"] or "Order Block" in item["title"]
                   for item in items)

    def test_fetch_arxiv_filters_low_relevance(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ARXIV_XML
        with patch("httpx.get", return_value=mock_resp):
            items = agent._fetch_arxiv()
        # Climate paper should be filtered out (relevance < 0.4)
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

    def test_fetch_mql5_parses_articles(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MQL5_HTML
        with patch("httpx.get", return_value=mock_resp):
            items = agent._fetch_mql5()
        assert len(items) >= 1

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
            agent.run_cycle()  # must not raise

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
        assert rows < 10  # not quadrupled

    def test_relevance_threshold_0_4(self, conn):
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
        rows = conn.execute(
            "SELECT DISTINCT source FROM research"
        ).fetchall()
        sources = [r["source"] for r in rows]
        assert "arxiv" in sources

    def test_mql5_source_label(self, conn):
        agent = ResearchAgent(conn=conn)
        mock_arxiv = MagicMock(status_code=503)
        mock_mql5 = MagicMock(status_code=200, text=MQL5_HTML)
        def side_effect(url, **kwargs):
            if "arxiv" in url:
                return mock_arxiv
            return mock_mql5
        with patch("httpx.get", side_effect=side_effect):
            agent.run_cycle()
        rows = conn.execute(
            "SELECT source FROM research"
        ).fetchall()
        if rows:
            assert any(r["source"] == "mql5" for r in rows)
```

- [ ] **Step 2: Run to verify failure**

```
.venv\Scripts\python -m pytest tests/core/test_research_agent.py -x -q 2>&1
```
Expected: `ModuleNotFoundError: No module named 'core.research_agent'`

- [ ] **Step 3: Implement `core/research_agent.py`**

```python
# core/research_agent.py
import re
import sqlite3
from typing import List, Optional
from memory.episodic_db import get_db, save_research

SMC_KEYWORDS = [
    "smart money", "smc", "order block", "ict", "liquidity",
    "fair value gap", "fvg", "choch", "bos", "market structure",
    "institutional", "imbalance", "sweep", "mitigation",
]

ARXIV_URL = (
    "https://export.arxiv.org/api/query"
    "?search_query=cat:q-fin.TR+AND+(SMC+OR+order+block+OR+ICT+OR+liquidity)"
    "&sortBy=submittedDate&sortOrder=descending&max_results=10"
)
MQL5_URL = "https://www.mql5.com/en/articles"


def _score_relevance(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in SMC_KEYWORDS if kw in text_lower)
    return min(1.0, hits * 0.15)


def _already_saved(url: str, conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT id FROM research WHERE url=?", (url,)).fetchone()
    return row is not None


class ResearchAgent:
    """
    Every 2 hours: fetch up to 5 new items from arXiv and MQL5,
    score relevance, and save those with relevance > 0.4 to research table.
    """

    RELEVANCE_THRESHOLD = 0.4

    def __init__(self, conn: sqlite3.Connection = None):
        self._conn = conn

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn or get_db()

    def _fetch_arxiv(self) -> List[dict]:
        try:
            import httpx
            resp = httpx.get(ARXIV_URL, timeout=10)
            if resp.status_code != 200:
                return []
            entries = re.findall(
                r"<entry>.*?<title>(.*?)</title>.*?<summary>(.*?)</summary>"
                r".*?<id>(http[^<]+)</id>",
                resp.text,
                re.DOTALL,
            )
            items = []
            for title, summary, url in entries[:5]:
                title   = re.sub(r"\s+", " ", title.strip())
                summary = re.sub(r"\s+", " ", summary.strip())[:500]
                score   = _score_relevance(f"{title} {summary}")
                if score >= self.RELEVANCE_THRESHOLD:
                    items.append({"source": "arxiv", "title": title,
                                  "summary": summary, "url": url.strip(),
                                  "relevance": score})
            return items
        except Exception as e:
            print(f"[RESEARCH] arxiv unavailable: {e}", flush=True)
            return []

    def _fetch_mql5(self) -> List[dict]:
        try:
            import httpx
            resp = httpx.get(MQL5_URL, timeout=10)
            if resp.status_code != 200:
                return []
            links = re.findall(
                r'href="(/articles/\d+)"[^>]*>(.*?)</a>', resp.text
            )
            items = []
            for href, title in links[:5]:
                title = re.sub(r"<[^>]+>", "", title).strip()
                if not title:
                    continue
                score = _score_relevance(title)
                if score >= self.RELEVANCE_THRESHOLD:
                    url = f"https://www.mql5.com{href}"
                    items.append({"source": "mql5", "title": title,
                                  "summary": f"MQL5 article: {title}",
                                  "url": url, "relevance": score})
            return items
        except Exception as e:
            print(f"[RESEARCH] mql5 unavailable: {e}", flush=True)
            return []

    def run_cycle(self):
        conn = self._get_conn()
        for source_items in [self._fetch_arxiv(), self._fetch_mql5()]:
            for item in source_items:
                if not _already_saved(item["url"], conn):
                    save_research(item, conn=conn)
                    print(
                        f"[RESEARCH] {item['source']} saved: {item['title'][:60]}",
                        flush=True,
                    )

    def get_top_research(self, n: int = 3, conn: sqlite3.Connection = None) -> list:
        c = conn or self._get_conn()
        rows = c.execute(
            "SELECT * FROM research WHERE applied=0 ORDER BY relevance DESC, id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests**

```
.venv\Scripts\python -m pytest tests/core/test_research_agent.py -v 2>&1
```
Expected: All 15 tests PASS (some MQL5 ones may be 0-item results depending on HTML regex match).

- [ ] **Step 5: Commit**

```
git add core/research_agent.py tests/core/test_research_agent.py
git commit -m "feat: research_agent — arXiv + MQL5 every 2h with relevance scoring"
```

---

## Task 4: `core/goals_manager.py`

**Files:**
- Create: `core/goals_manager.py`
- Create: `tests/core/test_goals_manager.py`

- [ ] **Step 1: Write failing tests**

```python
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
        assert row["progress_pct"] > 100.0  # 70% > 65% target

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
        assert "Win rate" in summary or "winrate" in summary.lower()

    def test_evaluate_no_crash_with_empty_db(self, conn):
        gm = GoalsManager(conn=conn)
        result = gm.evaluate()  # must not raise
        assert isinstance(result, dict)

    def test_win_rate_pct_100_uses_last_100_closed(self, conn):
        _add_closed(conn, 60, 40)  # 60% win rate over 100 trades
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

    def test_progress_pct_capped_at_reasonable_value(self, conn):
        _add_closed(conn, 100, 0)  # 100% win rate
        gm = GoalsManager(conn=conn)
        gm.evaluate()
        row = conn.execute(
            "SELECT progress_pct FROM goals WHERE goal_id='winrate_65'"
        ).fetchone()
        assert row["progress_pct"] <= 200.0

    def test_get_goals_snapshot_json(self, conn):
        gm = GoalsManager(conn=conn)
        snapshot = gm.get_goals_snapshot()
        import json
        data = json.loads(snapshot)
        assert isinstance(data, list)
        assert len(data) == 5
```

- [ ] **Step 2: Run to verify failure**

```
.venv\Scripts\python -m pytest tests/core/test_goals_manager.py -x -q 2>&1
```
Expected: `ModuleNotFoundError: No module named 'core.goals_manager'`

- [ ] **Step 3: Implement `core/goals_manager.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```
.venv\Scripts\python -m pytest tests/core/test_goals_manager.py -v 2>&1
```
Expected: All 15 tests PASS.

- [ ] **Step 5: Commit**

```
git add core/goals_manager.py tests/core/test_goals_manager.py
git commit -m "feat: goals_manager — 5 autonomous goals tracked from episode history"
```

---

## Task 5: `core/nightly_reporter.py`

**Files:**
- Create: `core/nightly_reporter.py`
- Create: `tests/core/test_nightly_reporter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_nightly_reporter.py
import pytest
import sqlite3
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timezone
from memory.episodic_db import _create_tables, record_episode, update_episode_result, seed_goals
from core.nightly_reporter import NightlyReporter


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _create_tables(c)
    seed_goals(conn=c)
    return c


def _add_today_trade(conn, result="WIN", pnl=50.0):
    date_str = "2026-05-25"
    eid = record_episode({
        "ts": f"{date_str}T10:00:00Z",
        "symbol": "USDJPY", "timeframe": "H4", "direction": "BUY",
        "entry": 155.5, "score": 70, "setup_type": "CHoCH+OB",
    }, conn=conn)
    update_episode_result(eid, 156.0, pnl, result, "lesson text", conn=conn)
    return eid


class TestNightlyReporter:
    def test_instantiates(self, conn):
        nr = NightlyReporter(conn=conn)
        assert nr is not None

    def test_generate_report_returns_str(self, conn):
        _add_today_trade(conn)
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert isinstance(report, str)
        assert len(report) > 50

    def test_report_includes_trade_count(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        _add_today_trade(conn, "LOSS", -30.0)
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert "2" in report

    def test_report_includes_pnl(self, conn):
        _add_today_trade(conn, "WIN", 75.0)
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert "75" in report

    def test_report_includes_win_rate(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert "100" in report or "%" in report

    def test_report_empty_data_fallback(self, conn):
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert isinstance(report, str)
        assert "Sin trades" in report or "0" in report

    def test_should_fire_returns_true_at_22(self, conn):
        nr = NightlyReporter(conn=conn)
        dt = datetime(2026, 5, 25, 22, 0, tzinfo=timezone.utc)
        assert nr.should_fire(dt) is True

    def test_should_fire_false_at_other_hours(self, conn):
        nr = NightlyReporter(conn=conn)
        for hour in [0, 10, 21, 23]:
            dt = datetime(2026, 5, 25, hour, 0, tzinfo=timezone.utc)
            assert nr.should_fire(dt) is False

    def test_should_fire_false_after_minute_2(self, conn):
        nr = NightlyReporter(conn=conn)
        dt = datetime(2026, 5, 25, 22, 5, tzinfo=timezone.utc)
        assert nr.should_fire(dt) is False

    def test_should_fire_not_twice_same_day(self, conn):
        nr = NightlyReporter(conn=conn)
        dt = datetime(2026, 5, 25, 22, 0, tzinfo=timezone.utc)
        nr.mark_fired("2026-05-25")
        assert nr.should_fire(dt) is False

    def test_save_report_persists_to_db(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        nr = NightlyReporter(conn=conn)
        nr.generate_and_save("2026-05-25")
        row = conn.execute("SELECT * FROM reports WHERE date='2026-05-25'").fetchone()
        assert row is not None

    def test_generate_report_includes_best_setup(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert "CHoCH" in report or "best" in report.lower()

    def test_multiple_dates_isolated(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        eid = record_episode({
            "ts": "2026-05-24T10:00:00Z",
            "symbol": "EURUSD", "timeframe": "H1", "direction": "SELL",
            "entry": 1.08, "score": 65, "setup_type": "FVG",
        }, conn=conn)
        update_episode_result(eid, 1.075, 30.0, "WIN", None, conn=conn)

        nr = NightlyReporter(conn=conn)
        report_25 = nr.generate_report("2026-05-25")
        report_24 = nr.generate_report("2026-05-24")
        assert report_25 != report_24

    def test_send_calls_telegram(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        nr = NightlyReporter(conn=conn, telegram_bot=mock_bot, chat_id="123")
        import asyncio
        asyncio.run(nr.send("2026-05-25"))
        mock_bot.send_message.assert_called_once()

    def test_send_no_crash_without_telegram(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        nr = NightlyReporter(conn=conn)
        import asyncio
        asyncio.run(nr.send("2026-05-25"))  # must not raise

    def test_format_contains_header(self, conn):
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert "REPORTE" in report.upper() or "2026-05-25" in report

    def test_win_loss_breakdown_in_report(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        _add_today_trade(conn, "WIN", 40.0)
        _add_today_trade(conn, "LOSS", -25.0)
        nr = NightlyReporter(conn=conn)
        report = nr.generate_report("2026-05-25")
        assert "2" in report  # 2 wins
        assert "1" in report  # 1 loss

    def test_generate_and_save_idempotent(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        nr = NightlyReporter(conn=conn)
        nr.generate_and_save("2026-05-25")
        nr.generate_and_save("2026-05-25")
        rows = conn.execute("SELECT COUNT(*) FROM reports WHERE date='2026-05-25'").fetchone()[0]
        assert rows == 1
```

- [ ] **Step 2: Run to verify failure**

```
.venv\Scripts\python -m pytest tests/core/test_nightly_reporter.py -x -q 2>&1
```
Expected: `ModuleNotFoundError: No module named 'core.nightly_reporter'`

- [ ] **Step 3: Implement `core/nightly_reporter.py`**

```python
# core/nightly_reporter.py
import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from memory.episodic_db import get_db, save_report, get_goals
from core.goals_manager import GoalsManager


class NightlyReporter:
    """
    Generates and optionally sends nightly Telegram reports at 22:00 UTC.
    Reads from episodic DB: today's trades, lessons, goals, top research.
    """

    def __init__(self, conn: sqlite3.Connection = None,
                 telegram_bot=None, chat_id: str = None):
        self._conn = conn
        self._bot = telegram_bot
        self._chat_id = chat_id
        self._fired_dates: set = set()

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn or get_db()

    def should_fire(self, now: datetime = None) -> bool:
        dt  = now or datetime.now(timezone.utc)
        key = dt.strftime("%Y-%m-%d")
        return dt.hour == 22 and dt.minute < 2 and key not in self._fired_dates

    def mark_fired(self, date_str: str):
        self._fired_dates.add(date_str)

    def generate_report(self, date: str) -> str:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT symbol, setup_type, pnl, result, lesson
               FROM episodes WHERE ts LIKE ? AND result != 'OPEN'""",
            (f"{date}%",),
        ).fetchall()

        if not rows:
            return (
                f"<b>REPORTE AUTONOMO - {date}</b>\n"
                f"Sin trades hoy. Bot en espera de setup."
            )

        wins   = sum(1 for r in rows if r["result"] == "WIN")
        losses = sum(1 for r in rows if r["result"] == "LOSS")
        total  = wins + losses
        pnl    = sum(r["pnl"] or 0 for r in rows)
        wr     = (wins / total * 100) if total > 0 else 0.0

        setup_pnl: dict = {}
        for r in rows:
            st = r["setup_type"] or "unknown"
            setup_pnl.setdefault(st, 0)
            setup_pnl[st] += r["pnl"] or 0
        best_setup  = max(setup_pnl, key=setup_pnl.get) if setup_pnl else "-"
        worst_setup = min(setup_pnl, key=setup_pnl.get) if setup_pnl else "-"

        lessons = [r["lesson"] for r in rows if r["lesson"]]
        lesson1 = lessons[0] if lessons else "Continuar en demo"
        lesson2 = lessons[1] if len(lessons) > 1 else "Monitorear regimen"

        gm      = GoalsManager(conn=conn)
        metrics = gm.evaluate()
        wr_prog = min(metrics.get("win_rate_pct_100", 0), 100)
        edge    = metrics.get("axi_edge_score", 0)

        research_row = conn.execute(
            "SELECT title FROM research ORDER BY id DESC LIMIT 1"
        ).fetchone()
        research_title = research_row["title"] if research_row else "Sin novedades"

        text = (
            f"<b>REPORTE AUTONOMO - {date}</b>\n"
            f"- - - - - - - - - - - -\n"
            f"Trades: {total} ({wins}W / {losses}L) - {wr:.1f}%\n"
            f"P&L: {pnl:+.2f} USD\n"
            f"- - - - - - - - - - - -\n"
            f"<b>Lecciones:</b>\n"
            f"- {lesson1}\n"
            f"- {lesson2}\n"
            f"<b>Plan manana:</b>\n"
            f"- Mejor setup: {best_setup}\n"
            f"- Evitar: {worst_setup}\n"
            f"<b>Metas:</b>\n"
            f"- Win rate 65%: {wr_prog:.0f}% completado\n"
            f"- Edge Score Axi: {edge:.1f}/50\n"
            f"<b>Nuevo:</b> {research_title[:80]}\n"
            f"- - - - - - - - - - - -"
        )

        save_report({
            "date": date, "trades_total": total, "trades_win": wins,
            "trades_loss": losses, "pnl_day": pnl, "win_rate": wr,
            "best_setup": best_setup, "worst_setup": worst_setup,
            "lessons_text": " | ".join(lessons[:3]),
            "plan_tomorrow": f"Focus {best_setup}",
            "goals_snapshot": gm.get_goals_snapshot(),
            "report_text": text,
        }, conn=conn)
        return text

    def generate_and_save(self, date: str) -> str:
        return self.generate_report(date)

    async def send(self, date: str):
        report = self.generate_report(date)
        if self._bot and self._chat_id:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=report,
                parse_mode="HTML",
            )
        else:
            print(f"[NIGHTLY] Report ready (no Telegram configured):\n{report[:200]}",
                  flush=True)
```

- [ ] **Step 4: Run tests**

```
.venv\Scripts\python -m pytest tests/core/test_nightly_reporter.py -v 2>&1
```
Expected: All 18 tests PASS.

- [ ] **Step 5: Commit**

```
git add core/nightly_reporter.py tests/core/test_nightly_reporter.py
git commit -m "feat: nightly_reporter — 22:00 UTC autonomous Telegram reports"
```

---

## Task 6: Enhanced Reasoning — modify `agents/analysis_agent.py`

**Files:**
- Modify: `agents/analysis_agent.py`
- Create: `tests/agents/test_reasoning_prompt.py`

- [ ] **Step 1: Read current end of analysis_agent.py to find append point**

```
.venv\Scripts\python -c "
with open('agents/analysis_agent.py') as f:
    lines = f.readlines()
print(f'Total lines: {len(lines)}')
print('Last 20 lines:')
for i, l in enumerate(lines[-20:], len(lines)-19):
    print(i, repr(l))
"
```

- [ ] **Step 2: Write failing tests**

```python
# tests/agents/test_reasoning_prompt.py
import json
import pytest
from unittest.mock import patch, MagicMock
import sqlite3
from memory.episodic_db import _create_tables, record_episode, update_episode_result
from agents.analysis_agent import SMCAnalysisAgent


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _create_tables(c)
    return c


def _mock_claude(json_body: dict):
    mock_content = MagicMock()
    mock_content.text = json.dumps(json_body)
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


def _add_episodes(conn, wins=3, losses=1):
    for i in range(wins + losses):
        eid = record_episode({
            "ts": "2026-05-25T10:00:00Z", "symbol": "USDJPY",
            "timeframe": "H4", "direction": "BUY", "entry": 155.5,
            "setup_type": "CHoCH+OB", "regime": "trending",
        }, conn=conn)
        update_episode_result(eid, 156.0, 50.0, "WIN" if i < wins else "LOSS",
                              None, conn=conn)


GOOD_JSON = {
    "smart_money_action": "Accumulating long positions",
    "historical_support": "supports",
    "regime_fit": "favorable",
    "lesson_applied": None,
    "decision": "LONG",
    "confidence": 80,
    "justification": "Strong CHoCH with OB, trending regime",
}

CONTRADICTS_JSON = {
    "smart_money_action": "Distributing",
    "historical_support": "contradicts",
    "regime_fit": "unfavorable",
    "lesson_applied": None,
    "decision": "WAIT",
    "confidence": 35,
    "justification": "Historical losses dominate this setup",
}


class TestReasonWithContext:
    def test_reason_with_context_returns_dict(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude(GOOD_JSON)
        episodes = [{"symbol": "USDJPY", "direction": "BUY",
                     "setup_type": "CHoCH+OB", "result": "WIN",
                     "pnl": 50.0, "ts": "2026-05-25"}]
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH confirmed at 155.50",
            similar_episodes=episodes,
            regime="trending", base_score=65,
        )
        assert isinstance(result, dict)
        assert "confidence" in result

    def test_reason_boosts_score_on_support(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude(GOOD_JSON)
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] >= 65

    def test_reason_reduces_score_on_unfavorable_regime(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude(CONTRADICTS_JSON)
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="high_vol", base_score=70,
        )
        assert result["adjusted_score"] < 70

    def test_low_confidence_triggers_wait(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "confidence": 30, "historical_support": "neutral"
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["wait_override"] is True

    def test_api_failure_returns_fallback(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.side_effect = Exception("API timeout")
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] == 65
        assert result.get("fallback") is True

    def test_contradicts_3_losses_triggers_wait(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **CONTRADICTS_JSON, "historical_support": "contradicts"
        })
        losses = [{"result": "LOSS"} for _ in range(3)]
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=losses,
            regime="trending", base_score=65,
        )
        assert result["wait_override"] is True

    def test_high_confidence_support_adds_10_pts(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "confidence": 80, "historical_support": "supports"
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] == min(100, 65 + 10)

    def test_unfavorable_regime_subtracts_15(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "regime_fit": "unfavorable", "confidence": 80
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="high_vol", base_score=70,
        )
        assert result["adjusted_score"] == max(0, 70 - 15)

    def test_json_decode_error_returns_fallback(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "NOT VALID JSON {{{"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        agent.client.messages.create.return_value = mock_response
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="trending", base_score=65,
        )
        assert result["adjusted_score"] == 65
        assert result.get("fallback") is True

    def test_build_prompt_includes_episodes(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        episodes = [{"symbol": "USDJPY", "direction": "BUY",
                     "setup_type": "CHoCH+OB", "result": "WIN",
                     "pnl": 50.0, "ts": "2026-05-25"}]
        prompt = agent._build_reasoning_prompt(
            smc_summary="CHoCH at 155.50",
            similar_episodes=episodes,
            regime="trending",
        )
        assert "CHoCH+OB" in prompt
        assert "WIN" in prompt

    def test_build_prompt_includes_regime(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        prompt = agent._build_reasoning_prompt(
            smc_summary="Test",
            similar_episodes=[],
            regime="high_vol",
        )
        assert "high_vol" in prompt

    def test_adjusted_score_capped_at_100(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "confidence": 80, "historical_support": "supports"
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="trending", base_score=95,
        )
        assert result["adjusted_score"] <= 100

    def test_adjusted_score_floored_at_0(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "regime_fit": "unfavorable",
            "historical_support": "contradicts", "confidence": 80
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[],
            regime="high_vol", base_score=10,
        )
        assert result["adjusted_score"] >= 0

    def test_neutral_historical_no_override(self, conn):
        agent = SMCAnalysisAgent.__new__(SMCAnalysisAgent)
        agent.client = MagicMock()
        agent.client.messages.create.return_value = _mock_claude({
            **GOOD_JSON, "historical_support": "neutral", "confidence": 80
        })
        result = agent.reason_with_context(
            symbol="USDJPY", timeframe="H4",
            smc_summary="CHoCH",
            similar_episodes=[{"result": "WIN"}, {"result": "WIN"}],
            regime="trending", base_score=65,
        )
        assert result.get("wait_override") is not True
```

- [ ] **Step 3: Run to verify failure**

```
.venv\Scripts\python -m pytest tests/agents/test_reasoning_prompt.py -x -q 2>&1
```
Expected: `AttributeError: type object 'SMCAnalysisAgent' has no attribute 'reason_with_context'`

- [ ] **Step 4: Add `reason_with_context` to `agents/analysis_agent.py`**

Append these methods to the `SMCAnalysisAgent` class (before the last closing of the file — find the exact line with Step 1 first):

```python
    # ── Autonomous reasoning ──────────────────────────────────────────────────

    REASONING_PROMPT = """Eres un trader institucional SMC. Analiza este setup y razona paso a paso.

## DATOS TECNICOS
{smc_summary}

## EPISODIOS HISTORICOS SIMILARES ({n} trades anteriores del bot)
{episodes_text}

## REGIMEN ACTUAL DE MERCADO
{regime}

## RAZONAMIENTO REQUERIDO
1. Que hace el Smart Money en este punto?
2. Los episodios historicos respaldan o contradicen este setup?
3. El regimen actual favorece esta estrategia?
4. Nivel de confianza 0-100 y por que

## RESPONDE SOLO EN JSON VALIDO:
{{"smart_money_action":"string","historical_support":"supports|contradicts|neutral","regime_fit":"favorable|neutral|unfavorable","lesson_applied":"string or null","decision":"LONG|SHORT|WAIT","confidence":0,"justification":"max 2 lines"}}"""

    def _build_reasoning_prompt(self, smc_summary: str,
                                 similar_episodes: list,
                                 regime: str) -> str:
        ep_lines = []
        for ep in similar_episodes[:10]:
            ep_lines.append(
                f"{ep.get('symbol','')} {ep.get('setup_type','')} "
                f"{ep.get('direction','')} -> {ep.get('result','')} "
                f"{ep.get('pnl',0):+.1f}pips ({ep.get('ts','')[:10]})"
            )
        episodes_text = "\n".join(ep_lines) if ep_lines else "Sin episodios previos"
        return self.REASONING_PROMPT.format(
            smc_summary=smc_summary,
            n=len(similar_episodes),
            episodes_text=episodes_text,
            regime=regime,
        )

    def reason_with_context(self, symbol: str, timeframe: str,
                             smc_summary: str, similar_episodes: list,
                             regime: str, base_score: int) -> dict:
        import json
        fallback = {"adjusted_score": base_score, "fallback": True,
                    "wait_override": False}
        try:
            prompt = self._build_reasoning_prompt(smc_summary, similar_episodes, regime)
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            data  = json.loads(raw[start:end])
        except Exception as e:
            print(f"[CLAUDE] reason_with_context fallback: {e}", flush=True)
            return fallback

        confidence = int(data.get("confidence", 50))
        hist       = data.get("historical_support", "neutral")
        regime_fit = data.get("regime_fit", "neutral")
        score      = base_score

        if confidence >= 75 and hist == "supports":
            score = min(100, score + 10)
        if regime_fit == "unfavorable":
            score = max(0, score - 15)

        wait_override = False
        if confidence < 40:
            wait_override = True
        loss_count = sum(1 for ep in similar_episodes if ep.get("result") == "LOSS")
        if hist == "contradicts" and loss_count >= 3:
            wait_override = True

        return {
            "adjusted_score": score,
            "wait_override":  wait_override,
            "confidence":     confidence,
            "decision":       data.get("decision", "WAIT"),
            "reasoning":      data,
            "fallback":       False,
        }
```

- [ ] **Step 5: Run tests**

```
.venv\Scripts\python -m pytest tests/agents/test_reasoning_prompt.py -v 2>&1
```
Expected: All 14 tests PASS.

- [ ] **Step 6: Commit**

```
git add agents/analysis_agent.py tests/agents/test_reasoning_prompt.py
git commit -m "feat: analysis_agent — Claude API reason_with_context() with historical episodes"
```

---

## Task 7: Wire supervisor.py — all loops + episode recording

**Files:**
- Modify: `core/supervisor.py`

- [ ] **Step 1: Read current asyncio.gather() call and _market_scan_loop location**

```
.venv\Scripts\python -c "
with open('core/supervisor.py') as f:
    lines = f.readlines()
for i, l in enumerate(lines, 1):
    if 'gather' in l or '_learning_loop' in l or '_nightly' in l or 'asyncio.gather' in l:
        print(i, l.rstrip())
"
```

Note the exact line numbers before editing.

- [ ] **Step 2: Read `_send_mt5_real_order` method (find exact line)**

```
.venv\Scripts\python -c "
with open('core/supervisor.py') as f:
    lines = f.readlines()
for i, l in enumerate(lines, 1):
    if '_send_mt5_real_order' in l or 'MT5 REAL' in l:
        print(i, l.rstrip())
"
```

- [ ] **Step 3: Add imports and new loop methods to supervisor.py**

Using the Write tool (NOT PowerShell), add after the existing imports block at the top of `core/supervisor.py`:

```python
# Autonomous mode imports
try:
    from memory.episodic_db import get_db, record_episode, update_episode_result, \
        query_similar_episodes, seed_goals
    from core.autonomous_learner import AutonomousLearner
    from core.research_agent import ResearchAgent
    from core.goals_manager import GoalsManager
    from core.nightly_reporter import NightlyReporter
    _AUTONOMOUS_AVAILABLE = True
except ImportError as _e:
    print(f"[AUTO] Autonomous mode modules unavailable: {_e}", flush=True)
    _AUTONOMOUS_AVAILABLE = False
```

- [ ] **Step 4: Add helper functions inside `TradingSupervisor.__init__` or as class methods**

Add `_detect_setup_type` and `_current_session` as static methods of `TradingSupervisor`:

```python
@staticmethod
def _detect_setup_type(signal_text: str) -> str:
    text = (signal_text or "").upper()
    if "CHOCH" in text and "OB" in text:
        return "CHoCH+OB"
    if "CHOCH" in text:
        return "CHoCH"
    if "BOS" in text:
        return "BOS"
    if "FVG" in text:
        return "FVG"
    if "OB" in text:
        return "OB"
    return "SMC"

@staticmethod
def _current_session() -> str:
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "overlap"
    if 17 <= hour < 21:
        return "ny"
    return "off"
```

- [ ] **Step 5: Initialize autonomous components in `__init__`**

Inside `TradingSupervisor.__init__`, after the existing initialization, add:

```python
        # Autonomous mode state
        self._current_regime = "unknown"
        self._last_glint_text = ""
        self._episode_db = None
        self._learner = None
        self._research = None
        self._goals = None
        self._reporter = None
        if _AUTONOMOUS_AVAILABLE:
            try:
                self._episode_db = get_db()
                seed_goals(conn=self._episode_db)
                self._learner  = AutonomousLearner(conn=self._episode_db)
                self._research = ResearchAgent(conn=self._episode_db)
                self._goals    = GoalsManager(conn=self._episode_db)
                self._reporter = NightlyReporter(
                    conn=self._episode_db,
                    telegram_bot=getattr(self, 'telegram_bot', None),
                    chat_id=getattr(self, 'telegram_chat_id', None),
                )
                print("[AUTO] Autonomous mode initialized", flush=True)
            except Exception as e:
                print(f"[AUTO] Init failed: {e}", flush=True)
```

- [ ] **Step 6: Add 4 new loop methods to TradingSupervisor**

```python
    async def _learning_loop(self):
        while self._running:
            try:
                if self._learner:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._learner.run_analysis)
            except Exception as e:
                print(f"[LEARN] Loop error: {e}", flush=True)
            await asyncio.sleep(3600)

    async def _research_loop(self):
        while self._running:
            try:
                if self._research:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._research.run_cycle)
            except Exception as e:
                print(f"[RESEARCH] Loop error: {e}", flush=True)
            await asyncio.sleep(7200)

    async def _goals_loop(self):
        while self._running:
            try:
                if self._goals:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._goals.evaluate)
            except Exception as e:
                print(f"[GOALS] Loop error: {e}", flush=True)
            await asyncio.sleep(86400)

    async def _nightly_report_loop(self):
        while self._running:
            try:
                if self._reporter and self._reporter.should_fire():
                    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    await self._reporter.send(date_str)
                    self._reporter.mark_fired(date_str)
            except Exception as e:
                print(f"[NIGHTLY] Loop error: {e}", flush=True)
            await asyncio.sleep(60)
```

- [ ] **Step 7: Add new loops to asyncio.gather() call**

Find the existing `asyncio.gather(` call in `TradingSupervisor.run()` (or equivalent entrypoint). Add the 4 new loops:

```python
await asyncio.gather(
    self._market_scan_loop(),
    self.glint.connect(),
    self._learning_loop(),
    self._research_loop(),
    self._goals_loop(),
    self._nightly_report_loop(),
    return_exceptions=True,
)
```

- [ ] **Step 8: Instrument `_send_mt5_real_order` with episode recording**

In the existing `_send_mt5_real_order(self, signal)` method, add BEFORE the order:

```python
            # Query similar episodes to enrich context
            similar = []
            if self._episode_db:
                try:
                    setup_type = self._detect_setup_type(
                        getattr(signal, 'analysis_text', '') or ''
                    )
                    similar = query_similar_episodes(
                        signal.symbol, setup_type, self._current_regime,
                        n=10, conn=self._episode_db,
                    )
                except Exception:
                    pass
```

And AFTER successful `result["ticket"]`:

```python
                # Record episode
                if self._episode_db:
                    try:
                        setup_type = self._detect_setup_type(
                            getattr(signal, 'analysis_text', '') or ''
                        )
                        record_episode({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "symbol": signal.symbol,
                            "timeframe": getattr(signal, 'timeframe', ''),
                            "direction": order_type,
                            "entry": result.get("price", 0.0),
                            "sl": sl_val,
                            "tp": tp_val,
                            "ticket": result["ticket"],
                            "score": signal.decision_score,
                            "setup_type": setup_type,
                            "regime": self._current_regime,
                            "session": self._current_session(),
                        }, conn=self._episode_db)
                    except Exception as _ep_err:
                        print(f"[EPISODE] record failed: {_ep_err}", flush=True)
```

- [ ] **Step 9: Add episode closer at end of `_market_scan_loop` cycle**

At the end of each scan cycle (after scanning all symbols), add:

```python
                # Close open episodes whose MT5 position was closed
                if self._episode_db and self._mt5_available:
                    try:
                        await self._close_resolved_episodes()
                    except Exception:
                        pass
```

And implement `_close_resolved_episodes`:

```python
    async def _close_resolved_episodes(self):
        open_eps = self._episode_db.execute(
            "SELECT id, ticket FROM episodes WHERE result IS NULL AND ticket IS NOT NULL"
        ).fetchall()
        if not open_eps:
            return
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(
            None, lambda: self.mt5.get_positions() or []
        )
        open_tickets = {p.get("ticket") for p in positions if isinstance(p, dict)}
        for ep in open_eps:
            if ep["ticket"] not in open_tickets:
                # Position closed — fetch last known tick as approximate exit
                try:
                    history = await loop.run_in_executor(
                        None, lambda: self.mt5.get_closed_order(ep["ticket"])
                    )
                    exit_price = history.get("price", 0.0) if history else 0.0
                    pnl        = history.get("profit", 0.0) if history else 0.0
                    result     = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")
                    lesson     = f"{'WIN' if pnl>0 else 'LOSS'} on auto-close"
                    update_episode_result(ep["id"], exit_price, pnl,
                                          result, lesson, conn=self._episode_db)
                except Exception as e:
                    print(f"[EPISODE] close failed for #{ep['ticket']}: {e}", flush=True)
```

Note: `self.mt5.get_positions()` and `self.mt5.get_closed_order()` may need to be added to `connectors/metatrader_connector.py` if they don't exist. Check with:

```
.venv\Scripts\python -c "
with open('connectors/metatrader_connector.py') as f:
    for i, l in enumerate(f, 1):
        if 'get_position' in l or 'get_closed' in l or 'positions' in l.lower():
            print(i, l.rstrip())
"
```

If missing, add to `metatrader_connector.py`:

```python
def get_positions(self) -> list:
    import MetaTrader5 as mt5
    positions = mt5.positions_get()
    if positions is None:
        return []
    return [{"ticket": p.ticket, "symbol": p.symbol,
             "profit": p.profit, "price_open": p.price_open} for p in positions]

def get_closed_order(self, ticket: int) -> dict:
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta, timezone
    now   = datetime.now(timezone.utc)
    start = now - timedelta(days=7)
    deals = mt5.history_deals_get(start, now, group="*")
    if deals is None:
        return {}
    for d in deals:
        if d.order == ticket:
            return {"ticket": d.order, "price": d.price, "profit": d.profit}
    return {}
```

- [ ] **Step 10: Run full test suite**

```
.venv\Scripts\python -m pytest -x -q 2>&1 | tail -15
```
Expected: All 1069 + ~135 new tests PASS (~1200+ total). Fix any failures before proceeding.

- [ ] **Step 11: Commit**

```
git add core/supervisor.py connectors/metatrader_connector.py
git commit -m "feat: supervisor — wire 4 autonomous loops + episodic recording per MT5 order"
```

---

## Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add autonomous mode section to CLAUDE.md**

Find the existing `## Estado actual del bot` or similar section in CLAUDE.md and add:

```markdown
## Autonomous Mode (2026-05-25)
- episodic_db: memory/episodes.db (SQLite WAL)
- learning loop: every 60min — core/autonomous_learner.py
- research loop: every 2h — core/research_agent.py (arXiv + MQL5)
- goals loop: every 24h — core/goals_manager.py (5 goals)
- nightly report: 22:00 UTC — core/nightly_reporter.py
- Claude reasoning: agents/analysis_agent.py reason_with_context()
- Episode recording: every real MT5 order in _send_mt5_real_order()
- Threshold: DEMO_SCORE_THRESHOLD=30, REAL=60, PREMIUM=90
```

- [ ] **Step 2: Commit**

```
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — autonomous mode 6 subsystems active"
```

---

## Task 9: Final validation + PM2 restart + push

- [ ] **Step 1: Run full test suite**

```
.venv\Scripts\python -m pytest -q 2>&1 | tail -10
```
Expected: green, 1069+ tests.

- [ ] **Step 2: Restart PM2 smc-bot**

```
pm2 restart smc-bot && pm2 logs smc-bot --lines 30
```
Expected: `[AUTO] Autonomous mode initialized` in logs.

- [ ] **Step 3: Verify 4 loops started**

Wait 10 seconds after restart, then:
```
pm2 logs smc-bot --lines 50 2>&1 | grep -E "AUTO|LEARN|RESEARCH|GOALS|NIGHTLY"
```
Expected: `[AUTO] Autonomous mode initialized` and at least one `[LEARN]`, `[RESEARCH]` line.

- [ ] **Step 4: Push to GitHub**

```
git push origin main
```
Expected: push succeeds, all commits visible on `github.com/joseg09dg-888/SMCBOTTRAIDING`.

---

## Self-Review Checklist

**Spec coverage:**
- [x] episodic_db.py: 5 tables, all 12 public functions → Task 1
- [x] autonomous_learner.py: hourly analysis, weight rules, threshold adj → Task 2
- [x] research_agent.py: arXiv + MQL5, relevance 0.4 threshold, fallback → Task 3
- [x] goals_manager.py: 5 goals, evaluate(), format_goals_summary() → Task 4
- [x] nightly_reporter.py: 22:00 UTC, should_fire(), send(), fallback → Task 5
- [x] reason_with_context(): prompt, JSON parse, score rules, fallback → Task 6
- [x] supervisor.py: 4 new loops in gather(), episode recording, closer → Task 7
- [x] CLAUDE.md update → Task 8
- [x] Full pytest + PM2 restart + git push → Task 9

**Placeholder scan:** No TBD or TODO — all steps have actual code.

**Type consistency:**
- `record_episode(ep, conn)` — consistent in Tasks 1, 7
- `query_similar_episodes(symbol, setup_type, regime, n, conn)` — consistent in Tasks 1, 7
- `reason_with_context(symbol, timeframe, smc_summary, similar_episodes, regime, base_score)` — consistent in Tasks 6
- `AutonomousLearner(conn)` — consistent Tasks 2, 7
- `NightlyReporter(conn, telegram_bot, chat_id)` — consistent Tasks 5, 7
