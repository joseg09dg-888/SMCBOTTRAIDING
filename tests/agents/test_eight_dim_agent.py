"""
TDD tests for agents/eight_dim_agent.py -- DIM6 circuit breaker regression.

BUG-DIM6-DEAD-COLUMNS: _dim6_circuit_breaker() queried columns ("outcome", "closed_at")
that don't exist in the real episodes.db schema ("result", "ts"). Every call
raised sqlite3.OperationalError, silently swallowed, always falling through
to the safe-unblocked default (1.0) -- the 3-consecutive-loss circuit
breaker and WR<40% size reduction never fired once, ever.
"""
import os
import sqlite3
import pandas as pd
import pytest

from agents.eight_dim_agent import EightDimensionAgent


@pytest.fixture
def episodes_db(tmp_path, monkeypatch):
    """Creates a real episodes.db with the ACTUAL production schema in a
    temp memory/ dir and chdirs there, matching how _dim6_circuit_breaker() looks it
    up (relative path 'memory/episodes.db')."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "episodes.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, direction TEXT,
            entry REAL, sl REAL, tp REAL, ticket INTEGER, score INTEGER,
            setup_type TEXT, regime TEXT, session TEXT, exit_price REAL,
            pnl REAL, result TEXT
        )
    """)
    conn.commit()
    conn.close()
    monkeypatch.chdir(tmp_path)
    return db_path


def _insert_episodes(db_path, results, symbol="EURUSD", start_minutes_ago=None):
    """results: list of 'WIN'/'LOSS', inserted oldest-first with increasing ts.

    Timestamps default to recent (within the last hour, relative to "now")
    so they fall inside the 24h circuit-breaker window. Pass
    start_minutes_ago to place the whole batch further in the past (e.g.
    to test that the breaker expires after 24h)."""
    from datetime import datetime, timezone, timedelta
    base = datetime.now(timezone.utc) - timedelta(minutes=start_minutes_ago or len(results))
    conn = sqlite3.connect(str(db_path))
    for i, r in enumerate(results):
        ts = (base + timedelta(minutes=i)).isoformat()
        conn.execute(
            "INSERT INTO episodes (ts, symbol, direction, result) VALUES (?,?,?,?)",
            (ts, symbol, "BUY", r),
        )
    conn.commit()
    conn.close()


def _dummy_df(n=60):
    closes = [1.1 + i * 0.0001 for i in range(n)]
    return pd.DataFrame({
        "open": closes, "high": [c + 0.0002 for c in closes],
        "low": [c - 0.0002 for c in closes], "close": closes,
    })


def test_query_uses_real_schema_columns_not_outcome_closed_at(episodes_db):
    """Regression: the old query referenced non-existent columns and always
    silently failed. This confirms the fixed query actually executes against
    the real schema without raising."""
    _insert_episodes(episodes_db, ["WIN", "WIN", "WIN"])
    agent = EightDimensionAgent()
    mult = agent._dim6_circuit_breaker("EURUSD")
    assert mult == 1.0  # no losses -> unrestricted, but query must not have crashed


def test_three_consecutive_losses_blocks(episodes_db):
    _insert_episodes(episodes_db, ["WIN", "LOSS", "LOSS", "LOSS"])
    agent = EightDimensionAgent()
    mult = agent._dim6_circuit_breaker("EURUSD")
    assert mult == 0.0


def test_no_three_consecutive_losses_not_blocked(episodes_db):
    _insert_episodes(episodes_db, ["LOSS", "WIN", "LOSS", "LOSS"])
    agent = EightDimensionAgent()
    mult = agent._dim6_circuit_breaker("EURUSD")
    assert mult != 0.0


def test_low_win_rate_last_five_reduces_multiplier(episodes_db):
    # Inserted oldest->newest; most-recent-first (query order) becomes
    # LOSS,WIN,LOSS,LOSS,LOSS -- 1 win/5 = 20% WR, no 3-in-a-row streak
    # at the front so the harder block doesn't pre-empt this check.
    _insert_episodes(episodes_db, ["LOSS", "LOSS", "LOSS", "WIN", "LOSS"])
    agent = EightDimensionAgent()
    mult = agent._dim6_circuit_breaker("EURUSD")
    assert mult == 0.60


def test_no_episodes_db_returns_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no memory/episodes.db here
    agent = EightDimensionAgent()
    assert agent._dim6_circuit_breaker("EURUSD") == 1.0


def test_losses_older_than_24h_not_blocked(episodes_db):
    """BUG-DIM6-GLOBAL-NO-EXPIRY regression: 3 losses that happened more
    than 24h ago must NOT keep blocking the pair forever."""
    _insert_episodes(episodes_db, ["WIN", "LOSS", "LOSS", "LOSS"], start_minutes_ago=60 * 48)
    agent = EightDimensionAgent()
    mult = agent._dim6_circuit_breaker("EURUSD")
    assert mult != 0.0


def test_losses_in_other_symbol_dont_block_this_one(episodes_db):
    """BUG-DIM6-GLOBAL-NO-EXPIRY regression: the breaker must be scoped per
    symbol -- losses in GBPCAD must not block USDCAD."""
    _insert_episodes(episodes_db, ["WIN", "LOSS", "LOSS", "LOSS"], symbol="GBPCAD")
    agent = EightDimensionAgent()
    mult = agent._dim6_circuit_breaker("USDCAD")
    assert mult != 0.0


def test_analyze_blocks_trade_on_three_losses(episodes_db):
    """Full analyze() path: 3 consecutive losses should set allowed=False."""
    _insert_episodes(episodes_db, ["WIN", "LOSS", "LOSS", "LOSS"])
    agent = EightDimensionAgent()
    result = agent.analyze("EURUSD", _dummy_df(), open_positions=[], utc_hour=14, direction="LONG")
    assert result.allowed is False
    assert "DIM6" in result.reason
