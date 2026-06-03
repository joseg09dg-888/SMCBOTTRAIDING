"""Tests for score_db outcome tracking."""
import pytest
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file for each test."""
    db_file = tmp_path / "test_scores.db"
    monkeypatch.setattr("core.score_db.DB_PATH", db_file)
    yield db_file


def test_save_score_returns_id():
    from core.score_db import save_score
    row_id = save_score("BTCUSDT", "1h", 80, "long", 65000.0, 64000.0, 67000.0, True)
    assert row_id > 0


def test_get_stats_empty():
    from core.score_db import get_stats
    stats = get_stats()
    assert stats["total"] == 0
    assert stats["win_rate"] == 0.0


def test_update_score_outcome_win():
    from core.score_db import save_score, update_score_outcome, get_stats
    save_score("BTCUSDT", "1h", 90, "long", 65000.0, 64000.0, 67000.0, True)
    update_score_outcome("BTCUSDT", 65000.0, "WIN", 3.07)
    stats = get_stats()
    assert stats["wins"] == 1
    assert stats["losses"] == 0
    assert stats["win_rate"] == 100.0
    assert stats["has_real_outcomes"] is True


def test_update_score_outcome_loss():
    from core.score_db import save_score, update_score_outcome, get_stats
    save_score("ETHUSDT", "4h", 70, "short", 1900.0, 1950.0, 1800.0, True)
    update_score_outcome("ETHUSDT", 1900.0, "LOSS", -2.63)
    stats = get_stats()
    assert stats["losses"] == 1
    assert stats["win_rate"] == 0.0


def test_win_rate_with_mixed_outcomes():
    from core.score_db import save_score, update_score_outcome, get_stats
    save_score("BTCUSDT", "1h", 90, "long", 65000.0, 64000.0, 67000.0, True)
    save_score("ETHUSDT", "1h", 75, "short", 1900.0, 1950.0, 1800.0, True)
    save_score("SOLUSDT", "1h", 80, "long", 75.0, 73.0, 80.0, True)
    update_score_outcome("BTCUSDT", 65000.0, "WIN", 3.07)
    update_score_outcome("ETHUSDT", 1900.0, "LOSS", -2.63)
    update_score_outcome("SOLUSDT", 75.0, "WIN", 6.67)
    stats = get_stats()
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert abs(stats["win_rate"] - 66.67) < 0.1


def test_profit_factor():
    from core.score_db import save_score, update_score_outcome, get_stats
    save_score("BTCUSDT", "1h", 90, "long", 65000.0, 64000.0, 67000.0, True)
    save_score("ETHUSDT", "1h", 75, "short", 1900.0, 1950.0, 1800.0, True)
    update_score_outcome("BTCUSDT", 65000.0, "WIN", 3.07)
    update_score_outcome("ETHUSDT", 1900.0, "LOSS", -2.63)
    stats = get_stats()
    assert stats["profit_factor"] == pytest.approx(1.0)  # 1 win / 1 loss
