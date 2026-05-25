# tests/core/test_nightly_reporter.py
import pytest
import sqlite3
from unittest.mock import AsyncMock, MagicMock
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
        nr.mark_fired("2026-05-25")
        dt = datetime(2026, 5, 25, 22, 0, tzinfo=timezone.utc)
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
        assert "CHoCH" in report or "setup" in report.lower()

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
        asyncio.run(nr.send("2026-05-25"))

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
        assert "2" in report
        assert "1" in report

    def test_generate_and_save_idempotent(self, conn):
        _add_today_trade(conn, "WIN", 50.0)
        nr = NightlyReporter(conn=conn)
        nr.generate_and_save("2026-05-25")
        nr.generate_and_save("2026-05-25")
        rows = conn.execute(
            "SELECT COUNT(*) FROM reports WHERE date='2026-05-25'"
        ).fetchone()[0]
        assert rows == 1
