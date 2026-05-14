# tests/agents/test_report_agent.py

from datetime import date, datetime, timezone
import pytest
from agents.report_agent import (
    ReportAgent, TradeRecord, WeeklyStats, MonthlyStats,
    GoLiveCriteria
)


# Helper
def make_trade(symbol="BTCUSDT", pnl=10.0, timestamp=None, direction="long"):
    return TradeRecord(
        symbol=symbol, direction=direction, entry=67000.0,
        exit_price=67000.0 + pnl, pnl=pnl,
        agents_confirmed=["SignalAgent"], setup_tags=["OB"],
        timeframe="1h", score=75,
        timestamp=timestamp or datetime(2026, 5, 12, 10, 0, tzinfo=timezone.utc)
    )


# ── Metricas ───────────────────────────────────────────────────────────────

def test_add_trade_and_count():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade())
    assert len(agent._trades) == 1


def test_calculate_weekly_stats_basic():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=50.0, timestamp=datetime(2026, 5, 12, 10, 0, tzinfo=timezone.utc)))
    agent.add_trade(make_trade(pnl=-20.0, timestamp=datetime(2026, 5, 13, 10, 0, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))  # semana Mon 11 May
    assert stats.total_trades == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert abs(stats.pnl - 30.0) < 0.01


def test_weekly_win_rate():
    agent = ReportAgent(capital=1000.0)
    for _ in range(3):
        agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    agent.add_trade(make_trade(pnl=-5.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    assert abs(stats.win_rate - 75.0) < 0.01


def test_weekly_profit_factor_no_losses():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    # No losses -> profit_factor == sum(wins)
    assert stats.profit_factor == pytest.approx(10.0)


def test_weekly_profit_factor_with_losses():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=30.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    agent.add_trade(make_trade(pnl=-10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    assert stats.profit_factor == pytest.approx(3.0)


def test_weekly_best_worst_trade():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=50.0, symbol="BTCUSDT", timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    agent.add_trade(make_trade(pnl=-20.0, symbol="EURUSD", timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    assert stats.best_trade.symbol == "BTCUSDT"
    assert stats.worst_trade.symbol == "EURUSD"


def test_weekly_empty_week():
    agent = ReportAgent(capital=1000.0)
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    assert stats.total_trades == 0
    assert stats.win_rate == 0.0


def test_monthly_stats_basic():
    agent = ReportAgent(capital=1000.0)
    for i in range(5):
        agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 10 + i, tzinfo=timezone.utc)))
    agent.add_trade(make_trade(pnl=-15.0, timestamp=datetime(2026, 5, 14, tzinfo=timezone.utc)))
    stats = agent.calculate_monthly_stats(2026, 5)
    assert stats.total_trades == 6
    assert stats.wins == 5
    assert stats.losses == 1


def test_monthly_sharpe_ratio_zero_std():
    agent = ReportAgent(capital=1000.0)
    # All same pnl -> std=0 -> sharpe=0
    for _ in range(3):
        agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_monthly_stats(2026, 5)
    assert stats.sharpe_ratio == 0.0


# ── Criterios go-live ──────────────────────────────────────────────────────

def test_go_live_criteria_mandatory_count():
    c = GoLiveCriteria(
        win_rate_60_3weeks=True, profit_factor_15=True,
        max_drawdown_5=True, min_50_trades=True,
        agents_operational=True, no_risk_violations=True
    )
    assert c.mandatory_total == 6
    assert c.mandatory_passed == 6
    assert c.all_mandatory_passed is True


def test_go_live_criteria_not_all_passed():
    c = GoLiveCriteria(
        win_rate_60_3weeks=False, profit_factor_15=True,
        max_drawdown_5=True, min_50_trades=False,
        agents_operational=True, no_risk_violations=True
    )
    assert c.mandatory_passed == 4
    assert c.all_mandatory_passed is False


def test_go_live_verdict_ready():
    c = GoLiveCriteria(
        win_rate_60_3weeks=True, profit_factor_15=True,
        max_drawdown_5=True, min_50_trades=True,
        agents_operational=True, no_risk_violations=True
    )
    text = c.verdict_text()
    assert "LISTO" in text or "REAL" in text


def test_go_live_verdict_not_ready():
    c = GoLiveCriteria(
        win_rate_60_3weeks=False, profit_factor_15=False,
        max_drawdown_5=True, min_50_trades=False,
        agents_operational=True, no_risk_violations=True
    )
    text = c.verdict_text()
    assert "NO" in text or "Faltan" in text


def test_evaluate_go_live_not_enough_trades():
    agent = ReportAgent(capital=1000.0)
    for _ in range(10):  # solo 10, necesita 50
        agent.add_trade(make_trade(pnl=5.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    criteria = agent.evaluate_go_live_criteria()
    assert criteria.min_50_trades is False


def test_evaluate_go_live_enough_trades():
    agent = ReportAgent(capital=1000.0)
    for _ in range(60):
        agent.add_trade(make_trade(pnl=5.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    criteria = agent.evaluate_go_live_criteria()
    assert criteria.min_50_trades is True


def test_evaluate_no_risk_violations():
    agent = ReportAgent(capital=1000.0)
    # Perdida > 5% del capital (1000 * 0.05 = 50)
    agent.add_trade(make_trade(pnl=-60.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    criteria = agent.evaluate_go_live_criteria()
    assert criteria.no_risk_violations is False


# ── Generacion de texto ────────────────────────────────────────────────────

def test_generate_weekly_report_text_contains_sections():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    text = agent.generate_weekly_report_text(stats)
    assert "REPORTE SEMANAL" in text
    assert "Win Rate" in text or "win_rate" in text.lower()


def test_generate_monthly_report_text_contains_sections():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_monthly_stats(2026, 5)
    text = agent.generate_monthly_report_text(stats)
    assert "REPORTE MENSUAL" in text


def test_generate_telegram_summary_short():
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    summary = agent.generate_telegram_summary(stats)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_generate_criteria_message():
    agent = ReportAgent(capital=1000.0)
    msg = agent.generate_criteria_message()
    assert "criterios" in msg.lower() or "CRITERIOS" in msg


def test_generate_projection_message():
    agent = ReportAgent(capital=1000.0)
    msg = agent.generate_projection_message()
    assert isinstance(msg, str)
    assert len(msg) > 0


# ── Guardado de archivos ───────────────────────────────────────────────────

def test_save_weekly_report_creates_file(tmp_path):
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    path = agent.save_weekly_report(stats, output_dir=str(tmp_path))
    import os
    assert os.path.exists(path)
    assert path.endswith(".html")


def test_save_monthly_report_creates_file(tmp_path):
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_monthly_stats(2026, 5)
    path = agent.save_monthly_report(stats, output_dir=str(tmp_path))
    import os
    assert os.path.exists(path)
    assert path.endswith(".html")


def test_save_weekly_report_content_is_html(tmp_path):
    agent = ReportAgent(capital=1000.0)
    agent.add_trade(make_trade(pnl=10.0, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc)))
    stats = agent.calculate_weekly_stats(date(2026, 5, 11))
    path = agent.save_weekly_report(stats, output_dir=str(tmp_path))
    content = open(path).read()
    assert "<html>" in content.lower() or "<!DOCTYPE" in content


# ── Scheduler ─────────────────────────────────────────────────────────────

def test_setup_scheduler_returns_bool():
    agent = ReportAgent(capital=1000.0)
    result = agent.setup_scheduler()
    assert isinstance(result, bool)
