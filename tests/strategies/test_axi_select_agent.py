# tests/strategies/test_axi_select_agent.py
import pytest
from datetime import date, datetime, timezone, timedelta
from strategies.axi_select_agent import (
    AxiSelectAgent, AxiState, AxiStage, AxiEdgeScore, AxiStageConfig
)

agent = AxiSelectAgent()

# ── new_state ─────────────────────────────────────────────────────────────
def test_new_state_defaults():
    s = AxiSelectAgent.new_state(500.0)
    assert s.current_balance == 500.0
    assert s.trades_closed == 0
    assert s.stage == AxiStage.PRE_SEED
    assert s.wins == 0

def test_new_state_initial_balance():
    s = AxiSelectAgent.new_state(1000.0)
    assert s.initial_balance == 1000.0

# ── Edge Score ─────────────────────────────────────────────────────────────
def test_habilidad_perfect():
    s = AxiSelectAgent.new_state(500.0)
    s.wins = 10; s.losses = 3; s.trades_closed = 13
    s.current_balance = 600.0
    h = agent.calculate_habilidad(s)
    assert h == 40  # all 4 conditions met

def test_habilidad_zero():
    s = AxiSelectAgent.new_state(500.0)
    # No trades, no profit
    h = agent.calculate_habilidad(s)
    assert h == 0

def test_consistencia_no_consecutive_losses():
    s = AxiSelectAgent.new_state(500.0)
    s.consecutive_losses = 0
    s.daily_records = [{'date': '2026-05-15', 'pnl': 10, 'trades': 1},
                       {'date': '2026-05-16', 'pnl': 10, 'trades': 1},
                       {'date': '2026-05-17', 'pnl': 10, 'trades': 1},
                       {'date': '2026-05-18', 'pnl': 10, 'trades': 1},
                       {'date': '2026-05-19', 'pnl': 10, 'trades': 1}]
    s.wins = 5; s.losses = 0; s.trades_closed = 5
    s.current_balance = 550.0
    c = agent.calculate_consistencia(s)
    assert c == 30  # all conditions met

def test_consistencia_5_consecutive_losses_penalty():
    s = AxiSelectAgent.new_state(500.0)
    s.consecutive_losses = 5
    c = agent.calculate_consistencia(s)
    assert c < 30  # penalty for 5 consecutive losses

def test_riesgo_perfect():
    s = AxiSelectAgent.new_state(500.0)
    s.max_drawdown_pct = 0.05
    s.current_balance = 510.0
    s.trades_closed = 5
    r = agent.calculate_riesgo(s)
    assert r == 30

def test_riesgo_high_drawdown():
    s = AxiSelectAgent.new_state(500.0)
    s.max_drawdown_pct = 0.12  # > 10%
    r = agent.calculate_riesgo(s)
    assert r < 30

def test_edge_score_total():
    s = AxiSelectAgent.new_state(500.0)
    s.wins = 10; s.losses = 3; s.trades_closed = 13
    s.current_balance = 600.0; s.max_drawdown_pct = 0.05
    es = agent.calculate_edge_score(s)
    assert isinstance(es, AxiEdgeScore)
    assert es.total == es.habilidad + es.consistencia + es.riesgo
    assert 0 <= es.total <= 100

def test_edge_score_is_eligible_seed():
    es = AxiEdgeScore(20, 15, 15, 50)
    assert es.is_eligible_seed is True

def test_edge_score_not_eligible():
    es = AxiEdgeScore(10, 10, 10, 30)
    assert es.is_eligible_seed is False

# ── Stage progression ──────────────────────────────────────────────────────
def test_pre_seed_not_enough_trades():
    s = AxiSelectAgent.new_state(500.0)
    s.trades_closed = 10  # < 20
    s.edge_score = AxiEdgeScore(20, 15, 15, 50)
    stage = agent.get_current_stage(s)
    assert stage == AxiStage.PRE_SEED

def test_seed_stage_reached():
    s = AxiSelectAgent.new_state(500.0)
    s.trades_closed = 25
    s.edge_score = AxiEdgeScore(20, 15, 15, 50)
    stage = agent.get_current_stage(s)
    assert stage == AxiStage.SEED

def test_incubation_stage():
    s = AxiSelectAgent.new_state(500.0)
    s.trades_closed = 25
    s.edge_score = AxiEdgeScore(25, 20, 15, 60)
    assert agent.get_current_stage(s) == AxiStage.INCUBATION

def test_pro_m_stage():
    s = AxiSelectAgent.new_state(500.0)
    s.trades_closed = 50
    s.edge_score = AxiEdgeScore(35, 30, 25, 90)
    assert agent.get_current_stage(s) == AxiStage.PRO_M

def test_funded_capital_seed():
    assert agent.get_funded_capital(AxiStage.SEED) == 5_000

def test_funded_capital_pro_m():
    assert agent.get_funded_capital(AxiStage.PRO_M) == 1_000_000

def test_stages_count():
    assert len(AxiSelectAgent.STAGES) == 6

# ── record_trade ───────────────────────────────────────────────────────────
def test_record_win():
    s = AxiSelectAgent.new_state(500.0)
    s = agent.record_trade(s, 25.0, date(2026, 5, 15))
    assert s.wins == 1
    assert s.current_balance == pytest.approx(525.0)
    assert s.consecutive_losses == 0

def test_record_loss():
    s = AxiSelectAgent.new_state(500.0)
    s = agent.record_trade(s, -20.0, date(2026, 5, 15))
    assert s.losses == 1
    assert s.consecutive_losses == 1
    assert s.max_drawdown_pct > 0

def test_consecutive_losses_reset():
    s = AxiSelectAgent.new_state(500.0)
    for _ in range(3):
        s = agent.record_trade(s, -10.0, date(2026, 5, 15))
    s = agent.record_trade(s, 50.0, date(2026, 5, 16))
    assert s.consecutive_losses == 0

def test_edge_score_updates_on_record():
    s = AxiSelectAgent.new_state(500.0)
    for i in range(25):
        s = agent.record_trade(s, 10.0, date(2026, 5, i % 30 + 1))
    assert s.edge_score.total > 0
    assert s.trades_closed == 25

# ── can_trade ──────────────────────────────────────────────────────────────
def test_can_trade_fresh_state():
    s = AxiSelectAgent.new_state(500.0)
    wednesday = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    ok, msg = agent.can_trade(s, wednesday)
    assert ok is True

def test_cant_trade_max_drawdown():
    s = AxiSelectAgent.new_state(500.0)
    s.max_drawdown_pct = 0.09  # >= 8% safety stop
    ok, msg = agent.can_trade(s)
    assert ok is False

def test_cant_trade_5_losses():
    s = AxiSelectAgent.new_state(500.0)
    s.consecutive_losses = 5
    ok, msg = agent.can_trade(s)
    assert ok is False

def test_cant_trade_friday_late():
    s = AxiSelectAgent.new_state(500.0)
    friday_late = datetime(2026, 5, 22, 17, 0, tzinfo=timezone.utc)
    ok, msg = agent.can_trade(s, friday_late)
    assert ok is False

# ── compare_with_ftmo ──────────────────────────────────────────────────────
def test_compare_axi_free():
    comp = agent.compare_with_ftmo()
    assert comp["axi_select"]["cost"] == "FREE"
    assert comp["axi_select"]["max_funded"] == 1_000_000
    assert comp["ftmo"]["max_funded"] == 200_000

def test_compare_axi_more_capital():
    comp = agent.compare_with_ftmo()
    assert comp["axi_select"]["max_funded"] > comp["ftmo"]["max_funded"]

def test_axi_monthly_income_1m():
    comp = agent.compare_with_ftmo()
    # $1M x 5% monthly x 80% = $40,000
    assert comp["axi_select"]["monthly_5pct_1M"] == pytest.approx(40000.0)

# ── format_telegram ────────────────────────────────────────────────────────
def test_format_telegram_contains_edge_score():
    s = AxiSelectAgent.new_state(500.0)
    msg = agent.format_telegram(s)
    assert "EDGE SCORE" in msg

def test_format_telegram_contains_stage():
    s = AxiSelectAgent.new_state(500.0)
    msg = agent.format_telegram(s)
    assert "PRE" in msg.upper() or "SEED" in msg.upper()

def test_format_telegram_html():
    s = AxiSelectAgent.new_state(500.0)
    assert "<b>" in agent.format_telegram(s)

def test_format_telegram_contains_comparison():
    s = AxiSelectAgent.new_state(500.0)
    msg = agent.format_telegram(s)
    assert "FTMO" in msg or "Axi" in msg.lower() or "axi" in msg.lower()
