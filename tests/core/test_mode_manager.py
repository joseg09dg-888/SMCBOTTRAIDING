import pytest
from datetime import datetime, timezone
from core.mode_manager import ModeManager, TradingMode, ModeDecision, ModeRule


def test_mode_enum_values():
    assert TradingMode.AUTO.value == "auto"
    assert TradingMode.SEMI.value == "semi"
    assert TradingMode.HYBRID.value == "hybrid"


def test_initial_mode_is_hybrid():
    mm = ModeManager()
    assert mm.current_mode == TradingMode.HYBRID


def test_auto_switches_to_semi_on_premium_score():
    mm = ModeManager(base_mode=TradingMode.AUTO)
    decision = mm.decide(
        score=92,
        glint_critical=False,
        atr_ratio=1.0,
        trades_today=20,
        last_5_wins=3,
        win_rate_today=65.0,
        atr_normal=True,
        hour_utc=14,
    )
    assert decision.mode == TradingMode.SEMI
    assert "premium" in decision.reason.lower() or "90" in decision.reason


def test_auto_switches_to_semi_on_critical_glint():
    mm = ModeManager(base_mode=TradingMode.AUTO)
    decision = mm.decide(
        score=75,
        glint_critical=True,
        atr_ratio=1.0,
        trades_today=20,
        last_5_wins=3,
        win_rate_today=65.0,
        atr_normal=True,
        hour_utc=14,
    )
    assert decision.mode == TradingMode.SEMI
    assert "glint" in decision.reason.lower() or "critica" in decision.reason.lower() or "critical" in decision.reason.lower()


def test_auto_switches_to_semi_on_high_volatility():
    mm = ModeManager(base_mode=TradingMode.AUTO)
    decision = mm.decide(
        score=75,
        glint_critical=False,
        atr_ratio=3.5,   # > 3x threshold
        trades_today=20,
        last_5_wins=3,
        win_rate_today=65.0,
        atr_normal=False,
        hour_utc=14,
    )
    assert decision.mode == TradingMode.SEMI
    assert "volatil" in decision.reason.lower() or "atr" in decision.reason.lower()


def test_auto_stays_auto_on_normal_conditions():
    mm = ModeManager(base_mode=TradingMode.AUTO)
    decision = mm.decide(
        score=78,
        glint_critical=False,
        atr_ratio=1.2,
        trades_today=20,
        last_5_wins=4,
        win_rate_today=70.0,
        atr_normal=True,
        hour_utc=14,
    )
    assert decision.mode == TradingMode.AUTO


def test_semi_switches_to_auto_on_good_streak():
    mm = ModeManager(base_mode=TradingMode.SEMI)
    decision = mm.decide(
        score=78,
        glint_critical=False,
        atr_ratio=1.0,
        trades_today=15,
        last_5_wins=5,     # 5/5 wins
        win_rate_today=68.0,
        atr_normal=True,
        hour_utc=14,
    )
    assert decision.mode == TradingMode.AUTO


def test_night_mode_forces_semi(monkeypatch):
    mm = ModeManager(base_mode=TradingMode.AUTO)
    decision = mm.decide(
        score=78,
        glint_critical=False,
        atr_ratio=1.0,
        trades_today=20,
        last_5_wins=4,
        win_rate_today=65.0,
        atr_normal=True,
        hour_utc=2,    # 2am UTC = night mode
    )
    assert decision.mode == TradingMode.SEMI
    assert "noche" in decision.reason.lower() or "night" in decision.reason.lower()


def test_night_mode_blocks_low_score():
    mm = ModeManager(base_mode=TradingMode.AUTO)
    decision = mm.decide(
        score=72,    # below 85 threshold for night
        glint_critical=False,
        atr_ratio=1.0,
        trades_today=10,
        last_5_wins=3,
        win_rate_today=60.0,
        atr_normal=True,
        hour_utc=3,
    )
    assert decision.mode == TradingMode.SEMI
    assert decision.night_mode is True


def test_early_warmup_forces_semi():
    mm = ModeManager(base_mode=TradingMode.AUTO)
    decision = mm.decide(
        score=78,
        glint_critical=False,
        atr_ratio=1.0,
        trades_today=5,   # < 10 trades = warmup
        last_5_wins=3,
        win_rate_today=60.0,
        atr_normal=True,
        hour_utc=14,
    )
    assert decision.mode == TradingMode.SEMI
    assert "calentamiento" in decision.reason.lower() or "warmup" in decision.reason.lower() or "10" in decision.reason


def test_mode_history_recorded():
    mm = ModeManager(base_mode=TradingMode.AUTO)
    mm.decide(score=92, glint_critical=False, atr_ratio=1.0,
              trades_today=20, last_5_wins=3, win_rate_today=65.0,
              atr_normal=True, hour_utc=14)
    assert len(mm.history) >= 1
    assert "reason" in mm.history[-1]
    assert "mode" in mm.history[-1]


def test_decision_has_required_fields():
    mm = ModeManager()
    decision = mm.decide(score=75, glint_critical=False, atr_ratio=1.0,
                         trades_today=20, last_5_wins=3, win_rate_today=65.0,
                         atr_normal=True, hour_utc=14)
    assert isinstance(decision, ModeDecision)
    assert isinstance(decision.mode, TradingMode)
    assert isinstance(decision.reason, str)
    assert isinstance(decision.night_mode, bool)
    assert isinstance(decision.rules_triggered, list)
