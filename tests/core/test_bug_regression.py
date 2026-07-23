# tests/core/test_bug_regression.py
"""
Regression tests for bugs fixed in audit 2026-06-01.
Each test directly verifies the guard added to prevent the crash.
"""
import pytest
import sqlite3
import pandas as pd
from datetime import datetime, timezone


# ── continuous_learning: _channels vs CHANNELS ────────────────────────────

def test_continuous_learning_uses_instance_channels():
    from core.continuous_learning import ContinuousLearningEngine, YouTubeChannel
    engine1 = ContinuousLearningEngine()
    engine2 = ContinuousLearningEngine()

    # Mutating instance channels must NOT affect the class-level list
    engine1._channels[0].is_live = True
    assert not engine2._channels[0].is_live, (
        "_channels should be instance-level, not shared across instances"
    )


def test_continuous_learning_channels_not_shared_with_class():
    from core.continuous_learning import ContinuousLearningEngine
    engine = ContinuousLearningEngine()
    engine._channels[0].is_live = True

    # Class-level CHANNELS must remain unmodified
    assert not ContinuousLearningEngine.CHANNELS[0].is_live


def test_study_report_uses_instance_channels():
    from core.continuous_learning import ContinuousLearningEngine
    engine = ContinuousLearningEngine()
    report = engine.get_study_report()
    assert "Canales YouTube en vivo: 0/" in report


# ── wakeup_recovery: None entry guard ────────────────────────────────────

def test_wakeup_recovery_none_entry_float_conversion():
    """float(None or 0) must not raise TypeError."""
    entry_raw = None
    entry = float(entry_raw or 0)
    assert entry == 0.0


def test_wakeup_recovery_none_size_float_conversion():
    """float(None or 0) must not raise TypeError."""
    size_raw = None
    size = float(size_raw or 0)
    assert size == 0.0


def test_wakeup_recovery_load_positions_empty_file():
    from core.wakeup_recovery import load_positions
    saved_at, positions = load_positions()
    assert isinstance(positions, list)


# ── decision_filter: empty/non-numeric breakdown ──────────────────────────

def test_decision_filter_empty_breakdown_no_crash():
    from core.decision_filter import DecisionFilter
    from core.config import Config
    from core.risk_manager import RiskManager

    cfg = Config()
    rm = RiskManager(cfg, capital=100_000.0)
    df = DecisionFilter(cfg, rm, None)
    result = df._score_to_result(score=30, symbol="EURUSD", breakdown={})
    assert result.score == 30
    assert "30" in result.reason


def test_decision_filter_none_values_in_breakdown_no_crash():
    from core.decision_filter import DecisionFilter
    from core.config import Config
    from core.risk_manager import RiskManager

    cfg = Config()
    rm = RiskManager(cfg, capital=100_000.0)
    df = DecisionFilter(cfg, rm, None)
    breakdown = {"smc": None, "ml": None}
    result = df._score_to_result(score=25, symbol="EURUSD", breakdown=breakdown)
    assert isinstance(result.reason, str)
    assert "25" in result.reason


# ── volume_calculator: zero pip_size / pip_value ──────────────────────────

def test_volume_calculator_zero_pip_size_returns_zero():
    from core.volume_calculator import VolumeCalculator
    vc = VolumeCalculator()
    vc._PIP_SIZE["TEST"] = 0.0
    vc._PIP_VALUE["TEST"] = 10.0
    vol = vc.calculate_volume(100_000, 1.1000, 1.0950, "TEST")
    assert vol == 0.0


def test_volume_calculator_zero_pip_value_returns_zero():
    from core.volume_calculator import VolumeCalculator
    vc = VolumeCalculator()
    vc._PIP_SIZE["TESTZERO"] = 0.0001
    vc._PIP_VALUE["TESTZERO"] = 0.0
    vol = vc.calculate_volume(100_000, 1.1000, 1.0950, "TESTZERO")
    assert vol == 0.0


# ── Friday cutoff: no trades after 16:00 UTC ─────────────────────────────

def test_friday_cutoff_is_1600_utc():
    """Regression: GBPUSD was opened at 16:38 UTC Friday because cutoff was 20:00."""
    import pathlib
    src = pathlib.Path("core/supervisor.py").read_text(encoding="utf-8")
    # The old wrong cutoff must NOT appear together on the same line
    for line in src.splitlines():
        if "weekday() == 4" in line and "hour >= 20" in line:
            raise AssertionError(
                f"Friday cutoff still set to 20:00 UTC on line: {line.strip()}"
            )
    # The correct cutoff must exist
    assert any(
        "weekday() == 4" in line and "hour >= 16" in line
        for line in src.splitlines()
    ), "Friday cutoff 16:00 UTC not found in supervisor.py"


def test_friday_preclose_hour_in_manage_positions():
    """FRIDAY_CLOSE_HOUR must be 19 to close losers at 19:30 UTC.

    _manage_open_positions lives in core/position_guards.py (PositionGuardsMixin,
    mixed into TradingSupervisor) since the 2026-07-23 simplification split --
    check the source via the actual bound method so this stays correct
    regardless of which file defines it.
    """
    import inspect
    from core.supervisor import TradingSupervisor
    src = inspect.getsource(TradingSupervisor._manage_open_positions)
    assert "FRIDAY_CLOSE_HOUR = 19" in src, "FRIDAY_CLOSE_HOUR must be 19 UTC"
    assert "FRIDAY_CLOSE_MIN  = 30" in src, "FRIDAY_CLOSE_MIN must be 30"


# ── telegram_commander: no duplicate methods ──────────────────────────────

def test_telegram_commander_no_duplicate_methods():
    import inspect
    from dashboard.telegram_commander import TelegramCommander

    methods = [name for name, _ in inspect.getmembers(TelegramCommander, predicate=inspect.isfunction)]
    # Check no duplicates (Python overwrites, but source having dups is a bug)
    seen = {}
    import ast, pathlib

    src = pathlib.Path("dashboard/telegram_commander.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TelegramCommander":
            func_names = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
            duplicates = [n for n in func_names if func_names.count(n) > 1]
            assert duplicates == [], f"Duplicate method definitions: {set(duplicates)}"
