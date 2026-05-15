# tests/agents/test_quant_anomalies.py
import pytest
from datetime import datetime, timezone
from agents.quant_anomalies import AnomalyDetector, AnomalySignal


def dt(year, month, day, weekday_check=None):
    return datetime(year, month, day, 10, 0, tzinfo=timezone.utc)


# ── Monday effect ─────────────────────────────────────────────────────────

def test_monday_effect_on_monday():
    # May 12 2026 es lunes
    signal = AnomalyDetector.check_monday_effect(dt(2026, 5, 11))
    # Verificar que el 12 May 2026 es lunes realmente
    monday = datetime(2026, 5, 11, tzinfo=timezone.utc)
    # Find actual Monday
    from datetime import timedelta
    d = datetime(2026, 5, 11, tzinfo=timezone.utc)
    while d.weekday() != 0:
        d += timedelta(days=1)
    signal = AnomalyDetector.check_monday_effect(d)
    assert signal is not None
    assert signal.direction == "bearish"
    assert signal.pts < 0


def test_monday_effect_not_on_tuesday():
    from datetime import timedelta
    d = datetime(2026, 5, 12, tzinfo=timezone.utc)
    while d.weekday() != 1:  # martes
        d += timedelta(days=1)
    signal = AnomalyDetector.check_monday_effect(d)
    assert signal is None


# ── Turn of month ─────────────────────────────────────────────────────────

def test_turn_of_month_last_days():
    signal = AnomalyDetector.check_turn_of_month(dt(2026, 5, 30))
    assert signal is not None
    assert signal.direction == "bullish"
    assert signal.pts > 0


def test_turn_of_month_first_days():
    signal = AnomalyDetector.check_turn_of_month(dt(2026, 5, 2))
    assert signal is not None


def test_turn_of_month_mid_month():
    signal = AnomalyDetector.check_turn_of_month(dt(2026, 5, 15))
    assert signal is None


# ── End of quarter ────────────────────────────────────────────────────────

def test_end_of_quarter_march():
    signal = AnomalyDetector.check_end_of_quarter(dt(2026, 3, 30))
    assert signal is not None
    assert signal.pts > 0


def test_end_of_quarter_not_quarter_end():
    signal = AnomalyDetector.check_end_of_quarter(dt(2026, 5, 15))
    assert signal is None


# ── Funding rate ──────────────────────────────────────────────────────────

def test_funding_extreme_positive():
    signal = AnomalyDetector.check_funding_rate(0.005)
    assert signal.direction == "bearish"
    assert signal.pts < 0
    assert signal.strength > 0.8


def test_funding_extreme_negative():
    signal = AnomalyDetector.check_funding_rate(-0.002)
    assert signal.direction == "bullish"
    assert signal.pts > 0


def test_funding_normal():
    signal = AnomalyDetector.check_funding_rate(0.0005)
    assert signal.direction == "neutral"
    assert signal.pts == 0


# ── Halving cycle ─────────────────────────────────────────────────────────

def test_halving_phase_1():
    signal = AnomalyDetector.check_halving_cycle_phase(180)
    assert signal.direction == "bullish"
    assert signal.pts >= 5


def test_halving_phase_2_bull():
    signal = AnomalyDetector.check_halving_cycle_phase(500)
    assert signal.pts >= 10


def test_halving_phase_3_dist():
    signal = AnomalyDetector.check_halving_cycle_phase(900)
    assert signal.direction == "bearish"


def test_halving_phase_4_bear():
    signal = AnomalyDetector.check_halving_cycle_phase(1200)
    assert signal.pts <= -5


# ── Gap fill ─────────────────────────────────────────────────────────────

def test_gap_fill_small_gap():
    signal = AnomalyDetector.check_gap_fill_probability(0.005, True)
    assert signal.strength == pytest.approx(0.75)


def test_gap_fill_large_gap_up_bearish():
    signal = AnomalyDetector.check_gap_fill_probability(0.04, True)
    assert signal.direction == "bearish"


def test_gap_fill_large_gap_down_bullish():
    signal = AnomalyDetector.check_gap_fill_probability(0.04, False)
    assert signal.direction == "bullish"


# ── Aggregation ───────────────────────────────────────────────────────────

def test_get_anomaly_score_clamped():
    detector = AnomalyDetector()
    # Funding extremo + halving phase 2 → muchos pts
    d = datetime(2026, 5, 2, tzinfo=timezone.utc)  # turn of month
    score = detector.get_anomaly_score(
        dt=d, funding_rate=-0.003, days_since_halving=500, gap_pct=0.005, is_gap_up=True
    )
    assert -15 <= score <= 15


def test_get_all_signals_returns_list():
    detector = AnomalyDetector()
    signals = detector.get_all_signals(dt(2026, 5, 2), funding_rate=0.005)
    assert isinstance(signals, list)
