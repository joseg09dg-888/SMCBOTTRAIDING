"""
TDD tests for EventDrivenStrategy.
"""
import pytest
from datetime import datetime, timezone, timedelta
from strategies.event_driven import (
    EventDrivenStrategy,
    EventType,
    EventImpact,
    EconomicEvent,
    EventSignal,
)


def dt(year, month, day, hour=12):
    return datetime(year, month, day, hour, 0, tzinfo=timezone.utc)


# ── Halving ────────────────────────────────────────────────────────────────

def test_days_since_halving_after_2024():
    strat = EventDrivenStrategy()
    days = strat.get_days_since_last_halving(dt(2024, 10, 1))
    # 2024 halving April 20 → Oct 1 = ~164 days
    assert 160 <= days <= 170


def test_days_since_halving_before_halving():
    strat = EventDrivenStrategy()
    days = strat.get_days_since_last_halving(dt(2016, 1, 1))
    # Before 2016 halving → days since 2012 halving
    assert days > 1000


def test_halving_phase_accumulation():
    strat = EventDrivenStrategy()
    phase = strat.get_halving_phase(90)  # 90 days post-halving
    assert phase["phase"] == "accumulation"
    assert phase["bias"] == "bullish"
    assert phase["pts"] >= 5


def test_halving_phase_bull_run():
    strat = EventDrivenStrategy()
    phase = strat.get_halving_phase(300)
    assert phase["phase"] == "bull_run"
    assert phase["pts"] >= 10


def test_halving_phase_bear():
    strat = EventDrivenStrategy()
    phase = strat.get_halving_phase(800)
    assert phase["bias"] == "bearish"


# ── FOMC ──────────────────────────────────────────────────────────────────

def test_next_fomc_in_2026():
    strat = EventDrivenStrategy()
    nxt = strat.get_next_fomc(dt(2026, 1, 1))
    assert nxt is not None
    assert nxt.year == 2026


def test_next_fomc_returns_future():
    strat = EventDrivenStrategy()
    now = dt(2026, 3, 1)
    nxt = strat.get_next_fomc(now)
    assert nxt > now


def test_is_fomc_window_true():
    strat = EventDrivenStrategy()
    # Jan 29 2026 19:00 UTC — during FOMC
    fomc_time = datetime(2026, 1, 29, 19, 30, tzinfo=timezone.utc)
    assert strat.is_fomc_window(fomc_time) is True


def test_is_fomc_window_false():
    strat = EventDrivenStrategy()
    far_from_fomc = dt(2026, 2, 15)
    assert strat.is_fomc_window(far_from_fomc) is False


# ── NFP ───────────────────────────────────────────────────────────────────

def test_nfp_dates_count():
    strat = EventDrivenStrategy()
    dates = strat.get_nfp_dates_2026()
    assert len(dates) == 12  # one per month


def test_nfp_dates_all_fridays():
    strat = EventDrivenStrategy()
    dates = strat.get_nfp_dates_2026()
    for d in dates:
        assert d.weekday() == 4  # Friday


def test_nfp_dates_first_of_month():
    strat = EventDrivenStrategy()
    dates = strat.get_nfp_dates_2026()
    for d in dates:
        assert d.day <= 7  # first Friday


def test_is_nfp_window_true():
    strat = EventDrivenStrategy()
    # Jan 2026 NFP is first Friday = Jan 2 at 12:30 UTC
    dates = strat.get_nfp_dates_2026()
    nfp = dates[0].replace(hour=12, minute=30)
    assert strat.is_nfp_window(nfp) is True


def test_is_nfp_window_false():
    strat = EventDrivenStrategy()
    far = dt(2026, 3, 15)
    assert strat.is_nfp_window(far) is False


# ── Risk adjustment ────────────────────────────────────────────────────────

def test_risk_normal():
    strat = EventDrivenStrategy()
    assert strat.get_risk_adjustment(dt(2026, 2, 15)) == pytest.approx(1.0)


def test_risk_during_fomc():
    strat = EventDrivenStrategy()
    fomc_time = datetime(2026, 1, 29, 19, 0, tzinfo=timezone.utc)
    adj = strat.get_risk_adjustment(fomc_time)
    assert adj <= 0.5


# ── Upcoming events ────────────────────────────────────────────────────────

def test_upcoming_events_returns_list():
    strat = EventDrivenStrategy()
    events = strat.get_upcoming_events(dt(2026, 1, 15), days_ahead=30)
    assert isinstance(events, list)


def test_upcoming_events_in_range():
    strat = EventDrivenStrategy()
    now = dt(2026, 1, 15)
    events = strat.get_upcoming_events(now, days_ahead=7)
    for e in events:
        assert e.scheduled_at >= now
        assert e.scheduled_at <= now + timedelta(days=7)


# ── Halving signal ─────────────────────────────────────────────────────────

def test_halving_signal_btc():
    strat = EventDrivenStrategy()
    signal = strat.get_halving_signal_btc(dt(2024, 10, 1))
    assert isinstance(signal, EventSignal)
    assert signal.symbol == "BTCUSDT"


def test_halving_signal_pts_range():
    strat = EventDrivenStrategy()
    signal = strat.get_halving_signal_btc()
    assert -20 <= signal.pts_adjustment <= 20


# ── format_telegram ────────────────────────────────────────────────────────

def test_format_telegram_empty():
    strat = EventDrivenStrategy()
    msg = strat.format_telegram([])
    assert isinstance(msg, str)


def test_format_telegram_with_events():
    strat = EventDrivenStrategy()
    events = strat.get_upcoming_events(dt(2026, 1, 20), days_ahead=30)
    msg = strat.format_telegram(events)
    assert isinstance(msg, str) and len(msg) > 0
