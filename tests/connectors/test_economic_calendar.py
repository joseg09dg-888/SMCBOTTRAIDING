"""
TDD tests for the real economic calendar connector (connectors/economic_calendar.py).
"""
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import connectors.economic_calendar as ec


def _sample_raw(impact="High", country="USD", date="2026-07-10T08:30:00-04:00"):
    return [{
        "title": "Test Event",
        "country": country,
        "date": date,
        "impact": impact,
        "forecast": "",
        "previous": "",
    }]


def setup_function(_):
    ec._cache["ts"] = 0.0
    ec._cache["events"] = []


# ── currencies_for_symbol ────────────────────────────────────────────────

def test_currencies_for_symbol_usdcad():
    assert ec.currencies_for_symbol("USDCAD") == {"USD", "CAD"}


def test_currencies_for_symbol_euraud():
    assert ec.currencies_for_symbol("EURAUD") == {"EUR", "AUD"}


def test_currencies_for_symbol_non_forex():
    assert ec.currencies_for_symbol("NAS100.fs") == set()


# ── fetch_calendar ───────────────────────────────────────────────────────

def test_fetch_calendar_parses_events():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _sample_raw()
    mock_resp.raise_for_status.return_value = None
    with patch("connectors.economic_calendar.requests.get", return_value=mock_resp):
        events = ec.fetch_calendar()
    assert len(events) == 1
    assert events[0]["country"] == "USD"
    assert events[0]["impact"] == "High"


def test_fetch_calendar_caches_within_ttl():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _sample_raw()
    mock_resp.raise_for_status.return_value = None
    with patch("connectors.economic_calendar.requests.get", return_value=mock_resp) as mget:
        ec.fetch_calendar()
        ec.fetch_calendar()
    assert mget.call_count == 1  # second call served from cache, no new HTTP request


def test_fetch_calendar_network_failure_returns_stale_cache():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _sample_raw()
    mock_resp.raise_for_status.return_value = None
    with patch("connectors.economic_calendar.requests.get", return_value=mock_resp):
        first = ec.fetch_calendar()

    with patch("connectors.economic_calendar.requests.get", side_effect=Exception("network down")):
        second = ec.fetch_calendar(force=True)

    assert second == first  # stale cache preserved, no crash


def test_fetch_calendar_never_raises_on_bad_response():
    with patch("connectors.economic_calendar.requests.get", side_effect=Exception("boom")):
        events = ec.fetch_calendar()
    assert events == []


# ── get_high_impact_window ───────────────────────────────────────────────

def test_high_impact_window_match():
    event_time = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    mock_resp = MagicMock()
    mock_resp.json.return_value = _sample_raw(date="2026-07-10T08:30:00-04:00")  # -04:00 -> 12:30 UTC
    mock_resp.raise_for_status.return_value = None
    with patch("connectors.economic_calendar.requests.get", return_value=mock_resp):
        result = ec.get_high_impact_window({"USD"}, as_of=event_time, window_minutes=30)
    assert result is not None
    assert result["country"] == "USD"


def test_high_impact_window_no_match_wrong_currency():
    event_time = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    mock_resp = MagicMock()
    mock_resp.json.return_value = _sample_raw(country="EUR", date="2026-07-10T08:30:00-04:00")
    mock_resp.raise_for_status.return_value = None
    with patch("connectors.economic_calendar.requests.get", return_value=mock_resp):
        result = ec.get_high_impact_window({"USD"}, as_of=event_time, window_minutes=30)
    assert result is None


def test_high_impact_window_no_match_low_impact():
    event_time = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    mock_resp = MagicMock()
    mock_resp.json.return_value = _sample_raw(impact="Low", date="2026-07-10T08:30:00-04:00")
    mock_resp.raise_for_status.return_value = None
    with patch("connectors.economic_calendar.requests.get", return_value=mock_resp):
        result = ec.get_high_impact_window({"USD"}, as_of=event_time, window_minutes=30)
    assert result is None


def test_high_impact_window_outside_window():
    event_time = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    far_time = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
    mock_resp = MagicMock()
    mock_resp.json.return_value = _sample_raw(date="2026-07-10T08:30:00-04:00")
    mock_resp.raise_for_status.return_value = None
    with patch("connectors.economic_calendar.requests.get", return_value=mock_resp):
        result = ec.get_high_impact_window({"USD"}, as_of=far_time, window_minutes=30)
    assert result is None
