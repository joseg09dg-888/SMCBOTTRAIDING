"""
Tests for agents/geopolitical_agent.py
All HTTP calls are mocked so tests run offline.
"""
import pytest
from unittest.mock import patch, MagicMock

from agents.geopolitical_agent import (
    GeopoliticalAgent,
    GeopoliticalSignal,
    GeopoliticalEvent,
    CATEGORY_IMPACT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conflict_event(severity: int = 8) -> GeopoliticalEvent:
    return GeopoliticalEvent(
        title="Major conflict erupted in region",
        category="conflict",
        severity=severity,
        affected_markets=["XAUUSD", "USOIL", "USDJPY", "BTCUSDT"],
        market_bias={"XAUUSD": "bullish", "USOIL": "bullish", "USDJPY": "bullish", "BTCUSDT": "bullish"},
        source="GDELT",
        timestamp="2025-05-13T00:00:00",
    )


def _make_sanctions_event(severity: int = 6) -> GeopoliticalEvent:
    return GeopoliticalEvent(
        title="New sanctions imposed on major economy",
        category="sanctions",
        severity=severity,
        affected_markets=["BTCUSDT", "EURUSD=X"],
        market_bias={"BTCUSDT": "bullish", "EURUSD=X": "bearish"},
        source="GDELT",
        timestamp="2025-05-13T00:00:00",
    )


def _mock_gdelt_response(events_data=None):
    """Mock GDELT API returning a list of articles."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "articles": events_data or [
            {"title": "Military conflict escalates in region", "seendate": "2025-05-13T00:00:00"},
        ]
    }
    return mock_resp


# ---------------------------------------------------------------------------
# Baseline / no-events tests
# ---------------------------------------------------------------------------

def test_baseline_risk_score_no_events():
    """With an empty events list, risk_score must be 3 (baseline)."""
    agent = GeopoliticalAgent()
    score = agent.calculate_risk_score([])
    assert score == 3


def test_no_events_graceful_degradation():
    """get_signal with no network must return a valid GeopoliticalSignal with baseline risk."""
    agent = GeopoliticalAgent()
    with patch("agents.geopolitical_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("no network")
        signal = agent.get_signal("XAUUSD")
    assert isinstance(signal, GeopoliticalSignal)
    assert signal.risk_score == 3
    assert signal.risk_label == "moderate"
    assert signal.trade_blocked is False


def test_get_signal_returns_geopolitical_signal():
    """get_signal must always return a GeopoliticalSignal instance."""
    agent = GeopoliticalAgent()
    with patch("agents.geopolitical_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("offline")
        signal = agent.get_signal()
    assert isinstance(signal, GeopoliticalSignal)
    assert isinstance(signal.recent_events, list)
    assert isinstance(signal.market_impact, dict)


# ---------------------------------------------------------------------------
# Risk score calculation
# ---------------------------------------------------------------------------

def test_high_severity_increases_risk():
    """A single severity-9 event should push risk_score to 9."""
    agent = GeopoliticalAgent()
    events = [_make_conflict_event(severity=9)]
    score = agent.calculate_risk_score(events)
    assert score == 9


def test_multiple_events_average_risk():
    """Average of severities 8 and 4 → risk_score = 6."""
    agent = GeopoliticalAgent()
    events = [
        _make_conflict_event(severity=8),
        _make_sanctions_event(severity=4),
    ]
    score = agent.calculate_risk_score(events)
    assert score == 6


def test_risk_score_capped_at_10():
    """risk_score must never exceed 10."""
    agent = GeopoliticalAgent()
    events = [_make_conflict_event(severity=10)] * 5
    score = agent.calculate_risk_score(events)
    assert score <= 10


def test_risk_score_minimum_1():
    """risk_score must never go below 1."""
    agent = GeopoliticalAgent()
    events = [_make_conflict_event(severity=1)]
    score = agent.calculate_risk_score(events)
    assert score >= 1


# ---------------------------------------------------------------------------
# Trade blocked
# ---------------------------------------------------------------------------

def test_trade_blocked_when_risk_above_7():
    """trade_blocked must be True when risk_score > 7."""
    agent = GeopoliticalAgent()
    with patch.object(agent, "fetch_events", return_value=[_make_conflict_event(severity=9)]):
        signal = agent.get_signal("XAUUSD")
    assert signal.risk_score > 7
    assert signal.trade_blocked is True


def test_trade_not_blocked_when_risk_low():
    """trade_blocked must be False when risk_score <= 7."""
    agent = GeopoliticalAgent()
    with patch.object(agent, "fetch_events", return_value=[_make_sanctions_event(severity=4)]):
        signal = agent.get_signal("XAUUSD")
    assert signal.trade_blocked is False


# ---------------------------------------------------------------------------
# Market impact / category impact
# ---------------------------------------------------------------------------

def test_conflict_makes_gold_bullish():
    """A conflict event must make XAUUSD bullish."""
    agent = GeopoliticalAgent()
    events = [_make_conflict_event(severity=8)]
    impact = agent.get_market_impact("XAUUSD", events)
    assert impact == "bullish"


def test_sanctions_makes_btc_bullish():
    """A sanctions event must make BTCUSDT bullish."""
    agent = GeopoliticalAgent()
    events = [_make_sanctions_event(severity=7)]
    impact = agent.get_market_impact("BTCUSDT", events)
    assert impact == "bullish"


def test_market_impact_conflict_xauusd():
    """Direct test of CATEGORY_IMPACT mapping for conflict → XAUUSD."""
    assert CATEGORY_IMPACT["conflict"]["XAUUSD"] == "bullish"


def test_market_impact_neutral_when_no_events():
    """No events → market impact for any symbol should be 'neutral'."""
    agent = GeopoliticalAgent()
    impact = agent.get_market_impact("XAUUSD", [])
    assert impact == "neutral"


def test_market_impact_unknown_symbol_neutral():
    """Unknown symbol not in any category map → neutral."""
    agent = GeopoliticalAgent()
    events = [_make_conflict_event(severity=8)]
    impact = agent.get_market_impact("EURUSD=X", events)
    # EURUSD=X is not in conflict map → neutral
    assert impact == "neutral"


# ---------------------------------------------------------------------------
# Score adjustment
# ---------------------------------------------------------------------------

def test_score_adjustment_returns_int():
    """score_adjustment must return an int."""
    agent = GeopoliticalAgent()
    with patch("agents.geopolitical_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("offline")
        result = agent.score_adjustment("XAUUSD", "bullish")
    assert isinstance(result, int)


def test_score_negative_when_high_risk():
    """score_adjustment must return -15 when risk_score > 7."""
    agent = GeopoliticalAgent()
    with patch.object(agent, "fetch_events", return_value=[_make_conflict_event(severity=9)]):
        result = agent.score_adjustment("XAUUSD", "bullish")
    assert result == -15


def test_score_positive_when_geopolitical_favors_direction():
    """
    score_adjustment returns +10 when geopolitical impact matches the trade bias
    and risk_score <= 7.
    """
    agent = GeopoliticalAgent()
    # severity=6 → risk_score=6 (not blocked), conflict → XAUUSD bullish
    with patch.object(agent, "fetch_events", return_value=[_make_conflict_event(severity=6)]):
        result = agent.score_adjustment("XAUUSD", "bullish")
    assert result == 10


def test_score_zero_when_no_events():
    """No events → risk_score=3, no directional impact → score=0."""
    agent = GeopoliticalAgent()
    with patch.object(agent, "fetch_events", return_value=[]):
        result = agent.score_adjustment("XAUUSD", "bullish")
    assert result == 0


# ---------------------------------------------------------------------------
# format_telegram
# ---------------------------------------------------------------------------

def test_format_telegram_has_risk_score():
    """format_telegram output must contain the risk score."""
    agent = GeopoliticalAgent()
    with patch("agents.geopolitical_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("offline")
        output = agent.format_telegram("XAUUSD")
    assert "Risk Score" in output or "risk" in output.lower()
    assert any(str(n) in output for n in range(1, 11))


def test_format_telegram_offline_no_raise():
    """format_telegram must not raise when network is unavailable."""
    agent = GeopoliticalAgent()
    with patch("agents.geopolitical_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("no network")
        output = agent.format_telegram()
    assert isinstance(output, str)
    assert len(output) > 0


def test_format_telegram_includes_symbol():
    """When a symbol is provided it should appear in the Telegram output."""
    agent = GeopoliticalAgent()
    with patch("agents.geopolitical_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("offline")
        output = agent.format_telegram("XAUUSD")
    assert "XAUUSD" in output
