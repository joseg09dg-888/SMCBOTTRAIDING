import pytest
from connectors.glint_connector import GlintConnector, GlintSignal


def test_signal_parsing():
    raw = {
        "id": "abc123",
        "category": "Crypto",
        "impact": "Critical",
        "text": "Bitcoin ETF aprobado por SEC",
        "source_tier": 1,
        "relevance_score": 9.2,
        "matched_market": "BTC-USD",
        "timestamp": "2026-05-12T10:30:00Z",
    }
    signal = GlintSignal.from_dict(raw)
    assert signal.category == "Crypto"
    assert signal.impact == "Critical"
    assert signal.relevance_score == 9.2
    assert signal.is_actionable()


def test_non_actionable_low_impact():
    raw = {
        "id": "xyz",
        "category": "Sports",
        "impact": "Low",
        "text": "Liga de fútbol resultado",
        "source_tier": 3,
        "relevance_score": 2.1,
        "matched_market": "N/A",
        "timestamp": "2026-05-12T10:00:00Z",
    }
    signal = GlintSignal.from_dict(raw)
    assert not signal.is_actionable()


def test_signal_to_trading_context():
    raw = {
        "id": "t1",
        "category": "Economics",
        "impact": "High",
        "text": "Fed sube tasas 50bps sorpresivamente",
        "source_tier": 1,
        "relevance_score": 8.5,
        "matched_market": "GOLD-USD",
        "timestamp": "2026-05-12T14:00:00Z",
    }
    signal = GlintSignal.from_dict(raw)
    ctx = signal.to_trading_context()
    assert "instruments" in ctx
    assert len(ctx["instruments"]) > 0
    assert ctx["urgency"] in ("immediate", "monitor", "ignore")


def test_format_alert_contains_impact():
    raw = {
        "id": "t2",
        "category": "Military",
        "impact": "Critical",
        "text": "Conflicto escala en Oriente Medio",
        "source_tier": 1,
        "relevance_score": 9.8,
        "matched_market": "XAUUSD",
        "timestamp": "2026-05-12T15:00:00Z",
    }
    signal = GlintSignal.from_dict(raw)
    alert = signal.format_alert()
    assert "CRITICAL" in alert.upper()
    assert signal.text in alert


def test_connector_stats():
    gc = GlintConnector(
        ws_url="wss://glint.trade/ws",
        session_token="test-token",
    )
    stats = gc.stats()
    assert stats["connected"] is False
    assert stats["signals_received"] == 0
