"""
Tests for agents/onchain_agent.py
All HTTP calls are mocked so tests run offline.
"""
import pytest
from datetime import date
from unittest.mock import patch, MagicMock

from agents.onchain_agent import (
    OnChainAgent,
    OnChainSignal,
    FearGreedData,
    MVRVData,
    HalvingCycleData,
    LAST_HALVING_DATE,
    HALVING_CYCLE_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fear_greed(value: int, label: str) -> FearGreedData:
    return FearGreedData(value=value, label=label, timestamp="1700000000")


def _mock_extreme_fear_response():
    """Returns a mock httpx response simulating extreme fear (value=10)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"value": "10", "value_classification": "Extreme Fear", "timestamp": "1700000000"}]
    }
    return mock_resp


def _mock_greed_response():
    """Returns a mock httpx response simulating greed (value=80)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"value": "80", "value_classification": "Greed", "timestamp": "1700000000"}]
    }
    return mock_resp


# ---------------------------------------------------------------------------
# Halving cycle tests (no network needed)
# ---------------------------------------------------------------------------

def test_halving_cycle_always_returns_data():
    """get_halving_cycle must work with no network — pure date math."""
    agent = OnChainAgent()
    result = agent.get_halving_cycle()
    assert isinstance(result, HalvingCycleData)
    assert result.last_halving == "2024-04-20"
    assert result.next_halving_est is not None


def test_halving_days_since_positive():
    """days_since must be >= 0 for any date on or after the last halving."""
    agent = OnChainAgent()
    result = agent.get_halving_cycle(as_of=date(2025, 1, 1))
    assert result.days_since >= 0


def test_halving_phase_expansion_in_2025():
    """
    2025-05-13 is ~388 days after the 2024-04-20 halving.
    cycle_pct ≈ 388/1458 ≈ 26.6% → expansion phase (25-60%).
    """
    agent = OnChainAgent()
    result = agent.get_halving_cycle(as_of=date(2025, 5, 13))
    assert result.phase == "expansion", f"Expected expansion, got {result.phase} (pct={result.cycle_pct})"


def test_halving_cycle_pct_between_0_and_1():
    """cycle_pct should always be in [0.0, 1.0]."""
    agent = OnChainAgent()
    result = agent.get_halving_cycle(as_of=date(2025, 6, 1))
    assert 0.0 <= result.cycle_pct <= 1.0


def test_halving_accumulation_phase_early():
    """Right after halving (0-25%) should be accumulation."""
    agent = OnChainAgent()
    # 10 days after halving → 10/1458 ≈ 0.7% → accumulation
    result = agent.get_halving_cycle(as_of=date(2024, 4, 30))
    assert result.phase == "accumulation"


# ---------------------------------------------------------------------------
# Fear & Greed + score adjustment
# ---------------------------------------------------------------------------

def test_fear_greed_extreme_fear_bullish_bonus():
    """
    Extreme fear (value <= 25) + bullish bias on a crypto symbol → +15 from fear/greed.
    With MVRV undervalued (price < 30k) → +10.
    Halving in expansion phase (2025) → +5.
    Total = 30 (capped at 30).
    """
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.return_value = _mock_extreme_fear_response()
        bonus = agent.score_adjustment("BTCUSDT", "bullish", price=25_000)
    # Should collect: +15 (extreme fear) + +10 (MVRV undervalued) + +5 (expansion) = 30
    assert bonus == 30


def test_fear_greed_greed_bearish_bonus():
    """When market is Greed and bias is bearish for crypto, no fear/greed bonus fires."""
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.return_value = _mock_greed_response()
        bonus = agent.score_adjustment("BTCUSDT", "bearish", price=50_000)
    # bias != "bullish" → returns 0 immediately
    assert bonus == 0


def test_score_adjustment_non_crypto_zero():
    """score_adjustment must return 0 for non-crypto symbols like EURUSD."""
    agent = OnChainAgent()
    bonus = agent.score_adjustment("EURUSD", "bullish", price=1.10)
    assert bonus == 0


def test_score_adjustment_non_crypto_xauusd_zero():
    """XAUUSD is not in CRYPTO_SYMBOLS → score must be 0."""
    agent = OnChainAgent()
    bonus = agent.score_adjustment("XAUUSD", "bullish", price=2000)
    assert bonus == 0


def test_score_adjustment_btc_bullish_expansion():
    """
    With no network (fear/greed returns None), BTC bullish in expansion:
    +5 from halving phase (expansion). MVRV undervalued adds +10.
    Total = 15.
    """
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("network error")
        # expansion phase + undervalued price
        bonus = agent.score_adjustment("BTCUSDT", "bullish", price=20_000)
    # +10 (MVRV undervalued) + +5 (expansion) = 15 (no fear/greed bonus)
    assert bonus == 15


# ---------------------------------------------------------------------------
# MVRV tests
# ---------------------------------------------------------------------------

def test_mvrv_undervalued_below_30k():
    agent = OnChainAgent()
    result = agent.estimate_mvrv(20_000)
    assert result.zone == "undervalued"
    assert result.signal == "buy_zone"
    assert result.ratio < 1.5


def test_mvrv_neutral_mid_range():
    agent = OnChainAgent()
    result = agent.estimate_mvrv(80_000)
    assert result.zone == "neutral"
    assert result.signal == "neutral"


def test_mvrv_overvalued_above_150k():
    agent = OnChainAgent()
    result = agent.estimate_mvrv(200_000)
    assert result.zone == "overvalued"
    assert result.signal == "sell_zone"
    assert result.ratio > 3.5


# ---------------------------------------------------------------------------
# get_signal tests
# ---------------------------------------------------------------------------

def test_get_signal_returns_onchain_signal():
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("offline")
        signal = agent.get_signal("BTCUSDT", price=50_000)
    assert isinstance(signal, OnChainSignal)
    assert signal.symbol == "BTCUSDT"
    assert signal.overall_bias in ("bullish", "bearish", "neutral")
    assert signal.halving_cycle is not None


def test_get_signal_no_price_mvrv_none():
    """When price=0 the MVRV should not be computed (stays None)."""
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("offline")
        signal = agent.get_signal("BTCUSDT", price=0)
    assert signal.mvrv is None


# ---------------------------------------------------------------------------
# format_telegram test
# ---------------------------------------------------------------------------

def test_format_telegram_has_fear_greed_label():
    """format_telegram output must include 'Fear' or 'N/A'."""
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.return_value = _mock_extreme_fear_response()
        output = agent.format_telegram("BTCUSDT", price=25_000)
    assert "Fear" in output or "Greed" in output or "N/A" in output
    assert "BTCUSDT" in output


def test_format_telegram_offline_graceful():
    """format_telegram should not raise even when the network is down."""
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("no network")
        output = agent.format_telegram("ETHUSDT", price=3_000)
    assert "ETHUSDT" in output
    assert "N/A" in output


# ---------------------------------------------------------------------------
# score_bonus cap test
# ---------------------------------------------------------------------------

def test_score_bonus_capped_at_30():
    """score_bonus in OnChainSignal must never exceed 30."""
    agent = OnChainAgent()
    with patch("agents.onchain_agent.httpx") as mock_httpx:
        mock_httpx.get.return_value = _mock_extreme_fear_response()
        signal = agent.get_signal("BTCUSDT", price=20_000)
    assert signal.score_bonus <= 30
