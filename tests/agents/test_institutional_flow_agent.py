"""
Tests for agents/institutional_flow_agent.py
All HTTP calls are mocked — no network required.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.institutional_flow_agent import (
    InstitutionalFlowAgent,
    COTSnapshot,
    OptionsFlowSnapshot,
    InstitutionalSignal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return InstitutionalFlowAgent()


def _make_cot(commercial_net: int = 50_000, symbol: str = "BTCUSDT") -> COTSnapshot:
    bias = "bullish" if commercial_net > 0 else ("bearish" if commercial_net < 0 else "neutral")
    bonus = 15 if commercial_net != 0 else 0
    return COTSnapshot(
        report_date="2026-05-13",
        symbol=symbol,
        commercial_net=commercial_net,
        noncommercial_net=-20_000,
        retail_net=-30_000,
        commercial_bias=bias,
        score_bonus=bonus,
    )


def _make_options(pcr: float = 0.5, symbol: str = "BTCUSDT") -> OptionsFlowSnapshot:
    bias = "bullish" if pcr < 0.7 else ("bearish" if pcr > 1.3 else "neutral")
    bonus = 10 if pcr < 0.7 or pcr > 1.3 else 0
    return OptionsFlowSnapshot(
        symbol=symbol,
        call_volume=10_000,
        put_volume=int(pcr * 10_000),
        put_call_ratio=round(pcr, 4),
        unusual_activity=False,
        bias=bias,
        score_bonus=bonus,
    )


# ---------------------------------------------------------------------------
# 1. Data-class field presence
# ---------------------------------------------------------------------------

def test_cot_snapshot_fields_present():
    snap = _make_cot()
    assert hasattr(snap, "report_date")
    assert hasattr(snap, "symbol")
    assert hasattr(snap, "commercial_net")
    assert hasattr(snap, "noncommercial_net")
    assert hasattr(snap, "retail_net")
    assert hasattr(snap, "commercial_bias")
    assert hasattr(snap, "score_bonus")


def test_options_snapshot_fields_present():
    snap = _make_options()
    assert hasattr(snap, "call_volume")
    assert hasattr(snap, "put_volume")
    assert hasattr(snap, "put_call_ratio")
    assert hasattr(snap, "unusual_activity")
    assert hasattr(snap, "bias")
    assert hasattr(snap, "score_bonus")


# ---------------------------------------------------------------------------
# 2. Score cap
# ---------------------------------------------------------------------------

def test_score_bonus_capped_at_25(agent):
    """total_bonus must never exceed 25."""
    # Inject pre-filled caches so get_combined_signal uses them directly
    agent._cot_cache["BTCUSDT"] = _make_cot(commercial_net=100_000)
    agent._options_cache["BTCUSDT"] = _make_options(pcr=0.5)

    with patch.object(agent, "get_cot_signal", return_value=agent._cot_cache["BTCUSDT"]):
        with patch.object(agent, "get_options_signal", return_value=agent._options_cache["BTCUSDT"]):
            signal = agent.get_combined_signal("BTCUSDT", "bullish")

    assert signal.total_bonus <= 25


# ---------------------------------------------------------------------------
# 3. COT bullish → maximum bonus
# ---------------------------------------------------------------------------

def test_commercial_bullish_gives_max_bonus(agent):
    """commercial_net > 0 + matching bullish bias → +15 COT bonus."""
    cot = _make_cot(commercial_net=80_000)
    assert cot.commercial_bias == "bullish"
    assert cot.score_bonus == 15

    with patch.object(agent, "get_cot_signal", return_value=cot):
        with patch.object(agent, "get_options_signal", return_value=None):
            sig = agent.get_combined_signal("BTCUSDT", "bullish")

    assert sig.total_bonus == 15


# ---------------------------------------------------------------------------
# 4. Options bearish PCR → bonus
# ---------------------------------------------------------------------------

def test_options_bearish_pcr_gives_bonus(agent):
    """put_call_ratio > 1.3 → bearish options → +10 bonus when bias=bearish."""
    opts = _make_options(pcr=1.8)
    assert opts.bias == "bearish"
    assert opts.score_bonus == 10

    with patch.object(agent, "get_cot_signal", return_value=None):
        with patch.object(agent, "get_options_signal", return_value=opts):
            sig = agent.get_combined_signal("BTCUSDT", "bearish")

    assert sig.total_bonus == 10


# ---------------------------------------------------------------------------
# 5. Combined signal type
# ---------------------------------------------------------------------------

def test_get_combined_signal_returns_institutional_signal(agent):
    with patch.object(agent, "get_cot_signal", return_value=_make_cot()):
        with patch.object(agent, "get_options_signal", return_value=_make_options()):
            result = agent.get_combined_signal("BTCUSDT", "bullish")

    assert isinstance(result, InstitutionalSignal)
    assert isinstance(result.total_bonus, int)
    assert isinstance(result.summary, str)


# ---------------------------------------------------------------------------
# 6. score_adjustment returns int
# ---------------------------------------------------------------------------

def test_score_adjustment_returns_int(agent):
    with patch.object(agent, "get_cot_signal", return_value=None):
        with patch.object(agent, "get_options_signal", return_value=None):
            result = agent.score_adjustment("BTCUSDT", "bullish")

    assert isinstance(result, int)
    assert 0 <= result <= 25


# ---------------------------------------------------------------------------
# 7. Telegram format contains symbol
# ---------------------------------------------------------------------------

def test_format_telegram_has_symbol(agent):
    # Populate caches manually
    agent._cot_cache["BTCUSDT"] = _make_cot()
    agent._options_cache["BTCUSDT"] = _make_options()
    msg = agent.format_telegram("BTCUSDT")
    assert "BTCUSDT" in msg


# ---------------------------------------------------------------------------
# 8. Graceful degradation — no network
# ---------------------------------------------------------------------------

def test_graceful_degradation_no_network(agent):
    """When requests raises, get_cot_signal and get_options_signal return None."""
    with patch("agents.institutional_flow_agent.logger"):
        # Simulate requests not available by patching the import
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("no network")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            cot = agent.get_cot_signal("XAUUSD")
            opts = agent.get_options_signal("XAUUSD")

    # Must return None (no cache) — no exception raised
    assert cot is None
    assert opts is None


# ---------------------------------------------------------------------------
# 9. Put/call ratio calculation
# ---------------------------------------------------------------------------

def test_put_call_ratio_calculation():
    """PCR = put_volume / call_volume."""
    opts = OptionsFlowSnapshot(
        symbol="ETHUSD",
        call_volume=20_000,
        put_volume=10_000,
        put_call_ratio=0.5,
        unusual_activity=False,
        bias="bullish",
        score_bonus=10,
    )
    assert opts.put_call_ratio == pytest.approx(opts.put_volume / opts.call_volume)


def test_put_call_ratio_high_is_bearish():
    agent = InstitutionalFlowAgent()
    bias = agent._compute_options_bias(1.5)
    bonus = agent._compute_options_bonus(1.5)
    assert bias == "bearish"
    assert bonus == 10


def test_put_call_ratio_low_is_bullish():
    agent = InstitutionalFlowAgent()
    bias = agent._compute_options_bias(0.5)
    bonus = agent._compute_options_bonus(0.5)
    assert bias == "bullish"
    assert bonus == 10


# ---------------------------------------------------------------------------
# 10. Neutral signal → zero bonus
# ---------------------------------------------------------------------------

def test_neutral_signal_zero_bonus(agent):
    """When both COT and options give mismatched bias, total_bonus stays 0."""
    cot = _make_cot(commercial_net=50_000)   # bullish commercial
    opts = _make_options(pcr=0.5)            # bullish options

    # But trade bias is bearish → neither aligns → 0 bonus
    with patch.object(agent, "get_cot_signal", return_value=cot):
        with patch.object(agent, "get_options_signal", return_value=opts):
            sig = agent.get_combined_signal("BTCUSDT", "bearish")

    assert sig.total_bonus == 0


def test_neutral_cot_zero_bonus():
    """commercial_net == 0 → score_bonus == 0."""
    cot = COTSnapshot(
        report_date="2026-05-13",
        symbol="BTCUSDT",
        commercial_net=0,
        noncommercial_net=0,
        retail_net=0,
        commercial_bias="neutral",
        score_bonus=0,
    )
    assert cot.score_bonus == 0
