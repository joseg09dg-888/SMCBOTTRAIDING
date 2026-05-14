"""
Tests for agents/alternative_data_agent.py
All HTTP calls are mocked — no network required.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.alternative_data_agent import (
    AlternativeDataAgent,
    AlternativeDataSignal,
    TrendsSignal,
    SentimentScore,
    _fg_label,
    _contrarian_bias,
    _sentiment_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return AlternativeDataAgent()


def _mock_fg_response(value: int):
    """Return a mock requests.Response for the F&G API."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "data": [
            {
                "value": str(value),
                "value_classification": _fg_label(value).replace("_", " "),
                "timestamp": "1715000000",
            }
        ]
    }
    return mock_resp


# ---------------------------------------------------------------------------
# 1. Extreme fear → contrarian bullish
# ---------------------------------------------------------------------------

def test_fear_greed_extreme_fear_contrarian_bullish(agent):
    """F&G = 10 (extreme fear) + bullish bias → +15 bonus."""
    with patch.object(agent, "get_fear_greed", return_value={"value": 10, "value_classification": "extreme fear"}):
        with patch.object(agent, "get_google_trends", return_value=[]):
            sig = agent.get_combined_signal("BTCUSDT", "bullish")

    assert sig.fear_greed_index == 10
    assert sig.fear_greed_label == "extreme_fear"
    assert sig.contrarian_bias == "bullish"
    assert sig.total_bonus == 15


# ---------------------------------------------------------------------------
# 2. Extreme greed → contrarian bearish
# ---------------------------------------------------------------------------

def test_fear_greed_extreme_greed_contrarian_bearish(agent):
    """F&G = 90 (extreme greed) + bearish bias → +15 bonus."""
    with patch.object(agent, "get_fear_greed", return_value={"value": 90, "value_classification": "extreme greed"}):
        with patch.object(agent, "get_google_trends", return_value=[]):
            sig = agent.get_combined_signal("BTCUSDT", "bearish")

    assert sig.fear_greed_index == 90
    assert sig.fear_greed_label == "extreme_greed"
    assert sig.contrarian_bias == "bearish"
    assert sig.total_bonus == 15


# ---------------------------------------------------------------------------
# 3. Neutral F&G → zero bonus
# ---------------------------------------------------------------------------

def test_fear_greed_neutral_zero_bonus(agent):
    """F&G = 50 (neutral) → no contrarian edge → bonus 0."""
    with patch.object(agent, "get_fear_greed", return_value={"value": 50, "value_classification": "neutral"}):
        with patch.object(agent, "get_google_trends", return_value=[]):
            sig = agent.get_combined_signal("BTCUSDT", "bullish")

    assert sig.fear_greed_label == "neutral"
    assert sig.total_bonus == 0


# ---------------------------------------------------------------------------
# 4. Score range 0-15
# ---------------------------------------------------------------------------

def test_score_adjustment_range_0_to_15(agent):
    """score_adjustment must always return a value in [0, 15]."""
    for fg_val, bias in [(10, "bullish"), (90, "bearish"), (50, "bullish"), (30, "bearish")]:
        with patch.object(agent, "get_fear_greed", return_value={"value": fg_val}):
            with patch.object(agent, "get_google_trends", return_value=[]):
                result = agent.score_adjustment("BTCUSDT", bias)
        assert 0 <= result <= 15, f"Out of range for F&G={fg_val} bias={bias}: {result}"


# ---------------------------------------------------------------------------
# 5. Combined signal type
# ---------------------------------------------------------------------------

def test_get_combined_signal_returns_signal(agent):
    with patch.object(agent, "get_fear_greed", return_value={"value": 20, "value_classification": "extreme fear"}):
        with patch.object(agent, "get_google_trends", return_value=[]):
            result = agent.get_combined_signal("BTCUSDT", "bullish")

    assert isinstance(result, AlternativeDataSignal)
    assert isinstance(result.fear_greed_index, int)
    assert isinstance(result.total_bonus, int)
    assert isinstance(result.summary, str)
    assert result.google_trends_signals == []


# ---------------------------------------------------------------------------
# 6. Telegram format contains fear_greed
# ---------------------------------------------------------------------------

def test_format_telegram_contains_fear_greed(agent):
    agent._fg_cache = {"value": 22, "value_classification": "extreme fear"}
    msg = agent.format_telegram("BTCUSDT")
    assert "Fear" in msg or "fear" in msg or "22" in msg


def test_format_telegram_contains_symbol(agent):
    agent._fg_cache = {"value": 55, "value_classification": "neutral"}
    msg = agent.format_telegram("ETHUSD")
    assert "ETHUSD" in msg


# ---------------------------------------------------------------------------
# 7. Graceful degradation — no network
# ---------------------------------------------------------------------------

def test_graceful_degradation_no_network(agent):
    """When requests fails, get_fear_greed returns None without raising."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("no network")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = agent.get_fear_greed()

    assert result is None  # no cache yet → None, no exception


# ---------------------------------------------------------------------------
# 8. No pytrends → empty list
# ---------------------------------------------------------------------------

def test_no_pytrends_returns_empty_list(agent):
    """If pytrends is not installed, get_google_trends returns []."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pytrends.request":
            raise ImportError("pytrends not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = agent.get_google_trends(["bitcoin"])

    assert result == []


# ---------------------------------------------------------------------------
# 9. Sentiment score is within [-1.0, +1.0]
# ---------------------------------------------------------------------------

def test_sentiment_score_between_minus1_and_plus1(agent):
    """overall_sentiment must be within [-1.0, +1.0] for any F&G value."""
    for fg_val in [0, 10, 25, 50, 75, 90, 100]:
        with patch.object(agent, "get_fear_greed", return_value={"value": fg_val}):
            with patch.object(agent, "get_google_trends", return_value=[]):
                sig = agent.get_combined_signal("BTCUSDT", "bullish")
        assert -1.0 <= sig.overall_sentiment <= 1.0, (
            f"overall_sentiment={sig.overall_sentiment} out of range for F&G={fg_val}"
        )


# ---------------------------------------------------------------------------
# 10. Contrarian bias is opposite of crowd
# ---------------------------------------------------------------------------

def test_contrarian_bias_opposite_of_crowd():
    """When crowd is greedy, contrarian bias is bearish, and vice-versa."""
    assert _contrarian_bias("extreme_fear") == "bullish"
    assert _contrarian_bias("fear") == "bullish"
    assert _contrarian_bias("neutral") == "neutral"
    assert _contrarian_bias("greed") == "bearish"
    assert _contrarian_bias("extreme_greed") == "bearish"


# ---------------------------------------------------------------------------
# Extra: helper function unit tests
# ---------------------------------------------------------------------------

def test_fg_label_boundaries():
    assert _fg_label(0) == "extreme_fear"
    assert _fg_label(25) == "extreme_fear"
    assert _fg_label(26) == "fear"
    assert _fg_label(45) == "fear"
    assert _fg_label(46) == "neutral"
    assert _fg_label(55) == "neutral"
    assert _fg_label(56) == "greed"
    assert _fg_label(75) == "greed"
    assert _fg_label(76) == "extreme_greed"
    assert _fg_label(100) == "extreme_greed"


def test_sentiment_score_helper():
    assert _sentiment_score(50) == 0.0
    assert _sentiment_score(100) == 1.0
    assert _sentiment_score(0) == -1.0


def test_trends_signal_dataclass():
    ts = TrendsSignal(
        keyword="buy bitcoin",
        current_value=95,
        peak_value=100,
        pct_of_peak=0.95,
        interpretation="euphoria",
        score_bonus=5,
    )
    assert ts.keyword == "buy bitcoin"
    assert ts.interpretation == "euphoria"
    assert ts.score_bonus == 5


def test_fear_bullish_bias_partial_bonus(agent):
    """F&G in 'fear' zone (not extreme) + bullish bias → 7 pts."""
    with patch.object(agent, "get_fear_greed", return_value={"value": 35}):
        with patch.object(agent, "get_google_trends", return_value=[]):
            sig = agent.get_combined_signal("BTCUSDT", "bullish")

    assert sig.fear_greed_label == "fear"
    assert sig.total_bonus == 7
