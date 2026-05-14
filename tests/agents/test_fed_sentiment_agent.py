"""Tests for FEDSentimentAgent."""
import pytest
from datetime import date
from unittest.mock import patch, MagicMock

from agents.fed_sentiment_agent import (
    FEDSentimentAgent,
    FEDSentimentResult,
    FOMCEvent,
    FOMC_DATES_2025,
    FOMC_DATES_2026,
)


@pytest.fixture
def agent():
    return FEDSentimentAgent()


# ── FOMC blackout tests ────────────────────────────────────────────────────

def test_fomc_blackout_true_within_24h(agent):
    """Blackout is True when an FOMC meeting is 0 days away (same day)."""
    fomc_day = date.fromisoformat("2025-01-29")
    assert agent.is_fomc_blackout(as_of=fomc_day) is True


def test_fomc_blackout_true_one_day_before(agent):
    """Blackout is True when FOMC is 1 day away (within 24 hours)."""
    one_day_before = date.fromisoformat("2025-01-28")
    assert agent.is_fomc_blackout(as_of=one_day_before) is True


def test_fomc_blackout_false_when_far(agent):
    """Blackout is False when no FOMC meeting is within 24 hours."""
    far_day = date.fromisoformat("2025-01-10")  # 19 days before Jan 29
    assert agent.is_fomc_blackout(as_of=far_day) is False


def test_fomc_blackout_false_two_days_before(agent):
    """Blackout is False when FOMC is still 2+ days away."""
    two_days_before = date.fromisoformat("2025-01-27")
    assert agent.is_fomc_blackout(as_of=two_days_before) is False


# ── Next FOMC tests ────────────────────────────────────────────────────────

def test_next_fomc_returns_future_date(agent):
    """get_next_fomc returns a FOMCEvent with a date >= as_of."""
    as_of = date.fromisoformat("2025-01-01")
    event = agent.get_next_fomc(as_of=as_of)
    assert event is not None
    assert isinstance(event, FOMCEvent)
    assert event.date == "2025-01-29"
    assert event.days_until >= 0


def test_next_fomc_skips_past_dates(agent):
    """get_next_fomc skips dates before as_of."""
    as_of = date.fromisoformat("2025-02-01")  # after Jan 29
    event = agent.get_next_fomc(as_of=as_of)
    assert event is not None
    assert event.date == "2025-03-19"


def test_next_fomc_days_until_correct(agent):
    """days_until is correctly calculated."""
    as_of = date.fromisoformat("2025-01-22")  # 7 days before Jan 29
    event = agent.get_next_fomc(as_of=as_of)
    assert event is not None
    assert event.days_until == 7


def test_next_fomc_in_2026(agent):
    """After all 2025 dates, next FOMC should be from 2026 list."""
    as_of = date.fromisoformat("2025-12-31")
    event = agent.get_next_fomc(as_of=as_of)
    assert event is not None
    assert event.date == "2026-01-28"


# ── Sentiment analysis tests ───────────────────────────────────────────────

def test_hawkish_keywords_increase_hawkish_score(agent):
    """Text with hawkish keywords should produce hawkish_score > dovish_score."""
    text = "The FED will hike rates due to persistent inflation and restrictive policy is needed to tighten conditions."
    result = agent.analyze_sentiment(text)
    assert result.hawkish_score > result.dovish_score
    assert result.net_sentiment > 0


def test_dovish_keywords_increase_dovish_score(agent):
    """Text with dovish keywords should produce dovish_score > hawkish_score."""
    text = "The FED will cut rates to support the economy with easing and accommodative policy."
    result = agent.analyze_sentiment(text)
    assert result.dovish_score > result.hawkish_score
    assert result.net_sentiment < 0


def test_neutral_text_gives_neutral_sentiment(agent):
    """Text with no keywords gives neutral sentiment."""
    text = "The weather today is sunny and the markets are open."
    result = agent.analyze_sentiment(text)
    assert result.usd_bias == "neutral"


def test_hawkish_sentiment_gives_positive_net(agent):
    """Strong hawkish text should produce positive net_sentiment."""
    text = "Inflation is too high, rate hike expected, restrictive tightening measures needed."
    result = agent.analyze_sentiment(text)
    assert result.net_sentiment > 0


def test_dovish_sentiment_gives_negative_net(agent):
    """Strong dovish text should produce negative net_sentiment."""
    text = "Rate cut imminent, easing accommodative support, pivot expected, lower rates ahead."
    result = agent.analyze_sentiment(text)
    assert result.net_sentiment < 0


# ── USD / Gold / Crypto bias ───────────────────────────────────────────────

def test_usd_bullish_on_hawkish_sentiment(agent):
    """Hawkish FED → USD bullish."""
    text = "Inflation is high, the FED will hike rates, tighten the economy restrictive measures."
    result = agent.analyze_sentiment(text)
    assert result.usd_bias == "bullish"


def test_gold_bullish_on_dovish_sentiment(agent):
    """Dovish FED → Gold bullish (opposite of USD)."""
    text = "FED will cut rates, easing and accommodative policy, support for the economy."
    result = agent.analyze_sentiment(text)
    assert result.gold_bias == "bullish"


def test_crypto_bullish_on_dovish_sentiment(agent):
    """Dovish FED → Crypto bullish."""
    text = "FED will cut rates, easing and accommodative policy, support for the economy."
    result = agent.analyze_sentiment(text)
    assert result.crypto_bias == "bullish"


def test_gold_bearish_on_hawkish_sentiment(agent):
    """Hawkish FED → Gold bearish."""
    text = "Inflation is high, the FED will hike rates, tighten the economy restrictive measures."
    result = agent.analyze_sentiment(text)
    assert result.gold_bias == "bearish"


# ── Score adjustments ──────────────────────────────────────────────────────

def test_score_negative_20_during_blackout(agent):
    """score_adjustment returns -20 during FOMC blackout."""
    fomc_day = date.fromisoformat("2025-01-29")
    score = agent.score_adjustment("BTCUSDT", "neutral", as_of=fomc_day)
    assert score == -20


def test_score_positive_10_when_aligned(agent):
    """score_adjustment returns +10 when FED sentiment aligns with trade bias."""
    # Set up a hawkish cached sentiment (USD bullish)
    hawkish_text = "Inflation is high, the FED will hike rates, restrictive tightening policy."
    agent.analyze_sentiment(hawkish_text)

    # USD bullish trade aligns with hawkish sentiment
    far_day = date.fromisoformat("2025-01-10")
    score = agent.score_adjustment("EURUSD=X", "bullish", as_of=far_day)
    # EURUSD with USD bullish → our bias should be bullish and FED agrees
    assert score == 10


def test_score_zero_when_not_aligned(agent):
    """score_adjustment returns 0 when FED sentiment does not align with bias."""
    # Set up dovish sentiment (USD bearish)
    dovish_text = "Rate cut expected, easing and accommodative policy to support markets."
    agent.analyze_sentiment(dovish_text)

    far_day = date.fromisoformat("2025-01-10")
    # Claiming USD is bullish when FED is dovish → not aligned → 0
    score = agent.score_adjustment("EURUSD=X", "bullish", as_of=far_day)
    assert score == 0


def test_score_adjustment_in_range(agent):
    """score_adjustment is always between -20 and +10."""
    texts = [
        "Inflation is high, the FED will hike, restrictive tightening measures.",
        "Rate cut expected, easing and accommodative support.",
        "No strong signals from the FED today.",
    ]
    dates_to_check = [
        date.fromisoformat("2025-01-10"),
        date.fromisoformat("2025-01-29"),  # blackout
        date.fromisoformat("2025-06-01"),
    ]

    for text in texts:
        agent.analyze_sentiment(text)
        for d in dates_to_check:
            for symbol, bias in [("BTCUSDT", "bullish"), ("GC=F", "bearish"), ("EURUSD=X", "neutral")]:
                score = agent.score_adjustment(symbol, bias, as_of=d)
                assert -20 <= score <= 10, f"score {score} out of range for {symbol} {bias} on {d}"


# ── Caching ────────────────────────────────────────────────────────────────

def test_get_cached_sentiment_returns_default_when_no_analysis(agent):
    """get_cached_sentiment returns neutral defaults before any analysis."""
    result = agent.get_cached_sentiment()
    assert isinstance(result, FEDSentimentResult)
    assert result.usd_bias in ("bullish", "bearish", "neutral")


def test_get_cached_sentiment_returns_last_analysis(agent):
    """get_cached_sentiment returns the result of the last analyze_sentiment call."""
    text = "Inflation high, FED will hike, restrictive tightening policy measures needed urgently."
    result = agent.analyze_sentiment(text)
    cached = agent.get_cached_sentiment()
    assert cached.net_sentiment == result.net_sentiment
    assert cached.usd_bias == result.usd_bias


# ── FOMC blackout penalty in analyze_sentiment ─────────────────────────────

def test_score_bonus_negative_20_in_analyze_sentiment_on_fomc_day(agent):
    """analyze_sentiment sets score_bonus=-20 when called on FOMC day."""
    # Mock is_fomc_blackout to return True
    with patch.object(agent, "is_fomc_blackout", return_value=True):
        result = agent.analyze_sentiment("neutral text about the economy")
    assert result.fomc_blackout is True
    assert result.score_bonus == -20


# ── Telegram format ────────────────────────────────────────────────────────

def test_format_telegram_has_fomc_info(agent):
    """Telegram output should contain FOMC-related information."""
    text = "Inflation is high, FED will hike rates with restrictive policy."
    agent.analyze_sentiment(text)
    output = agent.format_telegram()
    assert "FOMC" in output or "fomc" in output.lower()
    assert "FED" in output or "Sentiment" in output


def test_format_telegram_has_bias_info(agent):
    """Telegram output should contain USD/Gold/Crypto bias."""
    text = "Rate cut expected, easing accommodative policy support."
    agent.analyze_sentiment(text)
    output = agent.format_telegram()
    assert "USD" in output
    assert "Gold" in output or "gold" in output.lower()
    assert "Crypto" in output or "crypto" in output.lower()


def test_format_telegram_has_scores(agent):
    """Telegram output should contain hawkish/dovish scores."""
    text = "Inflation high hike restrictive tighten."
    agent.analyze_sentiment(text)
    output = agent.format_telegram()
    assert "Hawkish" in output or "hawkish" in output.lower()
    assert "Dovish" in output or "dovish" in output.lower()


# ── Result field validation ────────────────────────────────────────────────

def test_hawkish_score_in_valid_range(agent):
    """hawkish_score must be between 1 and 10."""
    text = "Inflation inflation hike hike restrictive tighten tighten tighten rate increase."
    result = agent.analyze_sentiment(text)
    assert 1 <= result.hawkish_score <= 10


def test_dovish_score_in_valid_range(agent):
    """dovish_score must be between 1 and 10."""
    text = "Cut cut easing easing support support accommodative pivot lower rates rate cut."
    result = agent.analyze_sentiment(text)
    assert 1 <= result.dovish_score <= 10


def test_rate_hike_probability_in_valid_range(agent):
    """rate_hike_probability must be between 0.0 and 1.0."""
    for text in [
        "Inflation high, hike rates restrictive tightening.",
        "Rate cut easing accommodative dovish support.",
        "No clear signals from markets.",
    ]:
        result = agent.analyze_sentiment(text)
        assert 0.0 <= result.rate_hike_probability <= 1.0
