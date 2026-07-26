import pytest
from smc.sentiment import SentimentAnalyzer, SentimentResult


def test_sentiment_no_signals():
    sa = SentimentAnalyzer()
    result = sa.analyze(symbol="BTCUSDT", glint_signals=[])
    assert isinstance(result, SentimentResult)
    assert result.score == 0
    assert result.component_score <= 20


def test_sentiment_score_disabled_pending_real_direction_classification():
    """BUG-GLINT-ALIGNMENT-ALWAYS-POSITIVE (2026-07-26): the score used to
    award up to 20pts based on "alignment" that could never actually be
    negative (no real bullish/bearish text classification exists), so a
    bearish headline could score as "aligned" for a bullish trade. Disabled
    until real direction classification exists -- component_score is 0
    even for a highly relevant, critical-impact signal."""
    sa = SentimentAnalyzer()
    signals = [
        {
            "impact": "Critical",
            "category": "Crypto",
            "relevance_score": 9.5,
            "source_tier": 1,
            "text": "SEC aprueba ETF Bitcoin al contado",
        }
    ]
    result = sa.analyze(symbol="BTCUSDT", glint_signals=signals, bias="bullish")
    assert result.component_score == 0
    assert result.signal_count == 1


def test_sentiment_conflicting_signals_reduce_score():
    sa = SentimentAnalyzer()
    signals = [
        {"impact": "High",   "category": "Crypto", "relevance_score": 8.0,
         "source_tier": 1, "text": "Bullish news"},
        {"impact": "High",   "category": "Crypto", "relevance_score": 8.0,
         "source_tier": 1, "text": "Bearish news"},
    ]
    r_conflict = sa.analyze(symbol="BTCUSDT", glint_signals=signals, bias="bullish")
    r_single   = sa.analyze(symbol="BTCUSDT",
                            glint_signals=[signals[0]], bias="bullish")
    assert r_single.component_score >= r_conflict.component_score


def test_sentiment_score_bounded():
    sa = SentimentAnalyzer()
    signals = [
        {"impact": "Critical", "category": "Crypto", "relevance_score": 10,
         "source_tier": 1, "text": "News"} for _ in range(10)
    ]
    result = sa.analyze(symbol="BTCUSDT", glint_signals=signals, bias="bullish")
    assert 0 <= result.component_score <= 20


def test_sentiment_result_has_reason():
    sa = SentimentAnalyzer()
    result = sa.analyze(symbol="BTCUSDT", glint_signals=[], bias="neutral")
    assert isinstance(result.reason, str)
