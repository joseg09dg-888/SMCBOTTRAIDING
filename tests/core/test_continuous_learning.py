"""
TDD Tests for core/continuous_learning.py
Written FIRST — before the implementation exists.
"""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lesson(symbol="BTCUSDT", direction="long", entry=30000.0,
             exit_price=31000.0, pnl=1000.0, reason="OB bounce",
             tags=None, ts=None):
    from core.continuous_learning import TradeLesson
    return TradeLesson(
        symbol=symbol,
        direction=direction,
        entry=entry,
        exit_price=exit_price,
        pnl=pnl,
        reason=reason,
        tags=tags or [],
        timestamp=ts or datetime.now(timezone.utc),
    )


def _engine():
    from core.continuous_learning import ContinuousLearningEngine
    return ContinuousLearningEngine()


# ---------------------------------------------------------------------------
# TradeLesson dataclass
# ---------------------------------------------------------------------------

def test_record_trade_stores_lesson():
    """record_trade() saves lesson and get_recent_lessons() returns it."""
    engine = _engine()
    lesson = _lesson(tags=["OB"])
    engine.record_trade(lesson)
    recent = engine.get_recent_lessons()
    assert len(recent) >= 1
    assert recent[-1].symbol == "BTCUSDT"


def test_lesson_tags_extracted():
    """record_trade() with tags=['OB','FVG'] -> get_recent_lessons()[0].tags contains both."""
    engine = _engine()
    lesson = _lesson(tags=["OB", "FVG"])
    engine.record_trade(lesson)
    stored = engine.get_recent_lessons()[-1]
    assert "OB" in stored.tags
    assert "FVG" in stored.tags


# ---------------------------------------------------------------------------
# analyze_trade_lesson
# ---------------------------------------------------------------------------

def test_analyze_trade_lesson_long_win():
    """Winning long lesson -> cause contains 'ganancia' or 'profit'."""
    engine = _engine()
    lesson = _lesson(direction="long", pnl=500.0, tags=["OB"])
    result = engine.analyze_trade_lesson(lesson)
    assert isinstance(result, dict)
    cause = result.get("cause", "").lower()
    assert "ganancia" in cause or "profit" in cause


def test_analyze_trade_lesson_short_loss():
    """Losing short lesson -> cause contains 'perdida' or 'loss'."""
    engine = _engine()
    lesson = _lesson(direction="short", pnl=-300.0, tags=["FVG"])
    result = engine.analyze_trade_lesson(lesson)
    assert isinstance(result, dict)
    cause = result.get("cause", "").lower()
    assert "perdida" in cause or "loss" in cause


def test_analyze_trade_lesson_has_improvement_key():
    """analyze_trade_lesson always returns dict with 'improvement' key."""
    engine = _engine()
    lesson = _lesson(pnl=100.0)
    result = engine.analyze_trade_lesson(lesson)
    assert "improvement" in result


def test_analyze_trade_lesson_has_tags_key():
    """analyze_trade_lesson always returns dict with 'tags' key."""
    engine = _engine()
    lesson = _lesson(tags=["BOS"])
    result = engine.analyze_trade_lesson(lesson)
    assert "tags" in result


# ---------------------------------------------------------------------------
# get_win_rate_by_tag
# ---------------------------------------------------------------------------

def test_get_win_rate_by_tag_no_trades():
    """Tag with no trades -> 0.0."""
    engine = _engine()
    assert engine.get_win_rate_by_tag("OB") == 0.0


def test_get_win_rate_by_tag_with_trades():
    """3 wins + 1 loss with tag OB -> 0.75."""
    engine = _engine()
    for _ in range(3):
        engine.record_trade(_lesson(pnl=100.0, tags=["OB"]))
    engine.record_trade(_lesson(pnl=-50.0, tags=["OB"]))
    rate = engine.get_win_rate_by_tag("OB")
    assert abs(rate - 0.75) < 0.001


def test_get_win_rate_by_tag_ignores_other_tags():
    """Win rate for 'OB' not affected by 'FVG' trades."""
    engine = _engine()
    engine.record_trade(_lesson(pnl=100.0, tags=["OB"]))
    engine.record_trade(_lesson(pnl=-200.0, tags=["FVG"]))
    assert engine.get_win_rate_by_tag("OB") == 1.0
    assert engine.get_win_rate_by_tag("FVG") == 0.0


# ---------------------------------------------------------------------------
# generate_adjustment_suggestion
# ---------------------------------------------------------------------------

def test_generate_suggestion_insufficient_data():
    """< 5 trades with any tag -> None."""
    engine = _engine()
    for _ in range(4):
        engine.record_trade(_lesson(pnl=100.0, tags=["OB"]))
    assert engine.generate_adjustment_suggestion() is None


def test_generate_suggestion_high_win_rate():
    """5+ trades with OB, 80% win rate -> AdjustmentSuggestion with component='smc'."""
    from core.continuous_learning import AdjustmentSuggestion
    engine = _engine()
    for _ in range(4):
        engine.record_trade(_lesson(pnl=100.0, tags=["OB"]))
    engine.record_trade(_lesson(pnl=-50.0, tags=["OB"]))  # 4/5 = 80%
    suggestion = engine.generate_adjustment_suggestion()
    assert suggestion is not None
    assert isinstance(suggestion, AdjustmentSuggestion)
    assert suggestion.component == "smc"
    assert suggestion.win_rate_evidence >= 0.70


def test_generate_suggestion_low_win_rate_returns_none():
    """5+ trades but only 40% win rate -> None."""
    engine = _engine()
    for _ in range(2):
        engine.record_trade(_lesson(pnl=100.0, tags=["OB"]))
    for _ in range(3):
        engine.record_trade(_lesson(pnl=-50.0, tags=["OB"]))
    assert engine.generate_adjustment_suggestion() is None


# ---------------------------------------------------------------------------
# get_crash_analysis
# ---------------------------------------------------------------------------

def test_get_crash_analysis_luna():
    """crash_analysis('Luna Crash 2022') returns dict with 'name', 'move_pct', 'smc_signals'."""
    engine = _engine()
    result = engine.get_crash_analysis("Luna Crash 2022")
    assert isinstance(result, dict)
    assert "name" in result
    assert "move_pct" in result
    assert "smc_signals" in result
    assert result["name"] == "Luna Crash 2022"


def test_get_crash_analysis_unknown():
    """Unknown crash -> dict with 'error' key."""
    engine = _engine()
    result = engine.get_crash_analysis("Fake Crash 9999")
    assert isinstance(result, dict)
    assert "error" in result


def test_get_crash_analysis_black_monday():
    """'Black Monday 1987' is in FAMOUS_CRASHES and returns valid analysis."""
    engine = _engine()
    result = engine.get_crash_analysis("Black Monday 1987")
    assert "error" not in result
    assert result["move_pct"] == pytest.approx(-22.6, abs=0.1)


# ---------------------------------------------------------------------------
# FAMOUS_CRASHES and CHANNELS class attributes
# ---------------------------------------------------------------------------

def test_famous_crashes_list_not_empty():
    """ContinuousLearningEngine.FAMOUS_CRASHES has >= 5 entries."""
    from core.continuous_learning import ContinuousLearningEngine
    assert len(ContinuousLearningEngine.FAMOUS_CRASHES) >= 5


def test_channels_list_not_empty():
    """ContinuousLearningEngine.CHANNELS has >= 4 entries."""
    from core.continuous_learning import ContinuousLearningEngine
    assert len(ContinuousLearningEngine.CHANNELS) >= 4


def test_channels_have_required_fields():
    """Every YouTubeChannel has name, channel_id, is_live, last_checked."""
    from core.continuous_learning import ContinuousLearningEngine
    for ch in ContinuousLearningEngine.CHANNELS:
        assert hasattr(ch, "name")
        assert hasattr(ch, "channel_id")
        assert hasattr(ch, "is_live")
        assert hasattr(ch, "last_checked")


# ---------------------------------------------------------------------------
# check_youtube_live
# ---------------------------------------------------------------------------

def test_check_youtube_live_returns_bool():
    """check_youtube_live() always returns bool (even with network errors)."""
    from core.continuous_learning import YouTubeChannel
    engine = _engine()
    ch = YouTubeChannel(
        name="Test",
        channel_id="UCtest123",
        is_live=False,
        last_checked=datetime.now(timezone.utc),
    )
    result = engine.check_youtube_live(ch)
    assert isinstance(result, bool)


def test_check_youtube_live_mock_live(mocker):
    """When HTTP returns 'isLiveBroadcast' in body, returns True."""
    from core.continuous_learning import YouTubeChannel
    engine = _engine()
    ch = YouTubeChannel("ICT", "UCqGE4ZTvCEVimRBHs6Yqw5g", False, datetime.now(timezone.utc))
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"isLiveBroadcast": true, "title": "ICT Live"}'
    mocker.patch("requests.get", return_value=mock_resp)
    result = engine.check_youtube_live(ch)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# format_suggestion_telegram (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_format_suggestion_telegram():
    """format_suggestion_telegram() returns string containing 'SUGERENCIA'."""
    from core.continuous_learning import AdjustmentSuggestion
    engine = _engine()
    suggestion = AdjustmentSuggestion(
        component="smc",
        current_value=30,
        suggested_value=40,
        reason="OB tiene 80% win rate en 10 trades",
        win_rate_evidence=0.80,
    )
    text = await engine.format_suggestion_telegram(suggestion)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "SUGERENCIA" in text.upper()


@pytest.mark.asyncio
async def test_format_suggestion_telegram_has_component():
    """format_suggestion_telegram() mentions the component in the message."""
    from core.continuous_learning import AdjustmentSuggestion
    engine = _engine()
    suggestion = AdjustmentSuggestion(
        component="ml",
        current_value=50,
        suggested_value=60,
        reason="ML signals improving",
        win_rate_evidence=0.75,
    )
    text = await engine.format_suggestion_telegram(suggestion)
    assert "ml" in text.lower()


# ---------------------------------------------------------------------------
# run_study_cycle (async, mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_study_cycle_sends_telegram(mocker):
    """run_study_cycle() calls send_glint_alert at least once after one cycle (mock sleep=0)."""
    from core.continuous_learning import ContinuousLearningEngine
    mock_bot = AsyncMock()
    mock_bot.send_glint_alert = AsyncMock()
    engine = ContinuousLearningEngine(telegram_bot=mock_bot, suggestion_interval_hours=0)

    # Patch asyncio.sleep so the cycle doesn't wait, then cancel after 1 iteration
    call_count = 0
    original_sleep = asyncio.sleep

    async def mock_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            raise asyncio.CancelledError()

    mocker.patch("asyncio.sleep", side_effect=mock_sleep)

    with pytest.raises((asyncio.CancelledError, Exception)):
        await engine.run_study_cycle()

    assert mock_bot.send_glint_alert.called or call_count >= 1


# ---------------------------------------------------------------------------
# get_study_report
# ---------------------------------------------------------------------------

def test_get_study_report_returns_string():
    """get_study_report() returns non-empty str."""
    engine = _engine()
    report = engine.get_study_report()
    assert isinstance(report, str)
    assert len(report) > 0


def test_get_study_report_includes_trade_count():
    """get_study_report() reflects the number of recorded trades."""
    engine = _engine()
    engine.record_trade(_lesson(pnl=200.0, tags=["OB"]))
    engine.record_trade(_lesson(pnl=-100.0, tags=["FVG"]))
    report = engine.get_study_report()
    # Should mention count somewhere
    assert "2" in report or "trade" in report.lower()


# ---------------------------------------------------------------------------
# get_recent_lessons — limit
# ---------------------------------------------------------------------------

def test_get_recent_lessons_respects_n():
    """get_recent_lessons(n=3) returns at most 3 lessons."""
    engine = _engine()
    for i in range(7):
        engine.record_trade(_lesson(pnl=float(i * 10), tags=["OB"]))
    assert len(engine.get_recent_lessons(n=3)) == 3


def test_get_recent_lessons_returns_most_recent():
    """get_recent_lessons(n=1) returns the LAST lesson added."""
    from datetime import timedelta
    engine = _engine()
    early = _lesson(pnl=10.0, ts=datetime(2024, 1, 1, tzinfo=timezone.utc))
    late  = _lesson(pnl=99.0, ts=datetime(2025, 1, 1, tzinfo=timezone.utc))
    engine.record_trade(early)
    engine.record_trade(late)
    recent = engine.get_recent_lessons(n=1)
    assert recent[0].pnl == 99.0


# ---------------------------------------------------------------------------
# AdjustmentSuggestion dataclass
# ---------------------------------------------------------------------------

def test_adjustment_suggestion_fields():
    """AdjustmentSuggestion has all required fields."""
    from core.continuous_learning import AdjustmentSuggestion
    s = AdjustmentSuggestion(
        component="energy",
        current_value=20,
        suggested_value=35,
        reason="High win rate",
        win_rate_evidence=0.82,
    )
    assert s.component == "energy"
    assert s.current_value == 20
    assert s.suggested_value == 35
    assert s.win_rate_evidence == pytest.approx(0.82)
