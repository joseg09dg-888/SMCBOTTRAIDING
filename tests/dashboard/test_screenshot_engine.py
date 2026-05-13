import pytest
import pandas as pd
import io
from unittest.mock import MagicMock, patch
from dashboard.screenshot_engine import (
    ScreenshotEngine, ChartAnnotation, TradeOutcome
)


@pytest.fixture
def sample_df():
    data = {
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "open":   [100 + i*0.5 for i in range(50)],
        "high":   [101 + i*0.5 for i in range(50)],
        "low":    [99 + i*0.5 for i in range(50)],
        "close":  [100.5 + i*0.5 for i in range(50)],
        "volume": [1000 + i*10 for i in range(50)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def annotation():
    return ChartAnnotation(
        symbol="BTCUSDT",
        timeframe="1h",
        entry=124.5,
        stop_loss=122.0,
        take_profit=130.0,
        score=87,
        rr=2.2,
        confidence=0.82,
        trigger="CHoCH + Bullish OB retest",
        ob_zone=(122.0, 123.5),
        fvg_zone=(123.5, 124.0),
    )


def test_annotation_has_required_fields(annotation):
    assert annotation.symbol == "BTCUSDT"
    assert annotation.entry == 124.5
    assert annotation.rr == pytest.approx(2.2, rel=0.01)


def test_annotation_rr_computed(annotation):
    assert annotation.stop_loss < annotation.entry < annotation.take_profit


def test_outcome_enum():
    assert TradeOutcome.WIN.value == "win"
    assert TradeOutcome.LOSS.value == "loss"


def test_engine_generates_bytes(sample_df, annotation):
    engine = ScreenshotEngine()
    with patch.object(engine, "_render_chart", return_value=b"PNG_DATA_FAKE"):
        img_bytes = engine.capture_entry(sample_df, annotation)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0


def test_engine_capture_close_returns_bytes(sample_df, annotation):
    engine = ScreenshotEngine()
    with patch.object(engine, "_render_chart", return_value=b"PNG_DATA_FAKE"):
        img_bytes = engine.capture_close(
            df=sample_df,
            annotation=annotation,
            pnl=23.50,
            pnl_pct=2.35,
            duration_min=263,
            outcome=TradeOutcome.WIN,
            win_rate=61.3,
        )
    assert isinstance(img_bytes, bytes)


def test_caption_entry_format(annotation):
    engine = ScreenshotEngine()
    caption = engine.build_entry_caption(annotation)
    assert "BTCUSDT" in caption
    assert "124.5" in caption or "124" in caption
    assert "87" in caption  # score


def test_caption_close_win():
    engine = ScreenshotEngine()
    caption = engine.build_close_caption(
        symbol="BTCUSDT",
        pnl=23.50,
        pnl_pct=2.35,
        duration_min=263,
        outcome=TradeOutcome.WIN,
        win_rate=61.3,
    )
    assert "23.50" in caption or "23" in caption
    assert "win" in caption.lower() or "cerrado" in caption.lower() or "+" in caption


def test_caption_close_loss():
    engine = ScreenshotEngine()
    caption = engine.build_close_caption(
        symbol="EURUSD",
        pnl=-12.0,
        pnl_pct=-1.2,
        duration_min=90,
        outcome=TradeOutcome.LOSS,
        win_rate=58.0,
    )
    assert "-12" in caption or "loss" in caption.lower() or "cerrado" in caption.lower()


def test_engine_no_crash_without_mplfinance(sample_df, annotation):
    """Engine must not crash even if mplfinance is not installed."""
    engine = ScreenshotEngine()
    # Force the no-mplfinance path
    with patch("dashboard.screenshot_engine.HAS_MPLFINANCE", False):
        with patch.object(engine, "_render_fallback", return_value=b"FALLBACK"):
            result = engine.capture_entry(sample_df, annotation)
    assert isinstance(result, bytes)
