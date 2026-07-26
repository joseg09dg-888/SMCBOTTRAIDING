import pytest
import pandas as pd
from smc.structure import MarketStructure, StructureType, SwingPoint


@pytest.fixture
def bullish_trend_data():
    highs  = [100, 110, 108, 120, 115, 130]
    lows   = [95,  102, 100, 112, 110, 122]
    closes = [99,  108, 104, 118, 113, 128]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


@pytest.fixture
def bearish_trend_data():
    highs  = [130, 120, 122, 110, 112, 100]
    lows   = [122, 112, 114, 102, 104, 92]
    closes = [125, 115, 118, 105, 108, 95]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def test_detects_bullish_trend(bullish_trend_data):
    ms = MarketStructure(bullish_trend_data, swing_lookback=1)
    result = ms.analyze()
    assert result.structure_type == StructureType.BULLISH_TREND
    assert result.higher_highs > 0
    assert result.higher_lows > 0


def test_detects_bearish_trend(bearish_trend_data):
    ms = MarketStructure(bearish_trend_data, swing_lookback=1)
    result = ms.analyze()
    assert result.structure_type == StructureType.BEARISH_TREND
    assert result.lower_highs > 0
    assert result.lower_lows > 0


def test_detects_bos(bullish_trend_data):
    ms = MarketStructure(bullish_trend_data, swing_lookback=1)
    bos_list = ms.detect_bos()
    assert len(bos_list) > 0
    assert bos_list[0]["type"] == "BOS"


def test_detects_choch(bearish_trend_data):
    extra = pd.DataFrame({"high": [115], "low": [108], "close": [113]})
    data = pd.concat([bearish_trend_data, extra], ignore_index=True)
    ms = MarketStructure(data, swing_lookback=1)
    choch_list = ms.detect_choch()
    assert len(choch_list) > 0
    assert choch_list[0]["type"] == "CHoCH"


def test_summary_returns_string(bullish_trend_data):
    ms = MarketStructure(bullish_trend_data)
    s = ms.summary()
    assert isinstance(s, str)
    assert "Estructura" in s or "BULLISH" in s.upper()


def test_bos_events_sorted_by_confirmation_not_swing_formation_order():
    """BUG-BOS-ORDER (2026-07-26): an HH that forms early (idx2, level=110)
    but doesn't get broken until idx15 used to be appended BEFORE an LL that
    forms later (idx10, level=80) but breaks quickly at idx12 -- because
    detect_bos() iterated swings in formation order, not confirmation order.
    bos_list[-1] returned the stale bearish@12 event instead of the truly
    most-recent bullish@15 event. core/supervisor.py uses exactly bos_list[-1]
    to decide LONG/SHORT when structural bias is neutral."""
    highs  = [100, 102, 110, 105, 104, 103, 101, 100, 98, 96, 94, 96, 90, 92, 100, 118, 116, 114, 112, 110]
    lows   = [95,  97,  100, 98,  97,  95,  93,  90,  88, 85, 80, 83, 75, 80, 85,  110, 108, 106, 104, 102]
    closes = [98,  100, 105, 100, 99,  97,  95,  92,  90, 87, 85, 84, 78, 88, 95,  115, 112, 110, 108, 106]
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    ms = MarketStructure(df, swing_lookback=1)
    bos_list = ms.detect_bos()

    confirmed_ats = [e["confirmed_at"] for e in bos_list]
    assert confirmed_ats == sorted(confirmed_ats), (
        f"bos_list not sorted by confirmed_at: {confirmed_ats}"
    )
    # The bearish break (idx10 LL, confirmed at candle 12) happened
    # chronologically BEFORE the bullish break (idx2 HH, confirmed at candle
    # 15) even though the HH swing formed first -- the truly most recent
    # confirmed event must be the bullish one.
    assert bos_list[-1]["direction"] == "bullish"
    assert bos_list[-1]["confirmed_at"] == 15


def test_choch_events_are_sorted_by_confirmed_at(bearish_trend_data):
    extra = pd.DataFrame({"high": [115], "low": [108], "close": [113]})
    data = pd.concat([bearish_trend_data, extra], ignore_index=True)
    ms = MarketStructure(data, swing_lookback=1)
    choch_list = ms.detect_choch()
    confirmed_ats = [e["confirmed_at"] for e in choch_list]
    assert confirmed_ats == sorted(confirmed_ats)
