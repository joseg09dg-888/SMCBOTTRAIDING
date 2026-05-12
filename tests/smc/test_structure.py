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
    ms = MarketStructure(bullish_trend_data)
    result = ms.analyze()
    assert result.structure_type == StructureType.BULLISH_TREND
    assert result.higher_highs > 0
    assert result.higher_lows > 0


def test_detects_bearish_trend(bearish_trend_data):
    ms = MarketStructure(bearish_trend_data)
    result = ms.analyze()
    assert result.structure_type == StructureType.BEARISH_TREND
    assert result.lower_highs > 0
    assert result.lower_lows > 0


def test_detects_bos(bullish_trend_data):
    ms = MarketStructure(bullish_trend_data)
    bos_list = ms.detect_bos()
    assert len(bos_list) > 0
    assert bos_list[0]["type"] == "BOS"


def test_detects_choch(bearish_trend_data):
    extra = pd.DataFrame({"high": [115], "low": [108], "close": [113]})
    data = pd.concat([bearish_trend_data, extra], ignore_index=True)
    ms = MarketStructure(data)
    choch_list = ms.detect_choch()
    assert len(choch_list) > 0
    assert choch_list[0]["type"] == "CHoCH"


def test_summary_returns_string(bullish_trend_data):
    ms = MarketStructure(bullish_trend_data)
    s = ms.summary()
    assert isinstance(s, str)
    assert "Estructura" in s or "BULLISH" in s.upper()
