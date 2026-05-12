import pytest
import pandas as pd
from smc.volume_profile import VolumeProfile, AnchoredVWAP


@pytest.fixture
def ohlcv():
    data = {
        "high":   [105, 108, 110, 107, 112, 115, 113, 118],
        "low":    [100, 103, 106, 102, 108, 110, 109, 114],
        "close":  [103, 107, 108, 104, 111, 113, 110, 117],
        "volume": [1000, 1500, 800, 2000, 3000, 1200, 500, 2500],
    }
    return pd.DataFrame(data)


def test_poc_is_highest_volume_price(ohlcv):
    vp = VolumeProfile(ohlcv)
    result = vp.calculate()
    assert result["poc"] > 0
    assert result["vah"] > result["poc"] > result["val"]


def test_value_area_contains_70_percent_volume(ohlcv):
    vp = VolumeProfile(ohlcv)
    result = vp.calculate()
    assert 0.65 <= result["value_area_pct"] <= 1.0


def test_anchored_vwap_returns_series(ohlcv):
    av = AnchoredVWAP(ohlcv, anchor_index=0)
    vwap = av.calculate()
    assert len(vwap) == len(ohlcv)
    assert all(v > 0 for v in vwap)


def test_vwap_above_below(ohlcv):
    av = AnchoredVWAP(ohlcv, anchor_index=0)
    vwap = av.calculate()
    last_vwap = vwap[-1]
    last_price = ohlcv["close"].iloc[-1]
    result = av.is_price_above_vwap(last_price)
    assert isinstance(result, bool)
    assert result == (last_price > last_vwap)
