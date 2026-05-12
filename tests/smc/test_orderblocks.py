import pytest
import pandas as pd
from smc.orderblocks import OrderBlockDetector, FVGDetector


@pytest.fixture
def sample_ohlc():
    data = {
        "open":   [100, 102, 101, 99,  110, 108, 115, 113],
        "high":   [103, 105, 103, 100, 115, 112, 120, 116],
        "low":    [99,  101, 99,  97,  108, 107, 113, 111],
        "close":  [102, 103, 100, 110, 112, 109, 118, 114],
        "volume": [1000, 1200, 800, 2500, 3000, 900, 3500, 1100],
    }
    return pd.DataFrame(data)


def test_detects_bullish_order_block(sample_ohlc):
    ob = OrderBlockDetector(sample_ohlc)
    blocks = ob.find_bullish_obs()
    assert len(blocks) >= 1
    assert all(b["type"] == "bullish_ob" for b in blocks)
    assert all("zone_high" in b and "zone_low" in b for b in blocks)


def test_detects_bearish_order_block(sample_ohlc):
    ob = OrderBlockDetector(sample_ohlc)
    blocks = ob.find_bearish_obs()
    assert isinstance(blocks, list)


def test_detects_fvg_bullish(sample_ohlc):
    fvg = FVGDetector(sample_ohlc)
    gaps = fvg.find_bullish_fvg()
    assert isinstance(gaps, list)
    assert all("gap_high" in g and "gap_low" in g for g in gaps)


def test_fvg_gap_is_real(sample_ohlc):
    fvg = FVGDetector(sample_ohlc)
    gaps = fvg.find_bullish_fvg()
    for g in gaps:
        assert g["gap_high"] > g["gap_low"], "FVG inválido: high <= low"


def test_price_in_ob(sample_ohlc):
    ob = OrderBlockDetector(sample_ohlc)
    blocks = ob.find_bullish_obs()
    if blocks:
        b = blocks[0]
        mid = (b["zone_high"] + b["zone_low"]) / 2
        assert ob.is_price_in_ob(mid, b)
        assert not ob.is_price_in_ob(b["zone_high"] + 10, b)
