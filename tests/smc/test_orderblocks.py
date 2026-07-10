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


def test_zero_close_bullish_no_crash():
    data = {
        "open":   [0.0, 0.0, 1.0, 1.0],
        "high":   [0.0, 0.0, 1.1, 1.1],
        "low":    [0.0, 0.0, 0.9, 0.9],
        "close":  [0.0, 0.0, 1.0, 1.0],
        "volume": [0, 0, 100, 100],
    }
    df = pd.DataFrame(data)
    ob = OrderBlockDetector(df)
    blocks = ob.find_bullish_obs()
    assert isinstance(blocks, list)


def test_zero_close_bearish_no_crash():
    data = {
        "open":   [0.0, 0.0, 1.0, 0.9],
        "high":   [0.0, 0.0, 1.1, 1.0],
        "low":    [0.0, 0.0, 0.8, 0.7],
        "close":  [0.0, 0.0, 0.9, 0.8],
        "volume": [0, 0, 100, 100],
    }
    df = pd.DataFrame(data)
    ob = OrderBlockDetector(df)
    blocks = ob.find_bearish_obs()
    assert isinstance(blocks, list)


# ── BUG-OB-FOREX-DEAD regression: forex-scale prices must still detect OBs ──

def _forex_scale_df(n=60):
    """~EURUSD-like H1 series: small pip-scale moves, then one real
    displacement impulse -- the 1.5% fixed-percent threshold would never
    fire here (verified: max real EURUSD H1 move over 200 bars was 0.415%),
    but the ATR-relative check should."""
    import numpy as np
    rng = np.random.default_rng(7)
    base = 1.1400
    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] + rng.normal(0, 0.0003))
    # Inject one clear bearish-then-bullish-impulse pair near the end
    closes[-3] = closes[-4] - 0.0010  # bearish candle (the OB)
    closes[-2] = closes[-3] + 0.0060  # strong bullish impulse next candle
    opens = [closes[i - 1] if i > 0 else closes[0] for i in range(n)]
    highs = [max(o, c) + 0.0003 for o, c in zip(opens, closes)]
    lows  = [min(o, c) - 0.0003 for o, c in zip(opens, closes)]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


def test_bullish_ob_detected_at_forex_scale():
    df = _forex_scale_df()
    ob = OrderBlockDetector(df)
    blocks = ob.find_bullish_obs()
    assert len(blocks) >= 1, "OB detector debe encontrar impulsos reales a escala forex, no solo cripto"


def test_atr_fallback_to_percentage_with_insufficient_data():
    # Con muy pocas velas (sin ATR14 posible), debe usar el umbral porcentual
    # de siempre -- no debe crashear ni comportarse distinto silenciosamente.
    ob = OrderBlockDetector(pd.DataFrame({
        "open": [100, 102], "high": [103, 105], "low": [99, 101], "close": [102, 103],
    }))
    assert ob._atr() is None
    blocks = ob.find_bullish_obs()
    assert isinstance(blocks, list)
