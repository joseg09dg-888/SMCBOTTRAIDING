import pytest, numpy as np
from agents.quant_regime import RegimeDetector, MarketRegime, RegimeAnalysis

def trend_up(n=100,drift=0.001):
    rng=np.random.default_rng(42); r=rng.normal(drift,0.005,n)
    p=[100.0]
    for x in r: p.append(p[-1]*(1+x))
    return p
def ranging(n=100):
    rng=np.random.default_rng(42); r=rng.normal(0,0.005,n)
    p=[100.0]
    for x in r: p.append(p[-1]*(1+x))
    return p
def high_vol(n=100):
    rng=np.random.default_rng(42); r=rng.normal(0,0.06,n)
    p=[100.0]
    for x in r: p.append(p[-1]*(1+x))
    return p
def trend_down(n=100,drift=-0.002):
    rng=np.random.default_rng(42); r=rng.normal(drift,0.005,n)
    p=[100.0]
    for x in r: p.append(p[-1]*(1+x))
    return p

def test_detect_returns_analysis():
    assert isinstance(RegimeDetector().detect(trend_up()), RegimeAnalysis)
def test_trending_up_detected():
    r = RegimeDetector(lookback=80).detect(trend_up(drift=0.003))
    assert r.regime in (MarketRegime.TRENDING_UP, MarketRegime.HIGH_VOL)
def test_trending_down_detected():
    r = RegimeDetector(lookback=80).detect(trend_down(drift=-0.003))
    assert r.regime in (MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOL)
def test_high_vol_detected():
    assert RegimeDetector(lookback=80).detect(high_vol()).regime == MarketRegime.HIGH_VOL
def test_ranging_detected():
    assert RegimeDetector(lookback=80).detect(ranging()).regime == MarketRegime.RANGING
def test_confidence_range():
    r = RegimeDetector().detect(trend_up())
    assert 0.0 <= r.confidence <= 1.0
def test_volatility_positive():
    assert RegimeDetector().detect(high_vol()).volatility > 0
def test_high_vol_low_risk():
    r = RegimeDetector(lookback=80).detect(high_vol())
    if r.regime == MarketRegime.HIGH_VOL:
        assert r.recommended_risk_multiplier <= 0.25
def test_trending_rr():
    r = RegimeDetector(lookback=80).detect(trend_up(drift=0.003))
    if r.regime == MarketRegime.TRENDING_UP:
        assert r.recommended_rr >= 2.0
def test_ranging_lower_rr():
    r = RegimeDetector(lookback=80).detect(ranging())
    if r.regime == MarketRegime.RANGING:
        assert r.recommended_rr <= 2.0
def test_regime_history():
    history = RegimeDetector(lookback=50).get_regime_history(trend_up(n=200), step=20)
    assert isinstance(history, list) and len(history) > 0
def test_dominant_regime():
    assert isinstance(RegimeDetector(50).get_dominant_regime(trend_up(n=200,drift=0.003)), MarketRegime)
def test_win_rate_trending_up():
    assert RegimeDetector.regime_win_rate_estimate(MarketRegime.TRENDING_UP) == pytest.approx(0.65)
def test_win_rate_high_vol():
    assert RegimeDetector.regime_win_rate_estimate(MarketRegime.HIGH_VOL) < 0.50
def test_all_regimes_have_wr():
    for r in MarketRegime:
        wr = RegimeDetector.regime_win_rate_estimate(r)
        assert 0.0 < wr <= 1.0
def test_short_series_no_crash():
    r = RegimeDetector(lookback=50).detect([100.0,101.0,99.0])
    assert isinstance(r, RegimeAnalysis)
def test_flat_prices():
    r = RegimeDetector(lookback=50).detect([100.0]*100)
    assert r.regime == MarketRegime.RANGING
