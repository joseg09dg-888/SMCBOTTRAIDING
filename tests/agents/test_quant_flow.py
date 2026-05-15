# tests/agents/test_quant_flow.py
import pytest
from agents.quant_flow import OrderFlowAnalyzer, OrderFlowSignal

# ── Imbalance ─────────────────────────────────────────────────────────────


def test_imbalance_all_bids():
    result = OrderFlowAnalyzer.calculate_imbalance([100, 200, 150], [])
    assert result == pytest.approx(1.0)


def test_imbalance_all_asks():
    result = OrderFlowAnalyzer.calculate_imbalance([], [100, 200])
    assert result == pytest.approx(-1.0)


def test_imbalance_balanced():
    result = OrderFlowAnalyzer.calculate_imbalance([100, 100], [100, 100])
    assert result == pytest.approx(0.0)


def test_imbalance_empty():
    assert OrderFlowAnalyzer.calculate_imbalance([], []) == 0.0


def test_imbalance_range():
    result = OrderFlowAnalyzer.calculate_imbalance([50, 60, 40], [30, 20, 10])
    assert -1.0 <= result <= 1.0


# ── VPIN ──────────────────────────────────────────────────────────────────


def test_vpin_perfectly_informed():
    """All buys → VPIN = 1.0"""
    buys  = [100.0] * 50
    sells = [0.0]   * 50
    v = OrderFlowAnalyzer.calculate_vpin(buys, sells, bucket_size=10)
    assert v == pytest.approx(1.0)


def test_vpin_balanced():
    """Equal buy/sell → VPIN ≈ 0"""
    buys  = [100.0] * 50
    sells = [100.0] * 50
    v = OrderFlowAnalyzer.calculate_vpin(buys, sells, bucket_size=10)
    assert v == pytest.approx(0.0)


def test_vpin_range():
    import numpy as np
    rng  = np.random.default_rng(42)
    buys  = list(rng.uniform(0, 200, 100))
    sells = list(rng.uniform(0, 200, 100))
    v = OrderFlowAnalyzer.calculate_vpin(buys, sells)
    assert 0.0 <= v <= 1.0


def test_vpin_empty():
    assert OrderFlowAnalyzer.calculate_vpin([], []) == 0.0


# ── Spread ────────────────────────────────────────────────────────────────


def test_spread_pct_normal():
    s = OrderFlowAnalyzer.calculate_spread_pct(100.0, 100.1)
    assert s == pytest.approx(0.001, abs=0.0001)


def test_spread_pct_zero_bid():
    assert OrderFlowAnalyzer.calculate_spread_pct(0.0, 100.0) == 0.0


# ── Classify pressure ─────────────────────────────────────────────────────


def test_classify_strong_buy():
    pressure, pts = OrderFlowAnalyzer.classify_pressure(0.6)
    assert pressure == "strong_buy"
    assert pts == 8


def test_classify_strong_sell():
    pressure, pts = OrderFlowAnalyzer.classify_pressure(-0.6)
    assert pressure == "strong_sell"
    assert pts == -8


def test_classify_neutral():
    pressure, pts = OrderFlowAnalyzer.classify_pressure(0.0)
    assert pressure == "neutral"
    assert pts == 0


# ── Analyze ───────────────────────────────────────────────────────────────


def test_analyze_returns_signal():
    analyzer = OrderFlowAnalyzer()
    signal   = analyzer.analyze([100, 200], [50, 30], best_bid=99.9, best_ask=100.1)
    assert isinstance(signal, OrderFlowSignal)


def test_analyze_toxic_flow():
    """High VPIN → toxic_flow=True"""
    buys  = [200.0] * 50
    sells = [0.0]   * 50
    analyzer = OrderFlowAnalyzer()
    signal   = analyzer.analyze(
        [200] * 20,
        [0]   * 20,
        buy_volumes  = buys,
        sell_volumes = sells,
    )
    assert signal.toxic_flow is True


def test_analyze_pts_range():
    analyzer = OrderFlowAnalyzer()
    signal   = analyzer.analyze([100], [100])
    assert -10 <= signal.pts <= 10


# ── Market impact ─────────────────────────────────────────────────────────


def test_market_impact_small_order():
    impact = OrderFlowAnalyzer.estimate_market_impact(1000, 1_000_000, 0.02)
    assert impact < 0.01  # tiny order vs large ADV


def test_market_impact_zero_volume():
    assert OrderFlowAnalyzer.estimate_market_impact(1000, 0, 0.02) == 0.0


# ── Iceberg ───────────────────────────────────────────────────────────────


def test_iceberg_high_executed():
    p = OrderFlowAnalyzer.detect_iceberg_probability(10, 1000)
    assert p > 0.9


def test_iceberg_normal_ratio():
    p = OrderFlowAnalyzer.detect_iceberg_probability(100, 80)
    assert p < 0.9
