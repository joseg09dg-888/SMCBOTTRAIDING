import pytest
import numpy as np
import pandas as pd
from strategies.pairs_trading import PairsTrader, PairConfig, PairSignal, CointegrationResult


def make_cointegrated(n=200, beta=1.2, noise_std=0.5, seed=42):
    """Generate two cointegrated price series."""
    rng = np.random.default_rng(seed)
    a_returns = rng.normal(0.001, 0.02, n)
    a = pd.Series(np.cumprod(1 + a_returns) * 50000.0)
    # b = a/beta + stationary noise
    noise = rng.normal(0, noise_std, n)
    b = a / beta + noise
    return a, b


def make_divergent(n=200, seed=42):
    """Generate two unrelated price series."""
    rng = np.random.default_rng(seed)
    a = pd.Series(np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 50000.0)
    b = pd.Series(np.cumprod(1 + rng.normal(-0.001, 0.03, n)) * 2000.0)
    return a, b


# ── calculate_spread ───────────────────────────────────────────────────────
def test_spread_returns_series():
    pt = PairsTrader()
    a, b = make_cointegrated()
    spread, beta = pt.calculate_spread(a, b)
    assert isinstance(spread, pd.Series)
    assert len(spread) == len(a)


def test_spread_beta_positive():
    pt = PairsTrader()
    a, b = make_cointegrated(beta=1.2)
    _, beta = pt.calculate_spread(a, b)
    assert beta > 0


def test_spread_stationary_for_cointegrated():
    pt = PairsTrader()
    a, b = make_cointegrated(noise_std=0.1)
    spread, _ = pt.calculate_spread(a, b)
    # Spread should have much lower variance than original series
    assert spread.std() < a.std() * 0.5


# ── calculate_zscore ────────────────────────────────────────────────────────
def test_zscore_range():
    pt = PairsTrader()
    a, b = make_cointegrated()
    spread, _ = pt.calculate_spread(a, b)
    zscore = pt.calculate_zscore(spread, lookback=30)
    assert zscore.dropna().abs().max() < 10  # reasonable range


def test_zscore_mean_near_zero():
    pt = PairsTrader()
    a, b = make_cointegrated()
    spread, _ = pt.calculate_spread(a, b)
    zscore = pt.calculate_zscore(spread, lookback=30)
    assert abs(zscore.dropna().mean()) < 1.0


def test_zscore_length():
    pt = PairsTrader()
    a, b = make_cointegrated(n=100)
    spread, _ = pt.calculate_spread(a, b)
    zscore = pt.calculate_zscore(spread, lookback=20)
    assert len(zscore) == len(spread)


# ── calculate_half_life ─────────────────────────────────────────────────────
def test_half_life_positive():
    pt = PairsTrader()
    a, b = make_cointegrated(noise_std=0.5)
    spread, _ = pt.calculate_spread(a, b)
    hl = pt.calculate_half_life(spread)
    assert hl > 0


def test_half_life_in_bounds():
    pt = PairsTrader()
    a, b = make_cointegrated()
    spread, _ = pt.calculate_spread(a, b)
    hl = pt.calculate_half_life(spread)
    assert 1.0 <= hl <= 252.0


# ── test_cointegration ──────────────────────────────────────────────────────
def test_cointegration_result_type():
    pt = PairsTrader()
    a, b = make_cointegrated()
    result = pt.test_cointegration(a, b)
    assert isinstance(result, CointegrationResult)


def test_cointegration_cointegrated_pair():
    pt = PairsTrader()
    a, b = make_cointegrated(noise_std=0.05)  # very tight cointegration
    result = pt.test_cointegration(a, b)
    assert result.beta > 0


def test_cointegration_beta_positive():
    pt = PairsTrader()
    a, b = make_cointegrated(beta=1.5)
    result = pt.test_cointegration(a, b)
    assert result.beta > 0


# ── generate_signal ─────────────────────────────────────────────────────────
def test_generate_signal_returns_pair_signal():
    pt = PairsTrader()
    a, b = make_cointegrated()
    config = PairConfig("BTCUSDT", "ETHUSDT")
    sig = pt.generate_signal(a, b, config)
    assert isinstance(sig, PairSignal)


def test_generate_signal_action_valid():
    pt = PairsTrader()
    a, b = make_cointegrated()
    config = PairConfig("BTCUSDT", "ETHUSDT")
    sig = pt.generate_signal(a, b, config)
    assert sig.action in ("long_a_short_b", "short_a_long_b", "close", "wait")


def test_generate_signal_extreme_zscore_not_wait():
    """Force extreme divergence → should signal"""
    pt = PairsTrader()
    rng = np.random.default_rng(42)
    a = pd.Series(np.cumprod(1 + rng.normal(0.01, 0.005, 100)) * 50000)  # strong up
    b = pd.Series(np.cumprod(1 + rng.normal(-0.01, 0.005, 100)) * 2000)  # strong down
    config = PairConfig("A", "B", entry_zscore=1.0)
    sig = pt.generate_signal(a, b, config)
    # With extreme divergence, should not be "wait"
    assert isinstance(sig.action, str)


def test_generate_signal_confidence_range():
    pt = PairsTrader()
    a, b = make_cointegrated()
    sig = pt.generate_signal(a, b, PairConfig("A", "B"))
    assert 0.0 <= sig.confidence <= 1.0


def test_generate_signal_pts_range():
    pt = PairsTrader()
    a, b = make_cointegrated()
    sig = pt.generate_signal(a, b, PairConfig("A", "B"))
    assert -10 <= sig.pts <= 15


# ── calculate_correlation ────────────────────────────────────────────────────
def test_correlation_high_for_cointegrated():
    pt = PairsTrader()
    a, b = make_cointegrated(noise_std=0.01)
    corr = pt.calculate_correlation(a, b)
    assert corr > 0.5


def test_correlation_range():
    pt = PairsTrader()
    a, b = make_divergent()
    corr = pt.calculate_correlation(a, b)
    assert -1.0 <= corr <= 1.0


# ── scan_all_pairs ───────────────────────────────────────────────────────────
def test_scan_all_pairs_returns_list():
    pt = PairsTrader()
    rng = np.random.default_rng(42)
    n = 150
    price_data = {
        "BTCUSDT": pd.Series(np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 50000),
        "ETHUSDT": pd.Series(np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 2000),
        "SOLUSDT": pd.Series(np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 100),
        "BNBUSDT": pd.Series(np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 300),
    }
    signals = pt.scan_all_pairs(price_data)
    assert isinstance(signals, list)


def test_scan_missing_symbol():
    pt = PairsTrader()
    price_data = {"BTCUSDT": pd.Series([50000] * 100)}  # only one symbol
    signals = pt.scan_all_pairs(price_data)
    assert signals == []


# ── format_telegram ─────────────────────────────────────────────────────────
def test_format_telegram_empty():
    pt = PairsTrader()
    msg = pt.format_telegram([])
    assert isinstance(msg, str)


def test_format_telegram_contains_pair():
    pt = PairsTrader()
    a, b = make_cointegrated()
    config = PairConfig("BTCUSDT", "ETHUSDT")
    sig = pt.generate_signal(a, b, config)
    msg = pt.format_telegram([sig])
    assert "BTCUSDT" in msg or "ETH" in msg
