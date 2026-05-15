# tests/agents/test_quant_factors.py
import pytest
import numpy as np
from agents.quant_factors import FactorAnalyzer, FactorResult


def make_correlated(n=100, corr=0.7, seed=42):
    """Factor y returns con correlaciÃ³n conocida."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = corr * x + np.sqrt(1 - corr**2) * noise
    return list(x), list(y)


def test_ic_positively_correlated():
    factors, returns = make_correlated(corr=0.6)
    ic = FactorAnalyzer.calculate_ic(factors, returns)
    assert ic > 0.3


def test_ic_negatively_correlated():
    factors, returns = make_correlated(corr=-0.6)
    ic = FactorAnalyzer.calculate_ic(factors, returns)
    assert ic < -0.3


def test_ic_range():
    factors, returns = make_correlated()
    ic = FactorAnalyzer.calculate_ic(factors, returns)
    assert -1.0 <= ic <= 1.0


def test_ic_empty_returns_zero():
    assert FactorAnalyzer.calculate_ic([], []) == 0.0


def test_ic_constant_returns_zero():
    assert FactorAnalyzer.calculate_ic([1.0] * 10, [1.0] * 10) == 0.0


def test_ir_positive_stable_ic():
    ic_series = [0.08, 0.09, 0.07, 0.10, 0.08]
    ir = FactorAnalyzer.calculate_ir(ic_series)
    assert ir > 0.5  # IR > 0.5 â†’ factor with edge


def test_ir_zero_std():
    assert FactorAnalyzer.calculate_ir([]) == 0.0
    assert FactorAnalyzer.calculate_ir([0.1, 0.1, 0.1]) == 0.0  # std=0


def test_t_stat_significant():
    t = FactorAnalyzer.calculate_t_stat(0.2, 100)
    assert abs(t) > 1.96


def test_t_stat_insignificant():
    t = FactorAnalyzer.calculate_t_stat(0.01, 10)
    assert abs(t) < 1.96


def test_momentum_factor_uptrend():
    prices = [100 + i for i in range(50)]  # sube 1 por periodo
    mom = FactorAnalyzer.calculate_momentum_factor(prices, period=20)
    assert mom > 0


def test_momentum_factor_downtrend():
    prices = [200 - i for i in range(50)]
    mom = FactorAnalyzer.calculate_momentum_factor(prices, period=20)
    assert mom < 0


def test_momentum_insufficient_data():
    assert FactorAnalyzer.calculate_momentum_factor([100, 101], period=20) == 0.0


def test_volatility_factor_high_vol():
    np.random.seed(42)
    returns = list(np.random.normal(0, 0.05, 50))  # alta vol
    vol = FactorAnalyzer.calculate_volatility_factor(returns, window=20)
    assert vol > 0.02


def test_analyze_factor_returns_factor_result():
    factors, returns = make_correlated()
    fa = FactorAnalyzer()
    result = fa.analyze_factor("test", factors, returns)
    assert isinstance(result, FactorResult)
    assert result.factor_name == "test"
    assert result.significant is True or result.significant is False


def test_analyze_all_factors_returns_list():
    np.random.seed(42)
    prices = list(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 100)
    returns = list(np.diff(prices) / prices[:-1])
    fa = FactorAnalyzer()
    results = fa.analyze_all_factors(prices, returns)
    assert isinstance(results, list)
    assert len(results) >= 3


def test_get_best_factor():
    r1 = FactorResult("mom", 0.05, 0.8, 2.0, True, "long")
    r2 = FactorResult("rev", -0.03, 0.3, 1.0, False, "short")
    best = FactorAnalyzer.get_best_factor([r1, r2])
    assert best.factor_name == "mom"


def test_get_best_factor_empty():
    assert FactorAnalyzer.get_best_factor([]) is None
