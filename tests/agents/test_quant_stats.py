import pytest, numpy as np
from agents.quant_stats import QuantStats, MonteCarloResult, WalkForwardResult

def test_expectancy_positive():
    trades = [{"pnl": 20}]*6 + [{"pnl": -10}]*4
    assert abs(QuantStats.calculate_expectancy(trades) - 8.0) < 0.01
def test_expectancy_negative():
    trades = [{"pnl": 10}]*3 + [{"pnl": -20}]*7
    assert QuantStats.calculate_expectancy(trades) < 0
def test_expectancy_empty():
    assert QuantStats.calculate_expectancy([]) == 0.0
def test_kelly_basic():
    k = QuantStats.calculate_kelly_fraction(0.6, 20, 10)
    assert k == pytest.approx(0.25)
def test_kelly_zero_loss():
    assert QuantStats.calculate_kelly_fraction(0.5, 10, 0) == 0.0
def test_kelly_minimum_zero():
    assert QuantStats.calculate_kelly_fraction(0.2, 5, 20) >= 0.0
def test_var_95():
    np.random.seed(42)
    r = list(np.random.uniform(-0.1,0.1,1000))
    assert 0 < QuantStats.calculate_var(r,0.95) < 0.15
def test_cvar_greater_than_var():
    np.random.seed(42)
    r = list(np.random.normal(0,0.02,500))
    assert QuantStats.calculate_cvar(r,0.95) >= QuantStats.calculate_var(r,0.95)
def test_sharpe_positive_returns():
    assert QuantStats.calculate_sharpe([0.01]*100) > 0
def test_sharpe_zero_std():
    assert QuantStats.calculate_sharpe([0.0]*50) == 0.0
def test_sortino_all_positive():
    assert QuantStats.calculate_sortino([0.01]*100) == 0.0
def test_max_drawdown_perfect_trend():
    eq = [1000+i*10 for i in range(100)]
    assert QuantStats.calculate_max_drawdown(eq) == pytest.approx(0.0)
def test_max_drawdown_known_crash():
    eq = [1000,1100,900,800,1000]
    dd = QuantStats.calculate_max_drawdown(eq)
    assert abs(dd - (1100-800)/1100) < 0.01
def test_max_drawdown_empty():
    assert QuantStats.calculate_max_drawdown([]) == 0.0
def test_profit_factor_basic():
    assert QuantStats.calculate_profit_factor([{"pnl":30},{"pnl":-10}]) == pytest.approx(3.0)
def test_profit_factor_no_losses():
    assert QuantStats.calculate_profit_factor([{"pnl":10},{"pnl":20}]) == pytest.approx(30.0)
def test_monte_carlo_reproducible():
    r = [0.01,-0.005,0.02,-0.01]*50
    r1 = QuantStats.run_monte_carlo(r,n_sims=100,seed=42)
    r2 = QuantStats.run_monte_carlo(r,n_sims=100,seed=42)
    assert r1.mean_return == pytest.approx(r2.mean_return)
def test_monte_carlo_returns_correct_type():
    r = [0.005,-0.003]*100
    result = QuantStats.run_monte_carlo(r,n_sims=100,seed=42)
    assert isinstance(result,MonteCarloResult)
    assert len(result.final_equities)==100
def test_monte_carlo_ruin_low():
    result = QuantStats.run_monte_carlo([0.01]*252,n_sims=1000,seed=42)
    assert result.ruin_probability == 0.0
def test_monte_carlo_ruin_high():
    result = QuantStats.run_monte_carlo([-0.05]*252,n_sims=200,seed=42)
    assert result.ruin_probability == pytest.approx(1.0)
def test_walk_forward_type():
    r = [0.01,-0.005]*200
    assert isinstance(QuantStats.run_walk_forward(r,n_splits=4),WalkForwardResult)
def test_walk_forward_degradation():
    r = [0.01,-0.005,0.02,-0.01]*100
    result = QuantStats.run_walk_forward(r,n_splits=5)
    assert result.degradation_ratio >= 0.0
def test_calmar_positive():
    r = [0.005]*252
    eq = [1000*(1+x)**i for i,x in enumerate(r)]
    assert QuantStats.calculate_calmar(r,eq) > 0
def test_ulcer_flat():
    assert QuantStats.calculate_ulcer_index([1000.0]*100) == pytest.approx(0.0)
def test_omega_positive_skew():
    assert QuantStats.calculate_omega_ratio([0.02,0.02,0.02,-0.01]) > 1.0
def test_omega_no_losses():
    result = QuantStats.calculate_omega_ratio([0.01,0.02,0.03])
    assert result == float('inf') or result > 100
def test_ruin_prob_range():
    p = QuantStats.calculate_ruin_probability([0.005,-0.003]*100,n_sims=500,seed=42)
    assert 0.0 <= p <= 1.0
