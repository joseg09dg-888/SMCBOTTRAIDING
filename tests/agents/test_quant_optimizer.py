# tests/agents/test_quant_optimizer.py
import pytest
from agents.quant_optimizer import BayesianOptimizer, OptimizationResult


def simple_objective(params):
    """Maximum at score_threshold=70, min_rr=2.5"""
    return -(params.get("score_threshold", 60) - 70) ** 2 - (params.get("min_rr", 2) - 2.5) ** 2


def test_optimize_returns_result():
    opt = BayesianOptimizer(seed=42)
    result = opt.optimize(simple_objective, n_trials=20)
    assert isinstance(result, OptimizationResult)


def test_optimize_finds_better_than_default():
    opt = BayesianOptimizer(seed=42)
    default_val = simple_objective(BayesianOptimizer.DEFAULT_PARAMS)
    result = opt.optimize(simple_objective, n_trials=30)
    assert result.best_value >= default_val


def test_optimize_n_trials():
    opt = BayesianOptimizer(seed=42)
    result = opt.optimize(simple_objective, n_trials=15)
    assert result.n_trials == 15


def test_optimize_params_within_bounds():
    opt = BayesianOptimizer(seed=42)
    result = opt.optimize(simple_objective, n_trials=20)
    for k, v in result.best_params.items():
        if k in BayesianOptimizer.PARAM_BOUNDS:
            lo, hi = BayesianOptimizer.PARAM_BOUNDS[k]
            assert lo <= v <= hi, f"{k}={v} out of [{lo}, {hi}]"


def test_optimize_specific_params():
    opt = BayesianOptimizer(seed=42)
    result = opt.optimize(
        simple_objective,
        param_names=["score_threshold", "min_rr"],
        n_trials=20,
    )
    assert "score_threshold" in result.best_params
    assert "min_rr" in result.best_params


def test_optimize_convergence_trial_valid():
    opt = BayesianOptimizer(seed=42)
    result = opt.optimize(simple_objective, n_trials=20)
    assert 0 <= result.convergence_trial <= 20


def test_optimize_improvement_pct():
    opt = BayesianOptimizer(seed=42)
    result = opt.optimize(simple_objective, n_trials=20)
    assert isinstance(result.improvement_pct, float)


def test_optimize_sharpe():
    import numpy as np

    def gen_returns(params):
        np.random.seed(42)
        drift = (params.get("score_threshold", 60) - 55) * 0.0001
        return list(np.random.normal(drift, 0.02, 100))

    opt = BayesianOptimizer(seed=42)
    result = opt.optimize_sharpe(gen_returns, n_trials=15)
    assert isinstance(result, OptimizationResult)
    assert result.best_value > -999


def test_clip_params():
    params = {"score_threshold": 200.0, "min_rr": -5.0}
    clipped = BayesianOptimizer.clip_params(params)
    assert clipped["score_threshold"] <= BayesianOptimizer.PARAM_BOUNDS["score_threshold"][1]
    assert clipped["min_rr"] >= BayesianOptimizer.PARAM_BOUNDS["min_rr"][0]


def test_optimize_reproducible_with_seed():
    opt1 = BayesianOptimizer(seed=42)
    opt2 = BayesianOptimizer(seed=42)
    r1 = opt1.optimize(simple_objective, n_trials=10)
    r2 = opt2.optimize(simple_objective, n_trials=10)
    assert abs(r1.best_value - r2.best_value) < 1e-6
