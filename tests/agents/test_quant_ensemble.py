"""
tests/agents/test_quant_ensemble.py
TDD tests for MLEnsemble, FeatureExtractor, EnsemblePrediction
"""

import pytest
import numpy as np
from agents.quant_ensemble import MLEnsemble, FeatureExtractor, EnsemblePrediction


def make_prices(n=200, drift=0.001, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.02, n)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


def make_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = [{"ret_1": rng.normal(0, 0.02),
          "ret_5": rng.normal(0, 0.03),
          "vol_20": rng.uniform(0.01, 0.05),
          "rsi_14": rng.uniform(20, 80),
          "above_ma20": float(rng.integers(0, 2)),
          "momentum_score": rng.normal(0, 0.5)} for _ in range(n)]
    y = [1 if rng.random() > 0.4 else 0 for _ in range(n)]
    return X, y


# ── FeatureExtractor ──────────────────────────────────────────────────────

def test_feature_extractor_returns_dict():
    prices = make_prices()
    features = FeatureExtractor.from_prices(prices)
    assert isinstance(features, dict)
    assert "ret_1" in features
    assert "rsi_14" in features


def test_feature_extractor_short_prices():
    prices = [100.0, 101.0, 102.0]
    features = FeatureExtractor.from_prices(prices)
    assert features["ret_1"] == 0.0  # todos cero por datos insuficientes


def test_rsi_midpoint_for_flat():
    prices = [100.0] * 50
    rsi = FeatureExtractor.calculate_rsi(prices)
    assert rsi == pytest.approx(50.0)


def test_rsi_range():
    prices = make_prices()
    rsi = FeatureExtractor.calculate_rsi(prices)
    assert 0.0 <= rsi <= 100.0


def test_feature_ret_1_correct():
    prices = [100.0, 105.0] + [105.0] * 50
    features = FeatureExtractor.from_prices(prices)
    # ret_1 = 0 porque último price es igual al anterior
    assert features["ret_1"] == pytest.approx(0.0)


def test_feature_above_ma20():
    """Precio sube mucho → debe estar sobre MA20"""
    prices = [100.0 + i for i in range(60)]  # trend up
    features = FeatureExtractor.from_prices(prices)
    assert features["above_ma20"] == 1.0


# ── MLEnsemble sin fit ────────────────────────────────────────────────────

def test_predict_without_fit_returns_prediction():
    ensemble = MLEnsemble()
    features = {"ret_1": 0.01, "momentum_score": 0.3, "above_ma20": 1.0}
    result = ensemble.predict(features)
    assert isinstance(result, EnsemblePrediction)
    assert 0.0 <= result.probability <= 1.0


def test_predict_probability_range():
    ensemble = MLEnsemble()
    features = {"ret_1": 0.05, "momentum_score": 1.0, "above_ma20": 1.0}
    result = ensemble.predict(features)
    assert 0.0 <= result.probability <= 1.0


def test_predict_should_trade_threshold():
    ensemble = MLEnsemble(threshold=0.65)
    # Features muy bullish → probability > 0.65
    features = {"ret_1": 0.03, "momentum_score": 2.0, "above_ma20": 1.0,
                "above_ma50": 1.0, "rsi_14": 60.0, "ret_5": 0.05}
    result = ensemble.predict(features)
    assert result.should_trade == (result.probability >= 0.65)


def test_confidence_levels():
    ensemble = MLEnsemble()
    # Test HIGH confidence
    features = {"momentum_score": 1.5, "above_ma20": 1.0}
    result = ensemble.predict(features)
    assert result.confidence in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")


def test_predict_from_prices_works():
    prices = make_prices(drift=0.002)
    ensemble = MLEnsemble()
    result = ensemble.predict_from_prices(prices)
    assert isinstance(result, EnsemblePrediction)


# ── MLEnsemble con fit ────────────────────────────────────────────────────

def test_fit_and_predict():
    X, y = make_dataset(200)
    ensemble = MLEnsemble()
    ensemble.fit(X, y)
    assert ensemble._is_fitted
    result = ensemble.predict(X[0])
    assert isinstance(result, EnsemblePrediction)


def test_fit_multiple_models():
    X, y = make_dataset(200)
    ensemble = MLEnsemble()
    ensemble.fit(X, y)
    if ensemble._has_sklearn:
        assert len(ensemble._models) >= 2


def test_model_votes_populated_after_fit():
    X, y = make_dataset(200)
    ensemble = MLEnsemble()
    ensemble.fit(X, y)
    result = ensemble.predict(X[0])
    assert len(result.model_votes) >= 1


def test_get_model_accuracies_after_fit():
    X, y = make_dataset(200)
    ensemble = MLEnsemble()
    ensemble.fit(X, y)
    accs = ensemble.get_model_accuracies()
    if ensemble._has_sklearn:
        for acc in accs.values():
            assert 0.0 <= acc <= 1.0


def test_get_model_accuracies_not_fitted():
    ensemble = MLEnsemble()
    assert ensemble.get_model_accuracies() == {}


# ── Edge cases ────────────────────────────────────────────────────────────

def test_predict_empty_features():
    ensemble = MLEnsemble()
    result = ensemble.predict({})
    assert isinstance(result, EnsemblePrediction)
    assert 0.0 <= result.probability <= 1.0


def test_ensemble_deterministic():
    """Mismos features → misma predicción (sin fit)"""
    ensemble = MLEnsemble()
    features = {"momentum_score": 0.5, "above_ma20": 1.0}
    r1 = ensemble.predict(features)
    r2 = ensemble.predict(features)
    assert r1.probability == pytest.approx(r2.probability)
