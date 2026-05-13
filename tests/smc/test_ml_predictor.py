import pytest
import pandas as pd
from smc.ml_predictor import MLPredictor, PredictionResult


@pytest.fixture
def ohlcv_trend():
    data = {
        "open":   [100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                   120, 122, 124, 121, 125, 123, 128, 126, 130, 128],
        "high":   [103, 105, 107, 109, 111, 113, 115, 117, 119, 121,
                   123, 125, 127, 123, 128, 126, 131, 129, 133, 131],
        "low":    [99,  101, 103, 105, 107, 109, 111, 113, 115, 117,
                   119, 121, 122, 118, 123, 121, 126, 124, 128, 126],
        "close":  [102, 104, 106, 108, 110, 112, 114, 116, 118, 120,
                   122, 124, 122, 123, 125, 127, 129, 128, 131, 129],
        "volume": [1000]*20,
    }
    return pd.DataFrame(data)


def test_prediction_has_required_fields(ohlcv_trend):
    pred = MLPredictor()
    result = pred.predict(ohlcv_trend, bias="bullish")
    assert isinstance(result, PredictionResult)
    assert hasattr(result, "direction")
    assert hasattr(result, "confidence")
    assert hasattr(result, "score")
    assert hasattr(result, "features")


def test_confidence_between_0_and_1(ohlcv_trend):
    pred = MLPredictor()
    result = pred.predict(ohlcv_trend, bias="bullish")
    assert 0.0 <= result.confidence <= 1.0


def test_score_between_0_and_25(ohlcv_trend):
    pred = MLPredictor()
    result = pred.predict(ohlcv_trend, bias="bullish")
    assert 0 <= result.score <= 25


def test_bullish_bias_scores_higher_on_uptrend(ohlcv_trend):
    pred = MLPredictor()
    r_bull = pred.predict(ohlcv_trend, bias="bullish")
    r_bear = pred.predict(ohlcv_trend, bias="bearish")
    assert r_bull.score >= r_bear.score


def test_direction_is_valid_string(ohlcv_trend):
    pred = MLPredictor()
    result = pred.predict(ohlcv_trend, bias="bullish")
    assert result.direction in ("bullish", "bearish", "neutral")


def test_features_dict_has_keys(ohlcv_trend):
    pred = MLPredictor()
    result = pred.predict(ohlcv_trend, bias="bullish")
    assert "momentum" in result.features
    assert "trend_consistency" in result.features
    assert "volume_confirmation" in result.features
