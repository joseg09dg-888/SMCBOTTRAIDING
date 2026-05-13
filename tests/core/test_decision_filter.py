import pytest
import pandas as pd
from datetime import datetime, timezone
from core.decision_filter import DecisionFilter, DecisionResult, TradeGrade
from core.config import Config
from core.risk_manager import RiskManager


@pytest.fixture
def df_bullish():
    data = {
        "open":   [100, 102, 104, 101, 110, 108, 115, 112, 118, 115],
        "high":   [103, 106, 107, 102, 115, 112, 120, 116, 122, 118],
        "low":    [99,  101, 102, 98,  108, 106, 112, 109, 115, 112],
        "close":  [102, 105, 103, 110, 112, 109, 118, 114, 120, 116],
        "volume": [1000, 1500, 800, 3000, 2000, 900, 3500, 1200, 4000, 1800],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cfg():
    c = Config()
    c.max_risk_per_trade = 0.005
    c.max_daily_loss = 0.05
    c.max_monthly_loss = 0.15
    c.max_open_positions = 3
    return c


@pytest.fixture
def rm(cfg):
    return RiskManager(cfg, capital=10000.0)


@pytest.fixture
def df():
    return pytest.lazy_fixture("df_bullish") if False else None


# --- Score range ---

def test_score_is_between_0_and_100(df_bullish, cfg, rm):
    df = DecisionFilter(cfg, rm)
    result = df.evaluate(
        df=df_bullish,
        symbol="BTCUSDT",
        timeframe="1h",
        entry=120.0,
        stop_loss=115.0,
        take_profit=135.0,
        bias="bullish",
    )
    assert 0 <= result.score <= 100


def test_result_has_grade(df_bullish, cfg, rm):
    df = DecisionFilter(cfg, rm)
    result = df.evaluate(
        df=df_bullish, symbol="BTCUSDT", timeframe="1h",
        entry=120.0, stop_loss=115.0, take_profit=135.0, bias="bullish",
    )
    assert isinstance(result.grade, TradeGrade)
    assert result.grade in list(TradeGrade)


# --- Grade thresholds ---

def test_grade_no_trade_below_60(cfg, rm):
    df = DecisionFilter(cfg, rm)
    result = df._score_to_result(score=45, symbol="X", breakdown={})
    assert result.grade == TradeGrade.NO_TRADE
    assert result.risk_multiplier == 0.0
    assert result.reason is not None


def test_grade_reduced_between_60_and_74(cfg, rm):
    df = DecisionFilter(cfg, rm)
    result = df._score_to_result(score=65, symbol="X", breakdown={})
    assert result.grade == TradeGrade.REDUCED
    assert result.risk_multiplier == 0.25


def test_grade_full_between_75_and_89(cfg, rm):
    df = DecisionFilter(cfg, rm)
    result = df._score_to_result(score=80, symbol="X", breakdown={})
    assert result.grade == TradeGrade.FULL
    assert result.risk_multiplier == 1.0


def test_grade_premium_at_90_plus(cfg, rm):
    df = DecisionFilter(cfg, rm)
    result = df._score_to_result(score=92, symbol="X", breakdown={})
    assert result.grade == TradeGrade.PREMIUM
    assert result.risk_multiplier == 1.0
    assert result.premium_alert is True


# --- Risk rules still enforced ---

def test_no_trade_when_daily_limit_hit(df_bullish, cfg, rm):
    rm.daily_pnl = -600.0  # over 5% of 10k
    df = DecisionFilter(cfg, rm)
    result = df.evaluate(
        df=df_bullish, symbol="BTCUSDT", timeframe="1h",
        entry=120.0, stop_loss=115.0, take_profit=135.0, bias="bullish",
    )
    assert result.grade == TradeGrade.NO_TRADE
    assert "diaria" in result.reason.lower()


def test_no_trade_when_max_positions_hit(df_bullish, cfg, rm):
    rm.open_positions = 3
    df = DecisionFilter(cfg, rm)
    result = df.evaluate(
        df=df_bullish, symbol="BTCUSDT", timeframe="1h",
        entry=120.0, stop_loss=115.0, take_profit=135.0, bias="bullish",
    )
    assert result.grade == TradeGrade.NO_TRADE


# --- Score breakdown exposed ---

def test_breakdown_has_components(df_bullish, cfg, rm):
    df = DecisionFilter(cfg, rm)
    result = df.evaluate(
        df=df_bullish, symbol="BTCUSDT", timeframe="1h",
        entry=120.0, stop_loss=115.0, take_profit=135.0, bias="bullish",
    )
    assert "smc" in result.breakdown
    assert "ml" in result.breakdown
    assert "sentiment" in result.breakdown
    assert "risk" in result.breakdown
    total = sum(result.breakdown.values())
    assert abs(total - result.score) < 1  # scores sum to total


# --- Neutral bias scores low ---

def test_neutral_bias_scores_lower(df_bullish, cfg, rm):
    df = DecisionFilter(cfg, rm)
    r_neutral = df.evaluate(
        df=df_bullish, symbol="BTCUSDT", timeframe="1h",
        entry=120.0, stop_loss=115.0, take_profit=135.0, bias="neutral",
    )
    r_bullish = df.evaluate(
        df=df_bullish, symbol="BTCUSDT", timeframe="1h",
        entry=120.0, stop_loss=115.0, take_profit=135.0, bias="bullish",
    )
    assert r_neutral.score < r_bullish.score
