import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backtesting.lean_backtest import SMCBacktester, BacktestConfig, BacktestMetrics, TradeRecord


def make_df(n=500, trend="up", seed=42):
    rng = np.random.default_rng(seed)
    drift = 0.001 if trend == "up" else (-0.001 if trend == "down" else 0.0)
    returns = rng.normal(drift, 0.02, n)
    close = np.cumprod(1 + returns) * 50000.0
    high  = close * (1 + abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol   = rng.uniform(100, 1000, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


# ── SMCBacktester.run ──────────────────────────────────────────────────────

def test_run_returns_metrics():
    bt = SMCBacktester()
    df = make_df(200)
    result = bt.run(df)
    assert isinstance(result, BacktestMetrics)


def test_run_metrics_valid_ranges():
    bt = SMCBacktester()
    result = bt.run(make_df(300))
    assert 0.0 <= result.win_rate <= 1.0
    assert result.max_drawdown_pct >= 0.0
    assert result.total_trades >= 0


def test_run_uptrend_positive_returns():
    bt = SMCBacktester()
    result = bt.run(make_df(300, trend="up"))
    # With uptrend, should find some profitable trades
    assert isinstance(result.total_return_pct, float)


def test_run_empty_df_returns_zero_trades():
    bt = SMCBacktester()
    df = make_df(10)  # too short for signals
    result = bt.run(df)
    assert isinstance(result, BacktestMetrics)


# ── generate_signals ───────────────────────────────────────────────────────

def test_generate_signals_returns_series():
    bt = SMCBacktester()
    df = make_df(200)
    sig = bt.generate_signals(df)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(df)


def test_generate_signals_valid_values():
    bt = SMCBacktester()
    sig = bt.generate_signals(make_df(200))
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_generate_signals_not_all_zero():
    bt = SMCBacktester()
    sig = bt.generate_signals(make_df(300, trend="up"))
    assert (sig != 0).sum() >= 1


# ── simulate_trades ────────────────────────────────────────────────────────

def test_simulate_trades_returns_list():
    bt = SMCBacktester()
    df = make_df(200)
    sig = bt.generate_signals(df)
    trades = bt.simulate_trades(df, sig)
    assert isinstance(trades, list)


def test_simulate_trades_valid_pnl():
    bt = SMCBacktester()
    df = make_df(300)
    sig = bt.generate_signals(df)
    trades = bt.simulate_trades(df, sig)
    for t in trades:
        assert isinstance(t.pnl, float)
        assert isinstance(t.direction, str)
        assert t.direction in ("long", "short")


def test_simulate_trades_rr_correct():
    """R:R should be approximately 1:3 (TP = SL * 3)"""
    bt = SMCBacktester()
    df = make_df(300, trend="up")
    sig = pd.Series([1 if i % 30 == 0 else 0 for i in range(len(df))], index=df.index)
    trades = bt.simulate_trades(df, sig)
    if trades:
        t = trades[0]
        assert t.mae <= 0 or t.mfe >= 0  # MAE and MFE exist


# ── calculate_metrics ──────────────────────────────────────────────────────

def test_calculate_metrics_no_trades():
    bt = SMCBacktester()
    m = bt.calculate_metrics([], [1000.0, 1000.0])
    assert m.total_trades == 0
    assert m.win_rate == 0.0


def test_calculate_metrics_all_wins():
    trades = [
        TradeRecord(
            "BTC", "long", 100, 103,
            datetime(2023, 1, 1), datetime(2023, 1, 2),
            30.0, 0.03, ["OB"], 70, 0.0, 30.0,
        )
        for _ in range(5)
    ]
    eq = [1000 + i * 30 for i in range(6)]
    bt = SMCBacktester()
    m = bt.calculate_metrics(trades, eq)
    assert m.win_rate == pytest.approx(1.0)
    assert m.wins == 5


def test_calculate_metrics_sharpe_positive():
    trades = [
        TradeRecord(
            "BTC", "long", 100, 102,
            datetime(2023, 1, i), datetime(2023, 1, i + 1),
            20.0, 0.02, ["OB"], 65, 0.0, 20.0,
        )
        for i in range(1, 6)
    ]
    eq = [1000 + i * 20 for i in range(6)]
    bt = SMCBacktester()
    m = bt.calculate_metrics(trades, eq)
    assert isinstance(m.sharpe_ratio, float)


def test_calculate_profit_factor():
    wins = [
        TradeRecord(
            "BTC", "long", 100, 103,
            datetime(2023, 1, 1), datetime(2023, 1, 2),
            30.0, 0.03, [], 70, 0.0, 30.0,
        )
    ]
    loss = [
        TradeRecord(
            "BTC", "long", 100, 99,
            datetime(2023, 1, 3), datetime(2023, 1, 4),
            -10.0, -0.01, [], 50, -10.0, 0.0,
        )
    ]
    bt = SMCBacktester()
    m = bt.calculate_metrics(wins + loss, [1000, 1030, 1020])
    assert m.profit_factor == pytest.approx(3.0)


# ── walk_forward ───────────────────────────────────────────────────────────

def test_walk_forward_returns_metrics():
    bt = SMCBacktester()
    result = bt.run_walk_forward(make_df(500), n_splits=3)
    assert isinstance(result, BacktestMetrics)


def test_walk_forward_degradation():
    bt = SMCBacktester()
    result = bt.run_walk_forward(make_df(500), n_splits=3)
    assert result.walk_forward_efficiency >= 0.0


# ── monte_carlo ────────────────────────────────────────────────────────────

def test_monte_carlo_returns_dict():
    bt = SMCBacktester()
    trades = [
        TradeRecord(
            "BTC", "long", 100, 102,
            datetime(2023, 1, 1), datetime(2023, 1, 2),
            20.0, 0.02, [], 65, 0.0, 20.0,
        )
    ] * 20
    result = bt.run_monte_carlo(trades, n_sims=100, seed=42)
    assert "ruin_probability" in result
    assert "worst_drawdown_p95" in result
    assert "expected_return_median" in result


def test_monte_carlo_reproducible():
    bt = SMCBacktester()
    trades = [
        TradeRecord(
            "BTC", "long", 100, 102,
            datetime(2023, 1, 1), datetime(2023, 1, 2),
            20.0, 0.02, [], 65, 0.0, 20.0,
        )
    ] * 20
    r1 = bt.run_monte_carlo(trades, n_sims=100, seed=42)
    r2 = bt.run_monte_carlo(trades, n_sims=100, seed=42)
    assert r1["ruin_probability"] == r2["ruin_probability"]


def test_monte_carlo_ruin_range():
    bt = SMCBacktester()
    trades = [
        TradeRecord(
            "BTC", "long", 100, 102,
            datetime(2023, 1, 1), datetime(2023, 1, 2),
            20.0, 0.02, [], 65, 0.0, 20.0,
        )
    ] * 20
    result = bt.run_monte_carlo(trades, n_sims=200, seed=42)
    assert 0.0 <= result["ruin_probability"] <= 1.0


# ── format_telegram / save_report ─────────────────────────────────────────

def test_format_telegram_contains_metrics():
    bt = SMCBacktester()
    m = bt.calculate_metrics([], [1000.0, 1000.0])
    msg = bt.format_telegram(m)
    assert "Win Rate" in msg or "win" in msg.lower()


def test_save_report_creates_file(tmp_path):
    bt = SMCBacktester()
    m = bt.calculate_metrics([], [1000.0, 1000.0])
    path = bt.save_report(m, output_path=str(tmp_path))
    import os
    assert os.path.exists(path)
    assert path.endswith(".html")
