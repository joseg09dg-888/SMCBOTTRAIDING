"""Tests for smart order execution — TWAP / VWAP / Iceberg / Market.

Run: pytest tests/execution/test_smart_execution.py -v
"""

import pytest
import asyncio
import numpy as np
from execution.smart_execution import (
    SmartExecutor,
    SmartOrderResult,
    OrderSlice,
    ExecutionStrategy,
)

# ── choose_strategy ────────────────────────────────────────────────────────

def test_small_order_market():
    assert SmartExecutor().choose_strategy(50.0) == ExecutionStrategy.MARKET


def test_medium_order_twap():
    assert SmartExecutor().choose_strategy(200.0) == ExecutionStrategy.TWAP


def test_large_order_vwap():
    assert SmartExecutor().choose_strategy(600.0) == ExecutionStrategy.VWAP


# ── calculate_slippage ──────────────────────────────────────────────────────

def test_slippage_buy_overpaid():
    ex = SmartExecutor()
    slip = ex.calculate_slippage(100.0, 100.5, "buy")
    assert slip == pytest.approx(0.005)


def test_slippage_sell_underpaid():
    ex = SmartExecutor()
    slip = ex.calculate_slippage(100.0, 99.5, "sell")
    assert slip == pytest.approx(0.005)


def test_slippage_zero():
    ex = SmartExecutor()
    assert ex.calculate_slippage(100.0, 100.0, "buy") == pytest.approx(0.0)


# ── split_twap ─────────────────────────────────────────────────────────────

def test_split_twap_count():
    ex = SmartExecutor()
    slices = ex.split_twap("BTCUSDT", "buy", 1.0, 50000.0, n_slices=5)
    assert len(slices) == 5


def test_split_twap_total_quantity():
    ex = SmartExecutor()
    slices = ex.split_twap("BTCUSDT", "buy", 1.0, 50000.0, n_slices=5)
    assert sum(s.quantity for s in slices) == pytest.approx(1.0)


def test_split_twap_equal_parts():
    ex = SmartExecutor()
    slices = ex.split_twap("BTCUSDT", "buy", 1.0, 50000.0, n_slices=4)
    for s in slices:
        assert s.quantity == pytest.approx(0.25)


# ── split_vwap ─────────────────────────────────────────────────────────────

def test_split_vwap_total_quantity():
    ex = SmartExecutor()
    vol_profile = [100.0] * 24  # uniform volume
    slices = ex.split_vwap("BTCUSDT", "buy", 1.0, vol_profile, 50000.0)
    assert abs(sum(s.quantity for s in slices) - 1.0) < 1e-6


def test_split_vwap_proportional():
    ex = SmartExecutor()
    # Double volume at hour 12 → should get double quantity
    vol_profile = [1.0] * 24
    vol_profile[12] = 2.0
    slices = ex.split_vwap("BTCUSDT", "buy", 24.0, vol_profile, 50000.0)
    # Hour 12 slice should be roughly double others
    assert slices[12].quantity > slices[0].quantity * 1.5


# ── simulate_execution ─────────────────────────────────────────────────────

def test_simulate_execution_buy_price_close_to_market():
    ex = SmartExecutor()
    sl = OrderSlice("BTCUSDT", "buy", 0.1, 50000.0)
    result = ex.simulate_execution(sl, 50000.0, volatility=0.001, seed=42)
    assert result.executed
    assert abs(result.executed_price - 50000.0) / 50000.0 < 0.01


def test_simulate_execution_marks_executed():
    ex = SmartExecutor()
    sl = OrderSlice("BTCUSDT", "sell", 0.1, 50000.0)
    result = ex.simulate_execution(sl, 50000.0, seed=42)
    assert result.executed is True


# ── execute_market ─────────────────────────────────────────────────────────

def test_execute_market_returns_result():
    ex = SmartExecutor()
    r = ex.execute_market("BTCUSDT", "buy", 0.01, 50000.0)
    assert isinstance(r, SmartOrderResult)
    assert r.success is True


def test_execute_market_commission():
    ex = SmartExecutor(commission=0.001)
    r = ex.execute_market("BTCUSDT", "buy", 1.0, 100.0)
    assert r.total_commission == pytest.approx(0.1 * 2, abs=0.01)  # entry + exit


def test_execute_market_slippage_small():
    ex = SmartExecutor()
    r = ex.execute_market("BTCUSDT", "buy", 0.001, 50000.0)
    assert abs(r.slippage_pct) < 0.005


# ── execute_twap (async) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_twap_returns_result():
    ex = SmartExecutor()
    r = await ex.execute_twap(
        "BTCUSDT", "buy", 0.01, 50000.0, interval_sec=0.01, n_slices=3
    )
    assert isinstance(r, SmartOrderResult)


@pytest.mark.asyncio
async def test_execute_twap_correct_quantity():
    ex = SmartExecutor()
    r = await ex.execute_twap(
        "BTCUSDT", "buy", 1.0, 50000.0, interval_sec=0.001, n_slices=5
    )
    assert abs(r.executed_quantity - 1.0) < 0.001


@pytest.mark.asyncio
async def test_execute_twap_slices_count():
    ex = SmartExecutor()
    r = await ex.execute_twap(
        "BTCUSDT", "buy", 0.1, 50000.0, interval_sec=0.001, n_slices=4
    )
    assert len(r.slices) == 4


# ── market_impact ─────────────────────────────────────────────────────────

def test_market_impact_small_order():
    ex = SmartExecutor()
    impact = ex.estimate_market_impact(100.0, 1_000_000.0, 0.02)
    assert impact < 0.001  # < 0.1%


def test_market_impact_large_order():
    ex = SmartExecutor()
    impact = ex.estimate_market_impact(100_000.0, 1_000_000.0, 0.02)
    assert impact > impact * 0 and impact < 1.0


def test_should_use_twap_large_order():
    ex = SmartExecutor()
    assert ex.should_use_twap(500.0, 0.001) is True


def test_should_use_twap_small_order():
    ex = SmartExecutor()
    assert ex.should_use_twap(50.0, 0.0001) is False


# ── format_report ──────────────────────────────────────────────────────────

def test_format_report_string():
    ex = SmartExecutor()
    r = ex.execute_market("BTCUSDT", "buy", 0.01, 50000.0)
    msg = ex.format_execution_report(r)
    assert isinstance(msg, str) and "BTCUSDT" in msg
