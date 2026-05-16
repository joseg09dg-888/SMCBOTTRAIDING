"""Smart order execution — TWAP, VWAP, Iceberg, Market.

No real API calls; all execution is simulated with realistic slippage.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, List, Optional

import numpy as np


# ── Enums & dataclasses ────────────────────────────────────────────────────


class ExecutionStrategy(Enum):
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


@dataclass
class OrderSlice:
    """One part of a split order."""

    symbol: str
    side: str           # "buy" | "sell"
    quantity: float
    target_price: float
    executed_price: float = 0.0
    executed: bool = False
    timestamp: str = ""


@dataclass
class SmartOrderResult:
    symbol: str
    side: str
    total_quantity: float
    executed_quantity: float
    avg_price: float
    slippage_pct: float
    strategy_used: ExecutionStrategy
    slices: list
    total_commission: float
    success: bool
    error: str = ""


# ── SmartExecutor ──────────────────────────────────────────────────────────


class SmartExecutor:
    """
    Smart order execution: TWAP, VWAP, Iceberg.
    For demo/small orders → standard market order.
    No real API calls — simulates execution.
    """

    TWAP_THRESHOLD_USD: float = 100.0
    DEFAULT_TWAP_SLICES: int = 5
    DEFAULT_TWAP_INTERVAL_SEC: float = 30.0
    ICEBERG_VISIBLE_PCT: float = 0.2

    def __init__(self, capital: float = 1000.0, commission: float = 0.001) -> None:
        self.capital = capital
        self.commission = commission

    # ── strategy selection ─────────────────────────────────────────────────

    def choose_strategy(self, order_value_usd: float) -> ExecutionStrategy:
        """
        < $100  → MARKET
        $100-$500 → TWAP
        > $500  → VWAP
        """
        if order_value_usd < self.TWAP_THRESHOLD_USD:
            return ExecutionStrategy.MARKET
        if order_value_usd <= 500.0:
            return ExecutionStrategy.TWAP
        return ExecutionStrategy.VWAP

    # ── slippage ───────────────────────────────────────────────────────────

    def calculate_slippage(
        self,
        target_price: float,
        avg_executed_price: float,
        side: str,
    ) -> float:
        """
        buy : (avg - target) / target   (positive → paid more)
        sell: (target - avg) / target   (positive → received less)
        """
        if side == "buy":
            return (avg_executed_price - target_price) / target_price
        return (target_price - avg_executed_price) / target_price

    # ── TWAP splitting ─────────────────────────────────────────────────────

    def split_twap(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        current_price: float,
        n_slices: int = None,
    ) -> List[OrderSlice]:
        """Split order into n_slices equal parts."""
        n = n_slices if n_slices is not None else self.DEFAULT_TWAP_SLICES
        slice_qty = total_quantity / n
        return [
            OrderSlice(
                symbol=symbol,
                side=side,
                quantity=slice_qty,
                target_price=current_price,
            )
            for _ in range(n)
        ]

    # ── VWAP splitting ─────────────────────────────────────────────────────

    def split_vwap(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        volume_profile: List[float],
        current_price: float,
    ) -> List[OrderSlice]:
        """
        Split proportional to volume profile.
        More quantity when volume is high, less when volume is low.
        sum(quantities) == total_quantity exactly.
        """
        vp = np.array(volume_profile, dtype=float)
        total_vol = vp.sum()
        weights = vp / total_vol
        quantities = weights * total_quantity

        # Fix floating-point residual so sum == total_quantity exactly
        residual = total_quantity - quantities.sum()
        quantities[-1] += residual

        return [
            OrderSlice(
                symbol=symbol,
                side=side,
                quantity=float(q),
                target_price=current_price,
            )
            for q in quantities
        ]

    # ── execution simulation ───────────────────────────────────────────────

    def simulate_execution(
        self,
        slice_: OrderSlice,
        market_price: float,
        volatility: float = 0.001,
        seed: int = None,
    ) -> OrderSlice:
        """
        Simulate order fill with realistic slippage:
          executed_price = market_price * (1 + N(0, volatility))
        Buy:  executed_price >= market_price (pay more)
        Sell: executed_price <= market_price (receive less)
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, volatility)
        # For buy: abs noise pushes price up; for sell: pushes price down
        if slice_.side == "buy":
            executed_price = market_price * (1.0 + abs(noise))
        else:
            executed_price = market_price * (1.0 - abs(noise))

        return OrderSlice(
            symbol=slice_.symbol,
            side=slice_.side,
            quantity=slice_.quantity,
            target_price=slice_.target_price,
            executed_price=executed_price,
            executed=True,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    # ── market order ───────────────────────────────────────────────────────

    def execute_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
    ) -> SmartOrderResult:
        """Immediate market order simulation."""
        sl = OrderSlice(symbol, side, quantity, current_price)
        filled = self.simulate_execution(sl, current_price)

        avg_price = filled.executed_price
        slippage = self.calculate_slippage(current_price, avg_price, side)

        # Commission on both entry and (simulated) exit notional
        notional = quantity * avg_price
        commission = notional * self.commission * 2  # round-trip

        return SmartOrderResult(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            executed_quantity=quantity,
            avg_price=avg_price,
            slippage_pct=slippage,
            strategy_used=ExecutionStrategy.MARKET,
            slices=[filled],
            total_commission=commission,
            success=True,
        )

    # ── async TWAP execution ───────────────────────────────────────────────

    async def execute_twap(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        current_price: float,
        interval_sec: float = 30.0,
        n_slices: int = 5,
        price_feed: Optional[Callable] = None,
    ) -> SmartOrderResult:
        """
        Execute TWAP order asynchronously.
        price_feed: async callable returning current price.
        If None → current_price with small random walk.
        """
        slices = self.split_twap(symbol, side, total_quantity, current_price, n_slices)
        filled_slices: List[OrderSlice] = []
        market_price = current_price
        rng = np.random.default_rng(None)

        for raw_slice in slices:
            if price_feed is not None:
                market_price = await price_feed()
            else:
                # small random walk ±0.05%
                market_price *= 1.0 + rng.uniform(-0.0005, 0.0005)

            filled = self.simulate_execution(raw_slice, market_price)
            filled_slices.append(filled)

            await asyncio.sleep(interval_sec)

        executed_qty = sum(s.quantity for s in filled_slices)
        avg_price = (
            sum(s.executed_price * s.quantity for s in filled_slices) / executed_qty
            if executed_qty > 0
            else 0.0
        )
        slippage = self.calculate_slippage(current_price, avg_price, side)
        notional = executed_qty * avg_price
        commission = notional * self.commission * 2

        return SmartOrderResult(
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            executed_quantity=executed_qty,
            avg_price=avg_price,
            slippage_pct=slippage,
            strategy_used=ExecutionStrategy.TWAP,
            slices=filled_slices,
            total_commission=commission,
            success=True,
        )

    # ── market impact (Kyle's lambda) ─────────────────────────────────────

    def estimate_market_impact(
        self,
        order_value_usd: float,
        avg_daily_volume_usd: float = 1_000_000.0,
        volatility: float = 0.02,
    ) -> float:
        """
        Kyle's lambda: impact_pct = vol * sqrt(order / adv)
        Returns percentage impact (e.g. 0.001 = 0.1%).
        """
        return volatility * math.sqrt(order_value_usd / avg_daily_volume_usd)

    def should_use_twap(
        self,
        order_value_usd: float,
        impact_pct: float,
    ) -> bool:
        """True if impact > 0.05% OR order > TWAP_THRESHOLD_USD."""
        return impact_pct > 0.0005 or order_value_usd > self.TWAP_THRESHOLD_USD

    # ── Telegram HTML report ───────────────────────────────────────────────

    def format_execution_report(self, result: SmartOrderResult) -> str:
        """HTML-formatted execution report for Telegram."""
        status = "SUCCESS" if result.success else "FAILED"
        slip_bps = result.slippage_pct * 10_000

        lines = [
            f"<b>Execution Report — {result.symbol}</b>",
            f"Status   : <b>{status}</b>",
            f"Strategy : {result.strategy_used.value.upper()}",
            f"Side     : {result.side.upper()}",
            f"Qty      : {result.total_quantity:.6f}",
            f"Executed : {result.executed_quantity:.6f}",
            f"Avg Price: {result.avg_price:.4f}",
            f"Slippage : {slip_bps:.2f} bps",
            f"Commission: {result.total_commission:.6f}",
            f"Slices   : {len(result.slices)}",
        ]
        if result.error:
            lines.append(f"Error    : {result.error}")

        return "\n".join(lines)
