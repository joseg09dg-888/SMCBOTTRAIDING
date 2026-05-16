"""
Backtesting engine for SMC Trading Bot.
Uses backtrader if available, otherwise a custom numpy/pandas engine.
"""
from backtesting.lean_backtest import (
    SMCBacktester,
    BacktestConfig,
    BacktestMetrics,
    TradeRecord,
)

__all__ = ["SMCBacktester", "BacktestConfig", "BacktestMetrics", "TradeRecord"]
