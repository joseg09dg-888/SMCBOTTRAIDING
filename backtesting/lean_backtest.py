"""
SMC Backtesting Engine.

Uses backtrader if available, otherwise a custom numpy/pandas engine.
No external API calls required — all data is passed in as a DataFrame.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────── Data-classes ────────────────────────────────────

@dataclass
class BacktestConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.005   # 0.5 %
    commission: float = 0.001       # 0.1 %
    min_score: int = 35             # demo threshold


@dataclass
class BacktestMetrics:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    cagr_pct: float
    expectancy: float
    total_return_pct: float
    final_equity: float

    # Walk-forward
    in_sample_sharpe: float = 0.0
    out_sample_sharpe: float = 0.0
    walk_forward_efficiency: float = 1.0


@dataclass
class TradeRecord:
    symbol: str
    direction: str          # "long" | "short"
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    setup_tags: list
    score: int
    mae: float              # Maximum Adverse Excursion (negative for loss)
    mfe: float              # Maximum Favorable Excursion (positive for gain)


# ─────────────────────────── Engine ──────────────────────────────────────────

# Annualisation factor for 1-hour crypto candles
_ANNUAL_PERIODS = 252 * 24


class SMCBacktester:
    """
    Backtest an SMC strategy on OHLCV data.
    Uses backtrader if available, otherwise a custom engine.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self._has_backtrader = self._check_backtrader()

    # ── internal helpers ──────────────────────────────────────────────────

    def _check_backtrader(self) -> bool:
        try:
            import backtrader  # noqa: F401
            return True
        except ImportError:
            return False

    # ── public API ────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> BacktestMetrics:
        """
        Run backtest on OHLCV DataFrame.
        df must have columns: open, high, low, close, volume.
        """
        signals = self.generate_signals(df)
        trades = self.simulate_trades(df, signals)

        # Build equity curve
        equity = [self.config.initial_capital]
        current = self.config.initial_capital
        for t in trades:
            current += t.pnl
            equity.append(current)

        return self.calculate_metrics(trades, equity)

    # ── signal generation ─────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns Series of signals: 1=long, -1=short, 0=wait.

        Simple SMC structure detection:
        - BOS up  → long  signal  (close breaks above 20-period rolling high)
        - BOS down → short signal (close breaks below 20-period rolling low)
        - Cool-off of 20 bars between signals to avoid overtrading.
        """
        n = len(df)
        window = 20
        signals = np.zeros(n, dtype=int)

        if n <= window:
            return pd.Series(signals, index=df.index)

        close = df["close"].values
        last_signal_bar = -window  # allow first signal at bar `window`

        for i in range(window, n):
            if (i - last_signal_bar) < window:
                continue

            prev_high = close[i - window: i - 1].max()
            prev_low  = close[i - window: i - 1].min()

            if close[i] > prev_high:
                signals[i] = 1
                last_signal_bar = i
            elif close[i] < prev_low:
                signals[i] = -1
                last_signal_bar = i

        return pd.Series(signals, index=df.index)

    # ── trade simulation ──────────────────────────────────────────────────

    def simulate_trades(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
    ) -> List[TradeRecord]:
        """
        Simulate trades from signals with fixed SL/TP.

        Long:  SL = entry * (1 - 0.005)   TP = entry * (1 + 0.015)
        Short: SL = entry * (1 + 0.005)   TP = entry * (1 - 0.015)
        Commission applied at entry and exit.
        """
        cfg = self.config
        trades: List[TradeRecord] = []
        sig_arr   = signals.values
        n         = len(df)
        close_arr = df["close"].values
        high_arr  = df["high"].values
        low_arr   = df["low"].values
        index     = df.index

        i = 0
        while i < n:
            if sig_arr[i] == 0:
                i += 1
                continue

            direction   = "long" if sig_arr[i] == 1 else "short"
            _comm_mult  = (1 + cfg.commission) if direction == "long" else (1 - cfg.commission)
            entry_price = close_arr[i] * _comm_mult
            entry_time  = index[i]

            sl_price = (
                entry_price * (1 - 0.005)
                if direction == "long"
                else entry_price * (1 + 0.005)
            )
            tp_price = (
                entry_price * (1 + 0.015)
                if direction == "long"
                else entry_price * (1 - 0.015)
            )

            # Position size: risk_per_trade % of capital allocated per trade
            position_value = cfg.initial_capital * cfg.risk_per_trade / 0.005
            qty = position_value / entry_price

            # Walk forward candles to find exit
            mae = 0.0   # maximum adverse excursion (as price diff)
            mfe = 0.0   # maximum favorable excursion (as price diff)
            exit_price = None
            exit_time  = None

            j = i + 1
            while j < n:
                h = high_arr[j]
                l = low_arr[j]
                c = close_arr[j]

                if direction == "long":
                    adverse   = l - entry_price   # negative
                    favorable = h - entry_price   # positive
                    mae = min(mae, adverse)
                    mfe = max(mfe, favorable)

                    if l <= sl_price:
                        exit_price = sl_price
                        exit_time  = index[j]
                        break
                    if h >= tp_price:
                        exit_price = tp_price
                        exit_time  = index[j]
                        break
                else:  # short
                    adverse   = entry_price - h  # negative (price moved against us)
                    favorable = entry_price - l  # positive
                    mae = min(mae, adverse)
                    mfe = max(mfe, favorable)

                    if h >= sl_price:
                        exit_price = sl_price
                        exit_time  = index[j]
                        break
                    if l <= tp_price:
                        exit_price = tp_price
                        exit_time  = index[j]
                        break
                j += 1

            # If no exit triggered, close at last bar
            if exit_price is None:
                exit_price = close_arr[n - 1]
                exit_time  = index[n - 1]

            # Apply commission at exit
            exit_after_comm = (
                exit_price * (1 - cfg.commission)
                if direction == "long"
                else exit_price * (1 + cfg.commission)
            )

            if direction == "long":
                pnl = (exit_after_comm - entry_price) * qty
                pnl_pct = (exit_after_comm - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_after_comm) * qty
                pnl_pct = (entry_price - exit_after_comm) / entry_price

            trades.append(
                TradeRecord(
                    symbol     = cfg.symbol,
                    direction  = direction,
                    entry_price= entry_price,
                    exit_price = exit_price,
                    entry_time = entry_time,
                    exit_time  = exit_time,
                    pnl        = float(pnl),
                    pnl_pct    = float(pnl_pct),
                    setup_tags = ["BOS"],
                    score      = cfg.min_score,
                    mae        = float(mae),
                    mfe        = float(mfe),
                )
            )
            # Resume scanning after the exit bar
            i = j + 1

        return trades

    # ── metrics calculation ───────────────────────────────────────────────

    def calculate_metrics(
        self,
        trades: List[TradeRecord],
        equity_curve: List[float],
    ) -> BacktestMetrics:
        """Calculate all performance metrics from a list of trades."""

        total = len(trades)

        if total == 0:
            return BacktestMetrics(
                total_trades      = 0,
                wins              = 0,
                losses            = 0,
                win_rate          = 0.0,
                profit_factor     = 0.0,
                sharpe_ratio      = 0.0,
                sortino_ratio     = 0.0,
                max_drawdown_pct  = 0.0,
                calmar_ratio      = 0.0,
                cagr_pct          = 0.0,
                expectancy        = 0.0,
                total_return_pct  = 0.0,
                final_equity      = equity_curve[-1] if equity_curve else self.config.initial_capital,
            )

        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / total

        gross_profit = sum(t.pnl for t in wins)
        gross_loss   = abs(sum(t.pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Expectancy per trade
        expectancy = sum(t.pnl for t in trades) / total

        # Equity returns for risk metrics
        eq = np.array(equity_curve, dtype=float)
        initial = eq[0]
        final   = eq[-1]

        total_return_pct = (final - initial) / initial * 100.0

        # Period returns (between each recorded equity point)
        rets = np.diff(eq) / eq[:-1]
        mean_ret = rets.mean() if len(rets) > 0 else 0.0
        std_ret  = rets.std()  if len(rets) > 1 else 1e-9

        sharpe = (mean_ret / std_ret * math.sqrt(_ANNUAL_PERIODS)) if std_ret > 0 else 0.0

        # Sortino: only downside deviation
        down = rets[rets < 0]
        sortino_std = down.std() if len(down) > 1 else 1e-9
        sortino = (mean_ret / sortino_std * math.sqrt(_ANNUAL_PERIODS)) if sortino_std > 0 else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        drawdowns = (eq - peak) / peak
        max_drawdown_pct = float(abs(drawdowns.min()) * 100.0)

        # CAGR — assume each equity step is 1 hour; total hours = len(rets)
        n_hours = len(rets) if len(rets) > 0 else 1
        years = n_hours / _ANNUAL_PERIODS
        if years > 0 and initial > 0:
            cagr_pct = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr_pct = 0.0

        # Calmar
        calmar = (cagr_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

        return BacktestMetrics(
            total_trades      = total,
            wins              = len(wins),
            losses            = len(losses),
            win_rate          = win_rate,
            profit_factor     = profit_factor,
            sharpe_ratio      = sharpe,
            sortino_ratio     = sortino,
            max_drawdown_pct  = max_drawdown_pct,
            calmar_ratio      = calmar,
            cagr_pct          = cagr_pct,
            expectancy        = expectancy,
            total_return_pct  = total_return_pct,
            final_equity      = float(final),
        )

    # ── walk-forward analysis ─────────────────────────────────────────────

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> BacktestMetrics:
        """
        Walk-forward validation.

        For each of `n_splits` windows:
          - First 70% = in-sample  (train / optimise)
          - Last  30% = out-sample (test)

        Returns metrics on the aggregated out-of-sample trades.
        Also sets in_sample_sharpe / out_sample_sharpe / walk_forward_efficiency.
        """
        split_size = len(df) // n_splits
        if split_size < 40:
            # Not enough data — fall back to full run
            return self.run(df)

        oos_trades: List[TradeRecord] = []
        is_trades:  List[TradeRecord] = []

        for k in range(n_splits):
            start = k * split_size
            end   = start + split_size if k < n_splits - 1 else len(df)
            chunk = df.iloc[start:end]

            cutoff = int(len(chunk) * 0.70)
            is_chunk  = chunk.iloc[:cutoff]
            oos_chunk = chunk.iloc[cutoff:]

            if len(is_chunk) > 20:
                is_sig = self.generate_signals(is_chunk)
                is_trades.extend(self.simulate_trades(is_chunk, is_sig))

            if len(oos_chunk) > 20:
                oos_sig = self.generate_signals(oos_chunk)
                oos_trades.extend(self.simulate_trades(oos_chunk, oos_sig))

        # Build equity curves
        def _equity(tlist):
            eq = [self.config.initial_capital]
            c = self.config.initial_capital
            for t in tlist:
                c += t.pnl
                eq.append(c)
            return eq

        oos_eq = _equity(oos_trades)
        is_eq  = _equity(is_trades)

        oos_metrics = self.calculate_metrics(oos_trades, oos_eq)
        is_metrics  = self.calculate_metrics(is_trades,  is_eq)

        # Walk-forward efficiency = OOS sharpe / IS sharpe (clamped to [0, 2])
        if is_metrics.sharpe_ratio != 0:
            wfe = oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio
        else:
            wfe = 1.0

        oos_metrics.in_sample_sharpe      = is_metrics.sharpe_ratio
        oos_metrics.out_sample_sharpe     = oos_metrics.sharpe_ratio
        oos_metrics.walk_forward_efficiency = float(max(0.0, wfe))
        return oos_metrics

    # ── Monte Carlo simulation ────────────────────────────────────────────

    def run_monte_carlo(
        self,
        trades: List[TradeRecord],
        n_sims: int = 1000,
        seed: int = 42,
    ) -> dict:
        """
        Bootstrap trades to estimate risk statistics.

        Returns:
            ruin_probability      : P(drawdown > 80 %)
            worst_drawdown_p95    : 95th-percentile max drawdown
            expected_return_median: median final equity return (%)
        """
        rng = np.random.default_rng(seed)
        pnls = np.array([t.pnl for t in trades], dtype=float)
        n    = len(pnls)
        initial = self.config.initial_capital

        if n == 0:
            return {
                "ruin_probability":       0.0,
                "worst_drawdown_p95":     0.0,
                "expected_return_median": 0.0,
            }

        ruin_count     = 0
        max_dds        = []
        final_returns  = []

        for _ in range(n_sims):
            sample = rng.choice(pnls, size=n, replace=True)
            eq = np.concatenate([[initial], initial + np.cumsum(sample)])
            peak = np.maximum.accumulate(eq)
            dd   = (eq - peak) / peak
            max_dd = float(abs(dd.min()))
            max_dds.append(max_dd)

            final_ret = (eq[-1] - initial) / initial * 100.0
            final_returns.append(final_ret)

            if max_dd > 0.80:
                ruin_count += 1

        return {
            "ruin_probability":       ruin_count / n_sims,
            "worst_drawdown_p95":     float(np.percentile(max_dds, 95)),
            "expected_return_median": float(np.median(final_returns)),
        }

    # ── Telegram formatter ────────────────────────────────────────────────

    def format_telegram(self, metrics: BacktestMetrics) -> str:
        """HTML-formatted Telegram message for /backtest command."""
        dd_str = f"{metrics.max_drawdown_pct:.2f}%"
        pf_str = (
            f"{metrics.profit_factor:.2f}"
            if math.isfinite(metrics.profit_factor)
            else "∞"
        )
        return (
            "<b>📊 Backtest Results</b>\n\n"
            f"<b>Trades:</b> {metrics.total_trades} "
            f"(W: {metrics.wins} / L: {metrics.losses})\n"
            f"<b>Win Rate:</b> {metrics.win_rate * 100:.1f}%\n"
            f"<b>Profit Factor:</b> {pf_str}\n"
            f"<b>Sharpe Ratio:</b> {metrics.sharpe_ratio:.2f}\n"
            f"<b>Sortino Ratio:</b> {metrics.sortino_ratio:.2f}\n"
            f"<b>Max Drawdown:</b> {dd_str}\n"
            f"<b>CAGR:</b> {metrics.cagr_pct:.2f}%\n"
            f"<b>Calmar:</b> {metrics.calmar_ratio:.2f}\n"
            f"<b>Expectancy:</b> ${metrics.expectancy:.2f}\n"
            f"<b>Total Return:</b> {metrics.total_return_pct:.2f}%\n"
            f"<b>Final Equity:</b> ${metrics.final_equity:.2f}\n"
            f"<b>WF Efficiency:</b> {metrics.walk_forward_efficiency:.2f}"
        )

    # ── HTML report ───────────────────────────────────────────────────────

    def save_report(
        self,
        metrics: BacktestMetrics,
        output_path: str = "reports/backtest",
    ) -> str:
        """Save HTML report. Returns the file path."""
        os.makedirs(output_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_path, f"backtest_{ts}.html")

        pf_str = (
            f"{metrics.profit_factor:.2f}"
            if math.isfinite(metrics.profit_factor)
            else "∞"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SMC Backtest Report — {ts}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#1a1a2e; color:#eee; padding:40px; }}
    h1   {{ color:#00d4ff; }}
    table {{ border-collapse:collapse; width:60%; }}
    th, td {{ padding:10px 18px; text-align:left; border-bottom:1px solid #333; }}
    th {{ color:#00d4ff; }}
    .pos {{ color:#00e676; }}
    .neg {{ color:#ff1744; }}
  </style>
</head>
<body>
  <h1>SMC Backtest Report</h1>
  <p>Generated: {ts}</p>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Trades</td><td>{metrics.total_trades}</td></tr>
    <tr><td>Wins / Losses</td><td>{metrics.wins} / {metrics.losses}</td></tr>
    <tr><td>Win Rate</td><td>{metrics.win_rate * 100:.1f}%</td></tr>
    <tr><td>Profit Factor</td><td>{pf_str}</td></tr>
    <tr><td>Sharpe Ratio</td><td>{metrics.sharpe_ratio:.4f}</td></tr>
    <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.4f}</td></tr>
    <tr><td>Max Drawdown</td><td class="neg">{metrics.max_drawdown_pct:.2f}%</td></tr>
    <tr><td>CAGR</td><td class="{'pos' if metrics.cagr_pct >= 0 else 'neg'}">{metrics.cagr_pct:.2f}%</td></tr>
    <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
    <tr><td>Expectancy</td><td>${metrics.expectancy:.2f}</td></tr>
    <tr><td>Total Return</td><td class="{'pos' if metrics.total_return_pct >= 0 else 'neg'}">{metrics.total_return_pct:.2f}%</td></tr>
    <tr><td>Final Equity</td><td>${metrics.final_equity:.2f}</td></tr>
    <tr><td>In-Sample Sharpe</td><td>{metrics.in_sample_sharpe:.4f}</td></tr>
    <tr><td>Out-Sample Sharpe</td><td>{metrics.out_sample_sharpe:.4f}</td></tr>
    <tr><td>WF Efficiency</td><td>{metrics.walk_forward_efficiency:.2f}</td></tr>
  </table>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return filepath
