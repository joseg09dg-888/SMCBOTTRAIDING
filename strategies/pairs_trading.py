"""
Pairs Trading (Statistical Arbitrage) strategy for the SMC Trading Bot.

Uses cointegration-based spread mean-reversion signals.
scipy/statsmodels are optional; numpy is the only hard dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairConfig:
    symbol_a: str          # e.g. "BTCUSDT"
    symbol_b: str          # e.g. "ETHUSDT"
    entry_zscore: float = 2.0    # open when |z| > this
    exit_zscore: float = 0.5     # close when |z| < this
    stop_zscore: float = 3.5     # stop loss when |z| > this
    lookback: int = 60           # periods for spread calculation


@dataclass
class PairSignal:
    pair: PairConfig
    zscore: float
    spread: float
    beta: float              # hedge ratio
    half_life: float         # estimated mean-reversion half-life in periods
    action: str              # "long_a_short_b" | "short_a_long_b" | "close" | "wait"
    confidence: float        # 0.0-1.0
    pts: int                 # -10 to +15 for DecisionFilter


@dataclass
class CointegrationResult:
    is_cointegrated: bool
    p_value: float           # Johansen or ADF test p-value
    beta: float              # hedge ratio (OLS coefficient)
    half_life: float         # Ornstein-Uhlenbeck half-life


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PairsTrader:
    """
    Statistical arbitrage: find cointegrated pairs and trade spread mean reversion.
    Uses scipy for statistics if available, numpy fallback otherwise.
    """

    # Known crypto pairs with historical correlation > 0.75
    KNOWN_PAIRS = [
        PairConfig("BTCUSDT", "ETHUSDT", entry_zscore=2.0, exit_zscore=0.5),
        PairConfig("ETHUSDT", "SOLUSDT", entry_zscore=2.0, exit_zscore=0.5),
        PairConfig("BNBUSDT", "ETHUSDT", entry_zscore=2.0, exit_zscore=0.5),
    ]

    def __init__(self) -> None:
        self._has_scipy = self._check_scipy()
        self._has_statsmodels = self._check_statsmodels()

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    def _check_scipy(self) -> bool:
        try:
            import scipy  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_statsmodels(self) -> bool:
        try:
            import statsmodels  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def calculate_spread(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> tuple[pd.Series, float]:
        """
        Calculate spread = prices_a - beta * prices_b.
        beta = OLS regression coefficient via np.polyfit.
        Returns (spread_series, beta).
        """
        a = prices_a.values.astype(float)
        b = prices_b.values.astype(float)
        beta = float(np.polyfit(b, a, 1)[0])
        spread = pd.Series(a - beta * b, index=prices_a.index)
        return spread, beta

    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: int = 60,
    ) -> pd.Series:
        """
        z-score = (spread - rolling_mean) / rolling_std
        Using lookback window.
        """
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()
        zscore = (spread - rolling_mean) / rolling_std
        return zscore

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Ornstein-Uhlenbeck half-life estimation:
        delta_spread = alpha + beta * spread_lag + error
        half_life = -log(2) / log(1 + beta)

        Returns half-life in periods.
        If calculation fails → return 20.0 (default).
        Clamp to [1.0, 252.0].
        """
        try:
            s = spread.values.astype(float)
            spread_lag = s[:-1]
            delta_spread = np.diff(s)

            # OLS: delta_spread = alpha + beta * spread_lag
            # Use np.polyfit(x, y, 1) → [beta, alpha]
            coeffs = np.polyfit(spread_lag, delta_spread, 1)
            beta = float(coeffs[0])

            # Avoid log(0) or negative argument
            val = 1.0 + beta
            if val <= 0 or val == 1.0:
                return 20.0

            half_life = -math.log(2) / math.log(val)

            # Clamp
            return float(max(1.0, min(252.0, half_life)))

        except Exception:
            return 20.0

    def test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> CointegrationResult:
        """
        Test pair cointegration using ADF test on spread residuals.

        If statsmodels available: use adfuller on residuals.
        Else: use simple correlation check (r > 0.7 → assume cointegrated).

        beta: OLS coefficient from regress(prices_a, prices_b)
        """
        spread, beta = self.calculate_spread(prices_a, prices_b)
        half_life = self.calculate_half_life(spread)

        if self._has_statsmodels:
            try:
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(spread.dropna(), autolag="AIC")
                p_value = float(result[1])
                is_cointegrated = p_value < 0.05
                return CointegrationResult(
                    is_cointegrated=is_cointegrated,
                    p_value=p_value,
                    beta=beta,
                    half_life=half_life,
                )
            except Exception:
                pass  # fall through to correlation proxy

        # Fallback: correlation proxy
        corr = self.calculate_correlation(prices_a, prices_b)
        is_cointegrated = abs(corr) > 0.7
        # Use a synthetic p-value based on correlation strength
        p_value = max(0.001, 1.0 - abs(corr))
        return CointegrationResult(
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            beta=beta,
            half_life=half_life,
        )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        config: PairConfig,
    ) -> PairSignal:
        """
        Full signal generation:
        1. Calculate spread + beta
        2. Calculate z-score
        3. Calculate half-life
        4. Determine action based on z-score thresholds

        Actions:
        zscore > entry_zscore  → "short_a_long_b" (A expensive vs B)
        zscore < -entry_zscore → "long_a_short_b" (A cheap vs B)
        |zscore| < exit_zscore → "close"
        |zscore| > stop_zscore → "close" (stop loss)
        else → "wait"

        confidence = min(abs(zscore) / entry_zscore, 1.0)
        pts: +15 if |z|>2, +10 if |z|>1.5, 0 if wait, -10 if stop
        """
        spread, beta = self.calculate_spread(prices_a, prices_b)
        zscore_series = self.calculate_zscore(spread, lookback=config.lookback)
        half_life = self.calculate_half_life(spread)

        # Use the last valid z-score
        valid_zscores = zscore_series.dropna()
        if len(valid_zscores) == 0:
            current_zscore = 0.0
        else:
            current_zscore = float(valid_zscores.iloc[-1])

        current_spread = float(spread.iloc[-1])
        abs_z = abs(current_zscore)

        # Determine action
        if abs_z > config.stop_zscore:
            action = "close"
            pts = -10
        elif abs_z < config.exit_zscore:
            action = "close"
            pts = 0
        elif current_zscore > config.entry_zscore:
            action = "short_a_long_b"
            pts = 15 if abs_z > 2.0 else 10
        elif current_zscore < -config.entry_zscore:
            action = "long_a_short_b"
            pts = 15 if abs_z > 2.0 else 10
        else:
            action = "wait"
            pts = 0 if abs_z <= 1.5 else 10

        confidence = min(abs_z / config.entry_zscore, 1.0)

        return PairSignal(
            pair=config,
            zscore=current_zscore,
            spread=current_spread,
            beta=beta,
            half_life=half_life,
            action=action,
            confidence=confidence,
            pts=pts,
        )

    # ------------------------------------------------------------------
    # Portfolio scanning
    # ------------------------------------------------------------------

    def scan_all_pairs(
        self,
        price_data: dict[str, pd.Series],
    ) -> list[PairSignal]:
        """
        Scan all KNOWN_PAIRS where both symbols exist in price_data.
        Returns list of active signals (not "wait").
        """
        active_signals: list[PairSignal] = []
        for config in self.KNOWN_PAIRS:
            if config.symbol_a not in price_data or config.symbol_b not in price_data:
                continue
            try:
                sig = self.generate_signal(
                    price_data[config.symbol_a],
                    price_data[config.symbol_b],
                    config,
                )
                if sig.action != "wait":
                    active_signals.append(sig)
            except Exception:
                continue
        return active_signals

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def calculate_correlation(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> float:
        """Pearson correlation between returns."""
        returns_a = prices_a.pct_change().dropna()
        returns_b = prices_b.pct_change().dropna()

        # Align by index length (in case indices differ)
        min_len = min(len(returns_a), len(returns_b))
        if min_len < 2:
            return 0.0

        ra = returns_a.values[-min_len:].astype(float)
        rb = returns_b.values[-min_len:].astype(float)

        corr_matrix = np.corrcoef(ra, rb)
        return float(corr_matrix[0, 1])

    def format_telegram(self, signals: list[PairSignal]) -> str:
        """HTML format for /pairs Telegram command."""
        if not signals:
            return "<b>Pairs Trading</b>\nNo active signals."

        lines = ["<b>Pairs Trading Signals</b>"]
        for sig in signals:
            a = sig.pair.symbol_a
            b = sig.pair.symbol_b
            action_emoji = {
                "long_a_short_b": "BUY",
                "short_a_long_b": "SELL",
                "close": "CLOSE",
                "wait": "WAIT",
            }.get(sig.action, sig.action)

            lines.append(
                f"\n<b>{a}/{b}</b> [{action_emoji}]\n"
                f"  Z-score: {sig.zscore:.2f} | Beta: {sig.beta:.3f}\n"
                f"  Half-life: {sig.half_life:.1f}p | Confidence: {sig.confidence:.0%}\n"
                f"  PTS: {sig.pts:+d}"
            )
        return "\n".join(lines)
