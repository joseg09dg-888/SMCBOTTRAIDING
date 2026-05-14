"""ElliottFibonacciAgent — Elliott Wave detection + Fibonacci levels."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import numpy as np


FIB_RATIOS = {
    "0.236": 0.236, "0.382": 0.382, "0.500": 0.500,
    "0.618": 0.618, "0.786": 0.786, "1.000": 1.000,
    "1.618": 1.618,
}


@dataclass
class ElliottResult:
    wave_count: int             # 1-5
    wave_type: str              # "impulse" or "corrective"
    current_wave: str           # "wave_1" … "wave_5" / "wave_A" … "wave_C"
    fib_levels: Dict[str, float]
    time_projection_days: int
    score_bonus: int            # 0-10
    confidence: float           # 0.0-1.0
    summary: str


class ElliottFibonacciAgent:
    """
    Estimates Elliott Wave position from OHLCV data and computes Fibonacci
    retracement / extension levels.
    """

    def calculate_fib_levels(self, high: float, low: float) -> Dict[str, float]:
        """Returns price levels for each Fibonacci ratio between low and high."""
        rng = high - low
        return {ratio: round(low + rng * val, 6)
                for ratio, val in FIB_RATIOS.items()}

    def _find_swings(self, df: pd.DataFrame, window: int = 5):
        """Returns list of (index, price, direction) swing points."""
        highs = df["high"].rolling(window, center=True).max()
        lows  = df["low"].rolling(window, center=True).min()
        swings = []
        for i in range(window, len(df) - window):
            if df["high"].iloc[i] == highs.iloc[i]:
                swings.append((i, df["high"].iloc[i], "high"))
            elif df["low"].iloc[i] == lows.iloc[i]:
                swings.append((i, df["low"].iloc[i], "low"))
        return swings

    def analyze(self, df: pd.DataFrame, bias: str = "bullish") -> ElliottResult:
        if len(df) < 20:
            return ElliottResult(
                wave_count=1, wave_type="impulse", current_wave="wave_1",
                fib_levels=self.calculate_fib_levels(
                    df["high"].max(), df["low"].min()
                ),
                time_projection_days=14,
                score_bonus=0, confidence=0.3,
                summary="Datos insuficientes para análisis Elliott",
            )

        high = float(df["high"].max())
        low  = float(df["low"].min())
        last = float(df["close"].iloc[-1])
        fib  = self.calculate_fib_levels(high, low)

        # Estimate wave position by price location in range
        position = (last - low) / (high - low) if high != low else 0.5

        swings = self._find_swings(df)
        n_swings = len(swings)

        if n_swings >= 5:
            wave_count   = 3
            current_wave = "wave_3"
            bonus        = 10
            confidence   = 0.75
            w_type       = "impulse"
        elif n_swings >= 3:
            wave_count   = 2
            current_wave = "wave_2"
            bonus        = 5
            confidence   = 0.55
            w_type       = "impulse"
        elif position > 0.80:
            wave_count   = 5
            current_wave = "wave_5"
            bonus        = 5
            confidence   = 0.60
            w_type       = "impulse"
        elif position < 0.20:
            wave_count   = 4
            current_wave = "wave_4" if bias == "bullish" else "wave_A"
            bonus        = 3
            confidence   = 0.45
            w_type       = "corrective"
        else:
            wave_count   = 1
            current_wave = "wave_1"
            bonus        = 3
            confidence   = 0.40
            w_type       = "impulse"

        # Time projection: avg bars between swings
        if len(swings) >= 2:
            avg_bars = int(np.mean([
                swings[i+1][0] - swings[i][0]
                for i in range(len(swings)-1)
            ]))
            time_proj = max(avg_bars, 5)
        else:
            time_proj = 14

        return ElliottResult(
            wave_count=wave_count,
            wave_type=w_type,
            current_wave=current_wave,
            fib_levels=fib,
            time_projection_days=time_proj,
            score_bonus=bonus,
            confidence=confidence,
            summary=(
                f"{current_wave.replace('_',' ').title()} detectada "
                f"({w_type}) | Confianza {confidence:.0%} | "
                f"Próximo swing en ~{time_proj} días"
            ),
        )

    def score_adjustment(self, df: pd.DataFrame, bias: str) -> int:
        return self.analyze(df, bias).score_bonus

    def format_telegram(self, symbol: str, df: pd.DataFrame) -> str:
        r = self.analyze(df)
        key_fibs = {k: v for k, v in r.fib_levels.items()
                    if k in ("0.382", "0.618", "1.618")}
        fib_str = " | ".join(f"{k}={v:,.2f}" for k, v in key_fibs.items())
        return (
            f"🌊 *Elliott — {symbol}*\n"
            f"Onda actual: {r.current_wave} ({r.wave_type})\n"
            f"Confianza: {r.confidence:.0%} | Bonus: +{r.score_bonus} pts\n"
            f"Fibonacci: {fib_str}\n"
            f"Proyección temporal: ~{r.time_projection_days} días\n"
            f"{r.summary}"
        )
