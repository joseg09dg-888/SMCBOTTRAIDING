"""
Bill Williams indicators: Alligator (3 smoothed, offset moving averages) and
Awesome Oscillator (momentum vs a longer baseline). Fractals are
intentionally NOT implemented here -- they detect 5-bar swing high/low
reversal points, which is the same job smc/structure.py's swing detection
already does; adding them would just duplicate that logic under a
different name.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class BillWilliamsSignal:
    jaw: float
    teeth: float
    lips: float
    ao: float
    pts_adjustment: int
    reason: str


class BillWilliamsIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _smma(self, series: pd.Series, period: int) -> pd.Series:
        """Smoothed moving average (Williams' method)."""
        return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    def alligator(self) -> tuple[float, float, float]:
        """Returns (jaw, teeth, lips) -- 13/8/5 SMMA on median price, shifted
        8/5/3 bars forward per the original definition. Falls back to the
        last median price for all three if not enough data."""
        median = (self.df["high"] + self.df["low"]) / 2
        if len(median) < 13 + 8:
            last = float(median.iloc[-1])
            return last, last, last
        jaw   = self._smma(median, 13).shift(8)
        teeth = self._smma(median, 8).shift(5)
        lips  = self._smma(median, 5).shift(3)
        return float(jaw.iloc[-1]), float(teeth.iloc[-1]), float(lips.iloc[-1])

    def awesome_oscillator(self) -> float:
        """AO = SMA(median,5) - SMA(median,34). Returns 0.0 (neutral) if not
        enough data."""
        median = (self.df["high"] + self.df["low"]) / 2
        if len(median) < 34:
            return 0.0
        ao = median.rolling(5).mean() - median.rolling(34).mean()
        return float(ao.iloc[-1])

    def score_for_signal(self, direction: str) -> BillWilliamsSignal:
        """
        Penalizes signals where the Alligator lines are still tangled
        (jaw/teeth/lips overlapping -- market sleeping/ranging, per Williams'
        own reading) or the Awesome Oscillator momentum contradicts the
        signal direction.
        """
        jaw, teeth, lips = self.alligator()
        ao = self.awesome_oscillator()
        is_long = direction.upper() in ("LONG", "BUY")

        pts = 0
        reasons = []

        lines = sorted([jaw, teeth, lips])
        spread = lines[-1] - lines[0]
        ref = abs(lips) if lips else 1.0
        if ref > 0 and (spread / ref) < 0.0015:
            pts -= 5
            reasons.append("Alligator dormido (lineas entrelazadas, mercado en rango)")

        if is_long and ao < 0:
            pts -= 5
            reasons.append(f"Awesome Oscillator={ao:.5f} negativo, momentum en contra")
        elif not is_long and ao > 0:
            pts -= 5
            reasons.append(f"Awesome Oscillator={ao:.5f} positivo, momentum en contra")

        reason = "; ".join(reasons) if reasons else "Alligator despierto y AO a favor"
        return BillWilliamsSignal(jaw, teeth, lips, ao, pts, reason)
