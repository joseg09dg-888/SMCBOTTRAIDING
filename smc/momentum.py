"""
Momentum/volatility indicators (RSI, Bollinger Bands, Stochastic, volume) --
classic technical analysis tools requested to complement pure SMC structure,
which has no concept of overbought/oversold, price extension from mean, or
volume confirmation.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class MomentumSignal:
    rsi: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    stoch_k: float
    stoch_d: float
    volume_ratio: Optional[float]
    pts_adjustment: int
    reason: str


class MomentumIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def rsi(self, period: int = 14) -> float:
        """Wilder's RSI. Returns 50.0 (neutral) if not enough data."""
        closes = self.df["close"]
        if len(closes) < period + 1:
            return 50.0
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        last_gain = avg_gain.iloc[-1]
        last_loss = avg_loss.iloc[-1]
        if last_loss == 0:
            return 100.0 if last_gain > 0 else 50.0
        rs = last_gain / last_loss
        return float(100 - (100 / (1 + rs)))

    def bollinger_bands(self, period: int = 20, num_std: float = 2.0) -> tuple[float, float, float]:
        """Returns (upper, mid, lower). Falls back to last close for all
        three bands if not enough data (no false extension signal)."""
        closes = self.df["close"]
        if len(closes) < period:
            last = float(closes.iloc[-1])
            return last, last, last
        window = closes.rolling(period)
        mid = float(window.mean().iloc[-1])
        std = float(window.std().iloc[-1])
        return mid + num_std * std, mid, mid - num_std * std

    def stochastic(self, period: int = 14, smooth_d: int = 3) -> tuple[float, float]:
        """Returns (%K, %D). Returns (50.0, 50.0) neutral if not enough data
        or if the period's range is zero (flat price -- avoids div/0)."""
        if len(self.df) < period:
            return 50.0, 50.0
        high, low, close = self.df["high"], self.df["low"], self.df["close"]
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        rng = highest_high - lowest_low
        k_series = 100 * (close - lowest_low) / rng.replace(0, pd.NA)
        k_series = k_series.fillna(50.0)
        last_k = float(k_series.iloc[-1])
        last_d = float(k_series.rolling(smooth_d).mean().iloc[-1]) if len(k_series) >= smooth_d else last_k
        return last_k, last_d

    def volume_ratio(self, period: int = 20) -> Optional[float]:
        """Current volume / average volume over `period`. None if the
        dataframe has no volume column (some feeds don't provide it)."""
        if "volume" not in self.df.columns or len(self.df) < period + 1:
            return None
        vol = self.df["volume"]
        avg = float(vol.rolling(period).mean().iloc[-2])  # avg up to prior bar
        if avg <= 0:
            return None
        return float(vol.iloc[-1]) / avg

    def score_for_signal(self, direction: str) -> MomentumSignal:
        """
        Penalizes entries into exhausted moves that pure SMC structure can't
        see: buying when RSI is already overbought / price is already above
        the upper Bollinger band, or selling when already oversold / below
        the lower band. Does not reward -- only flags extension risk.
        """
        rsi = self.rsi()
        upper, mid, lower = self.bollinger_bands()
        stoch_k, stoch_d = self.stochastic()
        vol_ratio = self.volume_ratio()
        last_close = float(self.df["close"].iloc[-1])
        is_long = direction.upper() in ("LONG", "BUY")

        pts = 0
        reasons = []

        if is_long and rsi >= 70:
            pts -= 10
            reasons.append(f"RSI={rsi:.0f} sobrecomprado")
        elif not is_long and rsi <= 30:
            pts -= 10
            reasons.append(f"RSI={rsi:.0f} sobrevendido")

        if is_long and upper > mid and last_close >= upper:
            pts -= 5
            reasons.append("precio sobre banda superior de Bollinger")
        elif not is_long and lower < mid and last_close <= lower:
            pts -= 5
            reasons.append("precio bajo banda inferior de Bollinger")

        if is_long and stoch_k >= 80 and stoch_d >= 80:
            pts -= 5
            reasons.append(f"Estocastico={stoch_k:.0f} sobrecomprado")
        elif not is_long and stoch_k <= 20 and stoch_d <= 20:
            pts -= 5
            reasons.append(f"Estocastico={stoch_k:.0f} sobrevendido")

        # Volumen bajo = movimiento sin conviccion real detras, sin importar
        # la direccion -- penaliza igual a LONG y SHORT.
        if vol_ratio is not None and vol_ratio < 0.5:
            pts -= 5
            reasons.append(f"volumen={vol_ratio:.2f}x promedio (bajo, sin conviccion)")

        reason = "; ".join(reasons) if reasons else "sin senal de sobreextension"
        return MomentumSignal(rsi, upper, mid, lower, stoch_k, stoch_d, vol_ratio, pts, reason)
