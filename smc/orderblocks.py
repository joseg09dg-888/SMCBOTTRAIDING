from typing import List, Dict
import pandas as pd


class OrderBlockDetector:
    """
    Bullish OB: last bearish candle before a strong bullish impulse.
    Bearish OB: last bullish candle before a strong bearish impulse.

    BUG-OB-FOREX-DEAD (2026-07-09): impulse_threshold=0.015 (1.5% move in a
    single next candle) was calibrated for crypto, where hourly moves of
    that size are common. Verified against 200 real EURUSD H1 candles: the
    LARGEST single-candle move was 0.415%, average 0.044% -- more than 3.6x
    below the required threshold. find_bullish_obs/find_bearish_obs returned
    ZERO order blocks, always, for the entire live forex pipeline since this
    bot's inception -- the toy test fixtures (8 rows, +-10% synthetic moves)
    never caught this because they don't reflect real forex price scale.
    Fixed: impulse is now ATR-relative (matches smc/structure.py's own
    displacement check, atr*1.5), asset-scale-agnostic -- works for both
    crypto and forex. Falls back to the old percentage threshold only when
    there isn't enough data for a 14-period ATR (small/synthetic dataframes).
    """

    def __init__(self, df: pd.DataFrame, impulse_threshold: float = 0.015, atr_mult: float = 1.0):
        self.df = df.copy()
        self.threshold = impulse_threshold
        self.atr_mult = atr_mult

    def _atr(self, period: int = 14):
        df = self.df
        if len(df) < period + 1:
            return None
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else None

    def find_bullish_obs(self) -> List[Dict]:
        obs = []
        opens  = self.df["open"].values
        closes = self.df["close"].values
        highs  = self.df["high"].values
        lows   = self.df["low"].values
        n = len(closes)
        atr = self._atr()

        for i in range(1, n - 1):
            is_bearish = closes[i] < opens[i]
            if closes[i] == 0:
                continue
            next_move = closes[i + 1] - closes[i]
            is_impulse = (next_move >= atr * self.atr_mult if atr
                          else (next_move / closes[i]) > self.threshold)
            if is_bearish and is_impulse:
                obs.append({
                    "type": "bullish_ob",
                    "index": i,
                    "zone_high": highs[i],
                    "zone_low": lows[i],
                    "ob_close": closes[i],
                    "ob_open": opens[i],
                    "strength": round(next_move / closes[i] * 100, 3),
                })
        return obs

    def find_bearish_obs(self) -> List[Dict]:
        obs = []
        opens  = self.df["open"].values
        closes = self.df["close"].values
        highs  = self.df["high"].values
        lows   = self.df["low"].values
        n = len(closes)
        atr = self._atr()

        for i in range(1, n - 1):
            is_bullish = closes[i] > opens[i]
            if closes[i] == 0:
                continue
            next_move = closes[i] - closes[i + 1]
            is_impulse = (next_move >= atr * self.atr_mult if atr
                          else (next_move / closes[i]) > self.threshold)
            if is_bullish and is_impulse:
                obs.append({
                    "type": "bearish_ob",
                    "index": i,
                    "zone_high": highs[i],
                    "zone_low": lows[i],
                    "ob_close": closes[i],
                    "ob_open": opens[i],
                    "strength": round(next_move / closes[i] * 100, 3),
                })
        return obs

    def is_price_in_ob(self, price: float, ob: Dict) -> bool:
        return ob["zone_low"] <= price <= ob["zone_high"]


class FVGDetector:
    """
    Fair Value Gap: 3-candle imbalance.
    Bullish FVG: low[i] > high[i-2]
    Bearish FVG: high[i] < low[i-2]
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def find_bullish_fvg(self) -> List[Dict]:
        gaps = []
        highs = self.df["high"].values
        lows  = self.df["low"].values
        n = len(highs)

        for i in range(2, n):
            gap_low  = highs[i - 2]
            gap_high = lows[i]
            if gap_high > gap_low:
                gaps.append({
                    "type": "bullish_fvg",
                    "index": i,
                    "gap_high": gap_high,
                    "gap_low": gap_low,
                    "gap_size": round(gap_high - gap_low, 5),
                    "midpoint": round((gap_high + gap_low) / 2, 5),
                })
        return gaps

    def find_bearish_fvg(self) -> List[Dict]:
        gaps = []
        highs = self.df["high"].values
        lows  = self.df["low"].values
        n = len(highs)

        for i in range(2, n):
            gap_high = lows[i - 2]
            gap_low  = highs[i]
            if gap_high > gap_low:
                gaps.append({
                    "type": "bearish_fvg",
                    "index": i,
                    "gap_high": gap_high,
                    "gap_low": gap_low,
                    "gap_size": round(gap_high - gap_low, 5),
                    "midpoint": round((gap_high + gap_low) / 2, 5),
                })
        return gaps

    def summary(self) -> str:
        b = self.find_bullish_fvg()
        bear = self.find_bearish_fvg()
        return (
            f"FVGs Alcistas: {len(b)} | Bajistas: {len(bear)}\n"
            + (f"Último bullish FVG: {b[-1]['gap_low']:.5f} - {b[-1]['gap_high']:.5f}" if b else "")
        )
