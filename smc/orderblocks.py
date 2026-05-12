from typing import List, Dict
import pandas as pd


class OrderBlockDetector:
    """
    Bullish OB: last bearish candle before a strong bullish impulse.
    Bearish OB: last bullish candle before a strong bearish impulse.
    """

    def __init__(self, df: pd.DataFrame, impulse_threshold: float = 0.005):
        self.df = df.copy()
        self.threshold = impulse_threshold

    def find_bullish_obs(self) -> List[Dict]:
        obs = []
        opens  = self.df["open"].values
        closes = self.df["close"].values
        highs  = self.df["high"].values
        lows   = self.df["low"].values
        n = len(closes)

        for i in range(1, n - 1):
            is_bearish = closes[i] < opens[i]
            next_move  = (closes[i + 1] - closes[i]) / closes[i]
            if is_bearish and next_move > self.threshold:
                obs.append({
                    "type": "bullish_ob",
                    "index": i,
                    "zone_high": highs[i],
                    "zone_low": lows[i],
                    "ob_close": closes[i],
                    "ob_open": opens[i],
                    "strength": round(next_move * 100, 3),
                })
        return obs

    def find_bearish_obs(self) -> List[Dict]:
        obs = []
        opens  = self.df["open"].values
        closes = self.df["close"].values
        highs  = self.df["high"].values
        lows   = self.df["low"].values
        n = len(closes)

        for i in range(1, n - 1):
            is_bullish = closes[i] > opens[i]
            next_move  = (closes[i] - closes[i + 1]) / closes[i]
            if is_bullish and next_move > self.threshold:
                obs.append({
                    "type": "bearish_ob",
                    "index": i,
                    "zone_high": highs[i],
                    "zone_low": lows[i],
                    "ob_close": closes[i],
                    "ob_open": opens[i],
                    "strength": round(next_move * 100, 3),
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
