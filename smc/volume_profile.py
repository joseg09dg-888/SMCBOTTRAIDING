from typing import Dict, List
import pandas as pd
import numpy as np


class VolumeProfile:
    """
    Calculates Point of Control (POC), Value Area High (VAH),
    Value Area Low (VAL). Value Area = 70% of volume.
    """

    def __init__(self, df: pd.DataFrame, bins: int = 50):
        self.df = df.copy()
        self.bins = bins

    def calculate(self) -> Dict:
        highs   = self.df["high"].values
        lows    = self.df["low"].values
        closes  = self.df["close"].values
        volumes = self.df["volume"].values

        price_min = lows.min()
        price_max = highs.max()
        price_levels = np.linspace(price_min, price_max, self.bins)
        vol_at_price = np.zeros(self.bins)

        for i in range(len(closes)):
            typical = (highs[i] + lows[i] + closes[i]) / 3
            idx = int(np.searchsorted(price_levels, typical))
            idx = min(idx, self.bins - 1)
            vol_at_price[idx] += volumes[i]

        poc_idx = int(np.argmax(vol_at_price))
        poc = float(price_levels[poc_idx])
        total_volume = vol_at_price.sum()
        target_volume = total_volume * 0.70

        upper = lower = poc_idx
        accumulated = vol_at_price[poc_idx]

        while accumulated < target_volume:
            can_up   = upper < self.bins - 1
            can_down = lower > 0
            if not can_up and not can_down:
                break
            up_vol   = vol_at_price[upper + 1] if can_up   else 0
            down_vol = vol_at_price[lower - 1] if can_down else 0
            if up_vol >= down_vol and can_up:
                upper += 1
                accumulated += up_vol
            elif can_down:
                lower -= 1
                accumulated += down_vol
            else:
                upper += 1
                accumulated += up_vol

        return {
            "poc": round(poc, 5),
            "vah": round(float(price_levels[upper]), 5),
            "val": round(float(price_levels[lower]), 5),
            "value_area_pct": round(accumulated / total_volume, 4),
            "poc_volume": round(float(vol_at_price[poc_idx]), 2),
        }

    def summary(self) -> str:
        r = self.calculate()
        return (
            f"Volume Profile:\n"
            f"  POC: {r['poc']:.5f} | VAH: {r['vah']:.5f} | VAL: {r['val']:.5f}\n"
            f"  Value Area: {r['value_area_pct']*100:.1f}% del volumen"
        )


class AnchoredVWAP:
    """
    VWAP anchored from a specific bar (swing high/low, event, etc.)
    Formula: VWAP = Σ(typical_price × volume) / Σ(volume)
    """

    def __init__(self, df: pd.DataFrame, anchor_index: int = 0):
        self.df = df.copy()
        self.anchor = anchor_index

    def calculate(self) -> List[float]:
        sub = self.df.iloc[self.anchor:].copy()
        typical = (sub["high"] + sub["low"] + sub["close"]) / 3
        cum_tp_vol = (typical * sub["volume"]).cumsum()
        cum_vol    = sub["volume"].cumsum()
        vwap_series = (cum_tp_vol / cum_vol).tolist()
        return vwap_series

    def is_price_above_vwap(self, current_price: float) -> bool:
        vwap = self.calculate()
        last_vwap = vwap[-1] if vwap else None
        if last_vwap is None:
            return False
        return bool(current_price > last_vwap)

    def summary(self, current_price: float) -> str:
        vwap = self.calculate()
        last = vwap[-1] if vwap else 0
        pos = "ENCIMA" if current_price > last else "DEBAJO"
        return f"VWAP Anclado: {last:.5f} | Precio {pos} del VWAP"
