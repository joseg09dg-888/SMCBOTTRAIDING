"""Order flow analysis: imbalance, VPIN, spread."""
from dataclasses import dataclass
import numpy as np

@dataclass
class OrderFlowSignal:
    imbalance: float
    vpin: float
    spread_pct: float
    pressure: str
    pts: int
    toxic_flow: bool

class OrderFlowAnalyzer:
    @staticmethod
    def calculate_imbalance(bid_volumes, ask_volumes):
        b = float(sum(bid_volumes)) if bid_volumes else 0.0
        a = float(sum(ask_volumes)) if ask_volumes else 0.0
        total = b + a
        return 0.0 if total == 0 else (b-a)/total

    @staticmethod
    def calculate_vpin(buy_volumes, sell_volumes, bucket_size=10):
        if not buy_volumes or not sell_volumes: return 0.0
        n = min(len(buy_volumes), len(sell_volumes))
        if n == 0: return 0.0
        bv = np.array(buy_volumes[:n]); sv = np.array(sell_volumes[:n])
        total = bv+sv
        if bucket_size >= n:
            mask = total > 0
            if not np.any(mask): return 0.0
            return float(np.mean(np.abs(bv[mask]-sv[mask])/total[mask]))
        n_buckets = n // bucket_size
        vpins = []
        for i in range(n_buckets):
            sl = slice(i*bucket_size,(i+1)*bucket_size)
            tb = float(np.sum(bv[sl])); ts = float(np.sum(sv[sl])); tot = tb+ts
            if tot > 0: vpins.append(abs(tb-ts)/tot)
        return float(np.mean(vpins)) if vpins else 0.0

    @staticmethod
    def calculate_spread_pct(bid, ask):
        if bid <= 0 or ask <= 0: return 0.0
        mid = (bid+ask)/2
        return (ask-bid)/mid if mid > 0 else 0.0

    @staticmethod
    def classify_pressure(imbalance):
        if imbalance > 0.4:   return ("strong_buy", 8)
        elif imbalance > 0.15: return ("buy", 4)
        elif imbalance < -0.4: return ("strong_sell", -8)
        elif imbalance < -0.15:return ("sell", -4)
        return ("neutral", 0)

    def analyze(self, bid_volumes, ask_volumes, buy_volumes=None, sell_volumes=None,
                best_bid=0.0, best_ask=0.0):
        imb = self.calculate_imbalance(bid_volumes, ask_volumes)
        bv = buy_volumes if buy_volumes is not None else bid_volumes
        sv = sell_volumes if sell_volumes is not None else ask_volumes
        vpin = self.calculate_vpin(bv, sv)
        spread = self.calculate_spread_pct(best_bid, best_ask)
        pressure, pts = self.classify_pressure(imb)
        return OrderFlowSignal(
            imbalance=imb, vpin=vpin, spread_pct=spread,
            pressure=pressure, pts=int(np.clip(pts,-10,10)),
            toxic_flow=vpin>0.7)

    @staticmethod
    def detect_iceberg_probability(visible_size, executed_size):
        return float(min(executed_size/(visible_size+1), 1.0))

    @staticmethod
    def estimate_market_impact(order_size, avg_daily_volume, volatility):
        if avg_daily_volume == 0: return 0.0
        return float(volatility * np.sqrt(order_size/avg_daily_volume))
