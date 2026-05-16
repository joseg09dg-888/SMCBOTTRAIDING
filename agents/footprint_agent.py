"""Footprint candle analysis — order flow delta, absorption, imbalances."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

IMBALANCE_THRESHOLD = 3.0   # ask/bid or bid/ask ratio to flag imbalance


@dataclass
class FootprintLevel:
    price: float
    bid_volume: float
    ask_volume: float
    delta: float
    imbalance: str   # "buying" | "selling" | "neutral"

    @property
    def total_volume(self) -> float:
        return self.bid_volume + self.ask_volume

    @property
    def imbalance_ratio(self) -> float:
        if self.bid_volume == 0:
            return float('inf') if self.ask_volume > 0 else 1.0
        return self.ask_volume / self.bid_volume


@dataclass
class FootprintCandle:
    open: float
    high: float
    low: float
    close: float
    total_volume: float
    total_delta: float
    poc_price: float
    poc_volume: float
    levels: list
    absorption: bool
    exhaustion_high: bool
    exhaustion_low: bool
    stacked_imbalances: int
    imbalance_zones: list

    @property
    def delta_ratio(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.total_delta / self.total_volume

    def to_decision_pts(self) -> int:
        pts = 0
        if self.absorption:              pts += 25
        if self.stacked_imbalances >= 3: pts += 20
        if self.exhaustion_high or self.exhaustion_low: pts -= 15
        return int(pts)


class FootprintAgent:
    TICK_SIZE = 10.0

    def __init__(self, tick_size: float = 10.0,
                 api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.tick_size  = tick_size
        self.api_key    = api_key
        self.api_secret = api_secret
        self.testnet    = testnet

    def _round_to_tick(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 8)

    def build_footprint_from_trades(self, trades: list, candle_open: float,
                                     candle_high: float, candle_low: float,
                                     candle_close: float) -> FootprintCandle:
        levels_dict: dict = {}
        for t in trades:
            p   = self._round_to_tick(float(t['price']))
            qty = float(t['qty'])
            if p not in levels_dict:
                levels_dict[p] = {'bid': 0.0, 'ask': 0.0}
            if t['isBuyerMaker']:
                levels_dict[p]['bid'] += qty
            else:
                levels_dict[p]['ask'] += qty

        levels = []
        for price, vols in sorted(levels_dict.items()):
            bid, ask = vols['bid'], vols['ask']
            delta = ask - bid
            if bid == 0:
                imbalance = "buying" if ask > 0 else "neutral"
            elif ask == 0:
                imbalance = "selling"
            elif ask / bid > IMBALANCE_THRESHOLD:
                imbalance = "buying"
            elif bid / ask > IMBALANCE_THRESHOLD:
                imbalance = "selling"
            else:
                imbalance = "neutral"
            levels.append(FootprintLevel(price, bid, ask, delta, imbalance))

        if not levels:
            return FootprintCandle(
                open=candle_open, high=candle_high, low=candle_low, close=candle_close,
                total_volume=0.0, total_delta=0.0, poc_price=candle_close, poc_volume=0.0,
                levels=[], absorption=False, exhaustion_high=False, exhaustion_low=False,
                stacked_imbalances=0, imbalance_zones=[])

        total_vol   = sum(l.total_volume for l in levels)
        total_delta = sum(l.delta for l in levels)
        poc_level   = max(levels, key=lambda l: l.total_volume)

        candle = FootprintCandle(
            open=candle_open, high=candle_high, low=candle_low, close=candle_close,
            total_volume=total_vol, total_delta=total_delta,
            poc_price=poc_level.price, poc_volume=poc_level.total_volume,
            levels=levels, absorption=False, exhaustion_high=False,
            exhaustion_low=False, stacked_imbalances=0, imbalance_zones=[])

        candle.absorption       = self._detect_absorption(candle)
        ex_h, ex_l              = self._detect_exhaustion(levels)
        candle.exhaustion_high  = ex_h
        candle.exhaustion_low   = ex_l
        count, zones            = self._count_stacked_imbalances(levels)
        candle.stacked_imbalances = count
        candle.imbalance_zones  = zones
        return candle

    def _detect_absorption(self, candle: FootprintCandle) -> bool:
        price_up   = candle.close > candle.open
        price_down = candle.close < candle.open
        tv = candle.total_volume
        return (price_up   and candle.total_delta < -tv * 0.1) or \
               (price_down and candle.total_delta >  tv * 0.1)

    def _detect_exhaustion(self, levels: list) -> tuple:
        if len(levels) < 3:
            return False, False
        n = max(1, len(levels) // 5)   # top/bottom 20%
        top_levels    = levels[-n:]
        bottom_levels = levels[:n]
        ex_high = sum(l.delta for l in top_levels)    < 0
        ex_low  = sum(l.delta for l in bottom_levels) > 0
        return ex_high, ex_low

    def _count_stacked_imbalances(self, levels: list) -> tuple:
        if not levels:
            return 0, []
        max_count = 0
        current   = 0
        current_dir = None
        zones: list = []
        start_price = levels[0].price if levels else 0.0
        for lvl in levels:
            if lvl.imbalance in ("buying", "selling"):
                if lvl.imbalance == current_dir:
                    current += 1
                    if current > max_count:
                        max_count = current
                else:
                    if current >= 3:
                        zones.append({"start_price": start_price,
                                      "end_price": lvl.price,
                                      "direction": current_dir})
                    current_dir = lvl.imbalance
                    current     = 1
                    start_price = lvl.price
            else:
                current = 0; current_dir = None
        if current >= 3:
            zones.append({"start_price": start_price,
                          "end_price": levels[-1].price,
                          "direction": current_dir})
        return max_count, zones

    def fetch_recent_trades(self, symbol: str, limit: int = 1000) -> list:
        try:
            from binance.client import Client
            c = Client(self.api_key, self.api_secret, testnet=self.testnet)
            raw = c.get_aggregate_trades(symbol=symbol.upper(), limit=min(limit, 1000))
            return [{'price': float(t['p']), 'qty': float(t['q']),
                     'isBuyerMaker': bool(t['m'])} for t in raw]
        except Exception:
            return []

    def build_live_footprint(self, symbol: str, candle_open: float = 0,
                              candle_high: float = 0, candle_low: float = 0,
                              candle_close: float = 0) -> Optional[FootprintCandle]:
        trades = self.fetch_recent_trades(symbol, 1000)
        if not trades:
            return None
        prices = [t['price'] for t in trades]
        o = candle_open  or min(prices)
        h = candle_high  or max(prices)
        l = candle_low   or min(prices)
        c = candle_close or prices[-1]
        return self.build_footprint_from_trades(trades, o, h, l, c)

    def score_for_trade(self, candle: FootprintCandle,
                        trade_direction: str, entry_price: float) -> int:
        pts = 0
        dr  = candle.delta_ratio
        # POC proximity (<0.5%)
        if entry_price > 0 and abs(candle.poc_price - entry_price) / entry_price < 0.005:
            pts += 15
        # Delta confirmation / divergence
        if trade_direction == "long":
            if dr > 0.1:    pts += 10
            elif dr < -0.1: pts -= 20
        else:
            if dr < -0.1:   pts += 10
            elif dr > 0.1:  pts -= 20
        # Absorption
        if candle.absorption:
            pts += 25
        # Stacked imbalances
        if candle.stacked_imbalances >= 3:
            pts += 20
        # Exhaustion penalty
        if trade_direction == "long"  and candle.exhaustion_high: pts -= 15
        if trade_direction == "short" and candle.exhaustion_low:  pts -= 15
        return int(max(-30, min(30, pts)))

    def format_telegram(self, candle: FootprintCandle, symbol: str = "") -> str:
        total_buy = sum(l.ask_volume for l in candle.levels)
        total_sell = sum(l.bid_volume for l in candle.levels)
        delta_emoji = "🟢" if candle.total_delta >= 0 else "🔴"
        signals = []
        if candle.absorption:       signals.append("Absorcion detectada")
        if candle.exhaustion_high:  signals.append("Exhaustion en maximos")
        if candle.exhaustion_low:   signals.append("Exhaustion en minimos")
        if candle.stacked_imbalances >= 3:
            signals.append(f"Stacked imbalances x{candle.stacked_imbalances}")
        sig_text = "\n  ".join(signals) if signals else "Sin señales especiales"
        return (
            f"FOOTPRINT ANALYSIS{(' — ' + symbol) if symbol else ''}\n"
            f"--------------------\n"
            f"Compradores: {total_buy:,.1f}\n"
            f"Vendedores:  {total_sell:,.1f}\n"
            f"Delta: {candle.total_delta:+,.1f} {delta_emoji}\n"
            f"POC: {candle.poc_price:,.4f} (vol {candle.poc_volume:,.1f})\n"
            f"Senales:\n  {sig_text}\n"
            f"Delta ratio: {candle.delta_ratio:+.3f}\n"
            f"Edge footprint: {candle.to_decision_pts():+d} pts"
        )
