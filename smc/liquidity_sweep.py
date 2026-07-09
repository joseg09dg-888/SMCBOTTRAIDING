"""
ICT "Silver Bullet" setup detector: liquidity sweep -> displacement -> fresh
FVG -> inside a kill-zone time window. Unlike the weighted-sum agent scoring
used elsewhere in this bot, real ICT/SMC traders require ALL of these
present together (AND logic) -- missing any one piece means no trade, not
a smaller score. This module implements that all-or-nothing gate.

Kill zones (ET, converted to UTC assuming EDT/UTC-4, the northern-hemisphere
summer offset used most of the trading year): 3-4am ET (07-08 UTC), 10-11am
ET (14-15 UTC), 2-3pm ET (18-19 UTC). Only the 10-11am window overlaps this
bot's actual active hours (14-16, 20-23 UTC) -- the other two fall inside
DEAD_HOURS_UTC, already empirically confirmed as bad trading hours for this
strategy (see bug_tracker.md DIM4 analysis).
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from smc.orderblocks import FVGDetector

KILL_ZONES_UTC = {
    "london_ny_overlap": (7, 8),     # 3-4am ET
    "ny_am": (14, 15),               # 10-11am ET -- the only one active for this bot
    "ny_pm": (18, 19),                # 2-3pm ET -- falls in DEAD_HOURS_UTC
}


@dataclass
class SweepEvent:
    direction: str       # "bullish" (swept a low, expect reversal up) | "bearish"
    swept_level: float
    sweep_index: int


@dataclass
class SilverBulletSignal:
    direction: str
    sweep_level: float
    fvg_high: float
    fvg_low: float
    entry: float
    stop_loss: float
    in_kill_zone: bool
    valid: bool
    reason: str


def in_kill_zone(dt_utc: Optional[datetime] = None) -> bool:
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    hour = dt_utc.hour
    return any(start <= hour < end for start, end in KILL_ZONES_UTC.values())


def in_active_kill_zone(dt_utc: Optional[datetime] = None) -> bool:
    """Only the NY AM window (14-15 UTC) that overlaps this bot's live hours."""
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    start, end = KILL_ZONES_UTC["ny_am"]
    return start <= dt_utc.hour < end


def detect_sweep(df: pd.DataFrame, lookback: int = 20, recent_window: int = 5) -> Optional[SweepEvent]:
    """
    Scans the last `recent_window` bars for a stop-hunt signature: a bar
    that pierced the `lookback`-bar high/low preceding it, then closed back
    inside that range (rejection). Returns the most recent match -- a sweep
    a few bars back is still "fresh" enough for the FVG that follows it to
    be checked, unlike requiring the sweep to be the literal last bar.
    """
    if len(df) < lookback + 2:
        return None

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    start = max(lookback, n - recent_window)
    for i in range(n - 1, start - 1, -1):
        prior_high = highs[i - lookback:i].max()
        prior_low = lows[i - lookback:i].min()

        if lows[i] < prior_low and closes[i] > prior_low:
            return SweepEvent("bullish", float(prior_low), i)
        if highs[i] > prior_high and closes[i] < prior_high:
            return SweepEvent("bearish", float(prior_high), i)

    return None


def check_setup(df: pd.DataFrame, lookback: int = 20, as_of: Optional[datetime] = None) -> Optional[SilverBulletSignal]:
    """
    Full Silver Bullet check: sweep -> fresh FVG in the reversal direction,
    formed at or immediately after the sweep bar -> inside the active kill
    zone. Returns None if any single piece is missing (all-or-nothing).
    """
    sweep = detect_sweep(df, lookback=lookback)
    if sweep is None:
        return None

    fvg_detector = FVGDetector(df)
    fvgs = (fvg_detector.find_bullish_fvg() if sweep.direction == "bullish"
            else fvg_detector.find_bearish_fvg())
    # "Fresh": the FVG's middle candle must be at/after the sweep bar -- not
    # some older gap from earlier in the window.
    fresh_fvgs = [g for g in fvgs if g["index"] >= sweep.sweep_index]
    if not fresh_fvgs:
        return None
    fvg = fresh_fvgs[0]

    kz = in_active_kill_zone(as_of)

    is_bullish = sweep.direction == "bullish"
    entry = fvg["gap_low"] if is_bullish else fvg["gap_high"]
    stop_loss = sweep.swept_level

    valid = kz
    reason = ("setup completo dentro de kill zone" if valid
              else "sweep+FVG validos pero fuera de la kill zone activa (14-15 UTC)")

    return SilverBulletSignal(
        direction=sweep.direction,
        sweep_level=sweep.swept_level,
        fvg_high=fvg["gap_high"],
        fvg_low=fvg["gap_low"],
        entry=entry,
        stop_loss=stop_loss,
        in_kill_zone=kz,
        valid=valid,
        reason=reason,
    )
