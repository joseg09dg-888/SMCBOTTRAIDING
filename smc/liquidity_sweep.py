"""
ICT "Silver Bullet" setup detector: liquidity sweep -> displacement -> fresh
FVG -> inside a kill-zone time window. Unlike the weighted-sum agent scoring
used elsewhere in this bot, real ICT/SMC traders require ALL of these
present together (AND logic) -- missing any one piece means no trade, not
a smaller score. This module implements that all-or-nothing gate.

Kill zones (ET, converted to UTC assuming EDT/UTC-4, the northern-hemisphere
summer offset used most of the trading year): 3-4am ET (07-08 UTC), 10-11am
ET (14-15 UTC), 2-3pm ET (18-19 UTC).

BUG-SB-DEAD-KILLZONE (2026-07-28): in_active_kill_zone() only recognized
hour 14 UTC as valid (14 <= hour < 15). Hour 14 UTC was hard-blocked in
DEAD_HOURS_UTC on 2026-07-26 (16-year real data: WR=29%, avg=-$35, the
worst active hour) -- since that deploy, this entire module could never
fire again in live trading or in the backtest (scripts/backtest_multiyear.py
literally gated the call behind `hour_utc == 14`). Updated to match the
bot's real active hours (15, 16, 20, 21, 22, 23 UTC -- see
core/session_manager.py _HOUR_MULT), so the sweep+FVG+killzone confluence
this module checks can actually be evaluated during hours real orders fire.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from smc.orderblocks import FVGDetector

KILL_ZONES_UTC = {
    "london_ny_overlap": (7, 8),     # 3-4am ET -- falls in DEAD_HOURS_UTC
    "ny_am": (14, 15),               # 10-11am ET -- 14 is DEAD_HOURS_UTC now, kept for reference
    "ny_pm": (18, 19),                # 2-3pm ET -- falls in DEAD_HOURS_UTC
}

# Real active hours for MT5 real orders (core/session_manager.py _HOUR_MULT,
# 16-year real data). Used by in_active_kill_zone() instead of the classic
# ICT NY-AM window, which is now entirely inside DEAD_HOURS_UTC.
ACTIVE_HOURS_UTC = {15, 16, 20, 21, 22, 23}


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
    """True during this bot's real active hours (15,16,20-23 UTC) -- see
    BUG-SB-DEAD-KILLZONE above for why this no longer uses the classic
    14-15 UTC NY-AM window."""
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    return dt_utc.hour in ACTIVE_HOURS_UTC


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
              else "sweep+FVG validos pero fuera de la kill zone activa (15,16,20-23 UTC)")

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
