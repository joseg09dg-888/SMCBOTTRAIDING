from datetime import datetime, timezone, time as dtime
from typing import Tuple


# UTC hours for each major session
SESSIONS = {
    "tokyo":   (0,  9),   # 00:00 - 09:00 UTC
    "london":  (7,  16),  # 07:00 - 16:00 UTC  (overlap: 07-09)
    "new_york":(12, 21),  # 12:00 - 21:00 UTC  (overlap: 12-16)
}

# Sessions with highest SMC opportunity (liquidity sweeps, BOS, OB taps)
PREMIUM_SESSIONS = {"london", "new_york"}
OVERLAP_HOURS = set(range(7, 9)) | set(range(12, 16))  # London-Tokyo & London-NY overlaps


def get_active_sessions(dt: datetime = None) -> list:
    dt = dt or datetime.now(timezone.utc)
    h = dt.hour
    active = []
    for name, (start, end) in SESSIONS.items():
        if start <= h < end:
            active.append(name)
    return active


def is_premium_session(dt: datetime = None) -> bool:
    dt = dt or datetime.now(timezone.utc)
    active = get_active_sessions(dt)
    return any(s in PREMIUM_SESSIONS for s in active)


def is_overlap(dt: datetime = None) -> bool:
    dt = dt or datetime.now(timezone.utc)
    return dt.hour in OVERLAP_HOURS


def session_score(dt: datetime = None) -> Tuple[int, str]:
    """
    Returns (score 0-8, reason) for the current session.
    Premium session overlap = 8, premium single = 6, Tokyo only = 3.
    """
    dt = dt or datetime.now(timezone.utc)
    active = get_active_sessions(dt)

    if not active:
        return 0, "Mercado cerrado — sin sesión activa"

    if is_overlap(dt):
        return 8, f"Overlap de sesiones ({'+'.join(active)}) — máxima liquidez"

    if "new_york" in active:
        return 7, "Sesión New York — alta liquidez USD"
    if "london" in active:
        return 6, "Sesión London — mejor para EUR/GBP/metals"
    if "tokyo" in active:
        return 3, "Sesión Tokyo — liquidez reducida (mejor para JPY/AUD)"

    return 2, f"Sesión fuera de hora óptima: {active}"


def session_multiplier(dt: datetime = None) -> float:
    """Hour-quality multiplier for threshold adjustment.
    Threshold = base_threshold / multiplier  (higher mult = lower bar = more trades)

    BUG-KZ-HOUR14-STALE-GOLD (2026-07-26): the "hour 14 = gold, WR=61%" claim
    below was from a 2-year backtest that (a) used yfinance data capped at
    <730 days and (b) predated today's session's market-reading/exit-guard
    fixes. Re-ran DIM4 against ~16.1 years of REAL MT5 history (3,915
    trading days, 25,962 trades) with all of today's fixes applied:
      14:00 UTC: 6,952 trades, WR=29%, avg=-$35   -- EVITAR (was called "gold")
      15:00 UTC: 2,657 trades, WR=57%, avg=+$149  -- PREMIUM (was "continuation")
      16:00 UTC: 2,880 trades, WR=46%, avg=+$80   -- BUENA
      20:00 UTC: 6,833 trades, WR=50%, avg=+$89   -- PREMIUM (was "closing", 1.05)
      21:00 UTC: 2,607 trades, WR=51%, avg=+$43   -- BUENA
      22:00 UTC: 2,210 trades, WR=50%, avg=+$44   -- BUENA
      23:00 UTC: 1,823 trades, WR=50%, avg=+$40   -- BUENA
    Hour 14 was getting the single LOWEST bar of the whole table (1.30x,
    easiest to qualify) while real data shows it's the only hour with
    negative expectancy of the 7 actively-scanned hours -- backwards.
    Rebuilt the table from these numbers directly (multiplier roughly
    tracks avg P&L rank); 17/18/19 stay untouched (blocked entirely by
    DEAD_HOURS_UTC in core/supervisor.py, so no trades exist there to
    re-measure).
    """
    dt = dt or datetime.now(timezone.utc)
    h = dt.hour
    _HOUR_MULT = {
        14: 0.70,  # 16y real data: WR=29%, avg=-$35 -- worst active hour, was 1.30 (backwards)
        15: 1.30,  # 16y real data: WR=57%, avg=+$149 -- best active hour
        16: 1.15,  # 16y real data: WR=46%, avg=+$80
        17: 0.85,  # no active trades (DEAD_HOURS_UTC blocks it) -- unchanged
        18: 0.80,  # no active trades (DEAD_HOURS_UTC blocks it) -- unchanged
        19: 1.00,  # no active trades (DEAD_HOURS_UTC blocks it) -- unchanged
        20: 1.20,  # 16y real data: WR=50%, avg=+$89
        21: 1.00,  # 16y real data: WR=51%, avg=+$43
        22: 1.00,  # 16y real data: WR=50%, avg=+$44
        23: 0.95,  # 16y real data: WR=50%, avg=+$40
    }
    return _HOUR_MULT.get(h, 0.90)
