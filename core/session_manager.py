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
    Based on 2-year backtest WR per hour UTC + ICT killzone data.
    Threshold = base_threshold / multiplier  (higher mult = lower bar = more trades)
    """
    dt = dt or datetime.now(timezone.utc)
    h = dt.hour
    _HOUR_MULT = {
        14: 1.30,  # NY open — WR=61% gold hour
        15: 1.20,  # NY continuation
        16: 1.10,  # London close / NY active
        17: 0.85,  # NY mid — WR drops per backtest
        18: 0.80,  # NY mid — worst hour in backtest
        19: 1.00,  # NY late PM
        20: 1.05,  # NY closing
        21: 0.90,  # NY end
        22: 0.95,  # After NY
        23: 0.90,  # Late session
    }
    return _HOUR_MULT.get(h, 0.90)
