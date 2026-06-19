"""Market hours for each symbol traded by the bot.

Rules (UTC):
  Forex (EURUSD, GBPUSD, USDJPY, GBPJPY): Mon 00:00 - Fri 21:00 UTC
  Gold (XAUUSD): Mon 01:00 - Fri 21:00 UTC (with short daily breaks)
  US30 / NAS100: Axi CFD hours -- Mon 01:00 to Fri 22:00 UTC
                 daily break 22:00-23:00 UTC (Axi maintenance)
  Crypto (BTC, ETH, etc.): 24/7 -- always open

For Axi specifically:
  - US30 and NAS100 have a 1-hour break at 22:00 UTC each day
  - Forex has no break (except weekend)
  - Weekend: all markets closed Fri 22:00 - Sun 23:00 UTC
"""
from __future__ import annotations

from datetime import datetime, timezone, time, timedelta


_HOURS: dict[str, dict] = {
    "EURUSD": {"type": "forex"},
    "GBPUSD": {"type": "forex"},
    "USDJPY": {"type": "forex"},
    "GBPJPY": {"type": "forex"},
    "USDCHF": {"type": "forex"},
    "AUDUSD": {"type": "forex"},
    "USDCAD": {"type": "forex"},
    "XAUUSD": {"type": "gold"},
    "US30":   {"type": "index_us"},
    "NAS100": {"type": "index_us"},
    "BTCUSDT":  {"type": "crypto"},
    "ETHUSDT":  {"type": "crypto"},
    "SOLUSDT":  {"type": "crypto"},
    "BNBUSDT":  {"type": "crypto"},
    "XRPUSDT":  {"type": "crypto"},
    "ADAUSDT":  {"type": "crypto"},
}

_INDEX_OPEN_HOUR  = 1   # 01:00 UTC -- Axi US index CFD open
_INDEX_CLOSE_HOUR = 22  # 22:00 UTC -- daily break starts


def is_market_open(symbol: str, dt: datetime | None = None) -> bool:
    """Return True if the market for symbol is open at the given UTC datetime."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    weekday = dt.weekday()  # 0=Mon, 6=Sun
    hour    = dt.hour

    info = _HOURS.get(symbol, {"type": "forex"})
    mtype = info["type"]

    if mtype == "crypto":
        return True

    # Weekend: closed Sat/Sun
    if weekday >= 5:
        return False

    # Friday end-of-week close
    if weekday == 4:
        if mtype == "index_us" and hour >= 22:
            return False
        elif mtype != "index_us" and hour >= 21:
            return False

    if mtype == "forex":
        return True

    if mtype == "gold":
        if hour == 21:  # daily break 21:00-22:00 UTC
            return False
        return True

    if mtype == "index_us":
        # Axi CFD: open 01:00-22:00 UTC Mon-Fri, break 22:00-23:00
        if hour < _INDEX_OPEN_HOUR or hour >= _INDEX_CLOSE_HOUR:
            return False
        return True

    return True


def minutes_until_open(symbol: str, dt: datetime | None = None) -> int:
    """Return minutes until market opens. 0 if already open."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if is_market_open(symbol, dt):
        return 0

    info = _HOURS.get(symbol, {"type": "forex"})
    mtype = info["type"]
    weekday = dt.weekday()

    if mtype == "index_us":
        if weekday < 4:
            # Mon-Thu: after daily close -> tomorrow 01:00; before open -> today 01:00
            if dt.hour >= _INDEX_CLOSE_HOUR:
                next_open = (dt + timedelta(days=1)).replace(
                    hour=_INDEX_OPEN_HOUR, minute=0, second=0, microsecond=0
                )
            else:
                next_open = dt.replace(
                    hour=_INDEX_OPEN_HOUR, minute=0, second=0, microsecond=0
                )
        elif weekday == 4:
            # Friday after 22:00 -> Monday 01:00 (3 days)
            next_open = (dt + timedelta(days=3)).replace(
                hour=_INDEX_OPEN_HOUR, minute=0, second=0, microsecond=0
            )
        else:
            # Sat(5) -> 2 days to Mon, Sun(6) -> 1 day to Mon
            days_to_monday = 7 - weekday
            next_open = (dt + timedelta(days=days_to_monday)).replace(
                hour=_INDEX_OPEN_HOUR, minute=0, second=0, microsecond=0
            )
        return int((next_open - dt).total_seconds() // 60)

    if mtype == "gold" and dt.hour == 21:
        return 60 - dt.minute

    if mtype in ("forex", "gold"):
        open_hour = 1 if mtype == "gold" else 0
        if weekday >= 5:
            days_to_monday = 7 - weekday
            next_open = (dt + timedelta(days=days_to_monday)).replace(
                hour=open_hour, minute=0, second=0, microsecond=0
            )
        else:
            # Friday after close -> next Monday
            next_open = (dt + timedelta(days=(7 - weekday))).replace(
                hour=open_hour, minute=0, second=0, microsecond=0
            )
        return int((next_open - dt).total_seconds() // 60)

    return 60
