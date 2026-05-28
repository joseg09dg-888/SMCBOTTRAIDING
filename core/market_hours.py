"""Market hours for each symbol traded by the bot.

Rules (UTC):
  Forex (EURUSD, GBPUSD, USDJPY, GBPJPY): Mon 00:00 - Fri 21:00 UTC
  Gold (XAUUSD): Mon 01:00 - Fri 21:00 UTC (with short daily breaks)
  US30 (Dow Jones): Mon-Fri 13:30-20:00 UTC (NYSE regular hours)
                    Pre/post market ignored — only trade regular hours
  NAS100 (Nasdaq): same as US30
  Crypto (BTC, ETH, etc.): 24/7 — always open

For Axi specifically:
  - US30 and NAS100 have a daily 15-min break at 21:00 UTC
  - Forex has no break (except weekend)
  - Weekend: all markets closed Fri 21:00 - Sun 23:00 UTC
"""
from __future__ import annotations

from datetime import datetime, timezone, time


# (open_hour_utc, open_min, close_hour_utc, close_min) per weekday 0-4 (Mon-Fri)
# None means closed that day.
_HOURS: dict[str, dict] = {
    # Forex — continuous Mon-Fri (Axi)
    "EURUSD": {"type": "forex"},
    "GBPUSD": {"type": "forex"},
    "USDJPY": {"type": "forex"},
    "GBPJPY": {"type": "forex"},
    "USDCHF": {"type": "forex"},
    "AUDUSD": {"type": "forex"},
    # Gold — nearly 24/5 but skip the 21:00-22:00 daily break
    "XAUUSD": {"type": "gold"},
    # US indices — NYSE hours only (13:30-20:00 UTC)
    "US30":   {"type": "index_us"},
    "NAS100": {"type": "index_us"},
    # Crypto — always open
    "BTCUSDT":  {"type": "crypto"},
    "ETHUSDT":  {"type": "crypto"},
    "SOLUSDT":  {"type": "crypto"},
    "BNBUSDT":  {"type": "crypto"},
    "XRPUSDT":  {"type": "crypto"},
    "ADAUSDT":  {"type": "crypto"},
}


def is_market_open(symbol: str, dt: datetime | None = None) -> bool:
    """Return True if the market for symbol is open at the given UTC datetime."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    weekday = dt.weekday()  # 0=Mon, 6=Sun
    hour    = dt.hour
    minute  = dt.minute

    info = _HOURS.get(symbol, {"type": "forex"})
    mtype = info["type"]

    # Weekend: all non-crypto closed
    if mtype != "crypto" and weekday >= 5:
        return False

    # Friday close: Axi closes forex/gold/indices at 21:00 UTC Friday
    if weekday == 4 and mtype != "crypto" and (hour > 21 or (hour == 21 and minute >= 0)):
        return False

    if mtype == "crypto":
        return True

    if mtype == "forex":
        # Monday opens 00:00 UTC, continuous until Friday 21:00 UTC
        if weekday == 0 and hour < 0:
            return False
        return True

    if mtype == "gold":
        # Daily break 21:00-22:00 UTC (Axi)
        if hour == 21:
            return False
        return True

    if mtype == "index_us":
        # NYSE regular hours: 13:30-20:00 UTC Mon-Fri
        open_time  = time(13, 30)
        close_time = time(20, 0)
        now_time   = dt.time().replace(tzinfo=None)
        return open_time <= now_time < close_time

    return True


def minutes_until_open(symbol: str, dt: datetime | None = None) -> int:
    """Return minutes until market opens. 0 if already open."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if is_market_open(symbol, dt):
        return 0

    info = _HOURS.get(symbol, {"type": "forex"})
    mtype = info["type"]

    if mtype == "index_us":
        # If before 13:30 UTC on a weekday, opens at 13:30
        now_time = dt.time()
        if dt.weekday() < 5:
            open_dt = dt.replace(hour=13, minute=30, second=0, microsecond=0)
            if dt < open_dt:
                delta = open_dt - dt
                return int(delta.total_seconds() // 60)
        # Weekend or after close: opens Monday 13:30
        days_to_monday = (7 - dt.weekday()) % 7 or 7
        from datetime import timedelta
        next_open = (dt + timedelta(days=days_to_monday)).replace(
            hour=13, minute=30, second=0, microsecond=0
        )
        return int((next_open - dt).total_seconds() // 60)

    if mtype == "gold" and dt.hour == 21:
        return 60 - dt.minute  # break lasts until 22:00

    return 60  # generic fallback
