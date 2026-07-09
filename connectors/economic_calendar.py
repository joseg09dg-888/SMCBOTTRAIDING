"""
Real-time economic calendar connector -- Forex Factory free weekly JSON feed
(no API key required). Replaces/augments the hardcoded FOMC/NFP dates in
strategies/event_driven.py with live high-impact events for any currency
(ECB, BOE, BOC, RBA, RBNZ, SNB rate decisions, CPI, NFP, etc.), not just
the US Fed calendar.

Source publishes at most 2 downloads / 5 min -- module-level cache enforces
a 1h refresh interval so the live bot (many scan cycles/hour) never gets
rate-limited (HTTP 429).
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_TTL_SECONDS = 3600  # respects the feed's 2-downloads/5min limit

_cache: dict = {"ts": 0.0, "events": []}


def currencies_for_symbol(symbol: str) -> set[str]:
    """'USDCAD' -> {'USD','CAD'}, 'EURAUD' -> {'EUR','AUD'}. Non-forex symbols -> empty set."""
    base = symbol.split(".")[0]
    if len(base) == 6 and base.isalpha():
        return {base[:3].upper(), base[3:].upper()}
    return set()


def _parse_events(raw: list) -> list[dict]:
    events = []
    for e in raw:
        try:
            events.append({
                "title": e.get("title", ""),
                "country": e.get("country", ""),
                "impact": e.get("impact", ""),
                "time": datetime.fromisoformat(e["date"]).astimezone(timezone.utc),
            })
        except Exception:
            continue
    return events


def fetch_calendar(force: bool = False) -> list[dict]:
    """Returns cached parsed events, refreshing at most once per hour.
    On any network failure, returns the last good cache (possibly empty)
    instead of raising -- must never break the trading loop."""
    now = time.time()
    if not force and _cache["events"] and (now - _cache["ts"]) < CACHE_TTL_SECONDS:
        return _cache["events"]

    try:
        r = requests.get(CALENDAR_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        events = _parse_events(r.json())
        _cache["events"] = events
        _cache["ts"] = now
    except Exception:
        pass  # keep stale cache (or empty list) -- never crash the caller

    return _cache["events"]


def get_high_impact_window(
    currencies: set[str],
    as_of: Optional[datetime] = None,
    window_minutes: int = 30,
) -> Optional[dict]:
    """Returns the first High-impact event for any of `currencies` within
    `window_minutes` of `as_of` (before or after), or None."""
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    window = timedelta(minutes=window_minutes)

    for ev in fetch_calendar():
        if ev["impact"] != "High" or ev["country"] not in currencies:
            continue
        if abs((ev["time"] - as_of).total_seconds()) <= window.total_seconds():
            return ev
    return None
