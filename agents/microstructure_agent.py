from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


@dataclass
class SessionWindow:
    name: str              # "asia", "london", "new_york", "overlap_LN", "dead"
    start_utc: int         # hour (0-23)
    end_utc: int
    score_bonus: int       # bonus points for being in this session
    should_trade: bool     # False for "dead" session


@dataclass
class PsychologicalLevel:
    price: float
    label: str             # "$50,000", "1.1000", etc.
    distance_pct: float    # how far current price is from this level
    stop_hunt_likely: bool # True if price is within 0.5% of level


@dataclass
class MicrostructureSignal:
    current_session: SessionWindow
    session_bonus: int
    day_of_week: str       # "Monday", "Tuesday", etc.
    day_bonus: int         # Tuesday-Thursday = +5, Monday/Friday = 0
    psychological_levels: List[PsychologicalLevel]
    nearest_level: Optional[PsychologicalLevel]
    stop_hunt_detected: bool
    should_trade: bool     # False during dead hours/Friday night
    total_bonus: int       # 0-15
    summary: str


# Session schedule (UTC hours)
SESSIONS = {
    "asia":       SessionWindow("asia",        0,  8,  5, True),
    "pre_london": SessionWindow("pre_london",  7,  8,  8, True),
    "london":     SessionWindow("london",      8, 12, 10, True),
    "overlap_LN": SessionWindow("overlap_LN", 12, 16, 15, True),
    "new_york":   SessionWindow("new_york",   13, 20,  8, True),
    "dead":       SessionWindow("dead",       21, 23,  0, False),
}

PSYCH_LEVELS = {
    "BTCUSDT": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
    "ETHUSDT": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000],
    "EURUSD=X": [1.0500, 1.0700, 1.1000, 1.1200, 1.1500],
    "GC=F":     [1800, 2000, 2200, 2500, 2700, 3000],
}


class MarketMicrostructureAgent:
    """
    Analyzes market microstructure: trading sessions, psychological price levels,
    stop hunt patterns, and day-of-week effects.
    """

    def get_current_session(self, utc_now: datetime = None) -> SessionWindow:
        """Return the most relevant active session for the given UTC time."""
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        hour = utc_now.hour

        # Check sessions in priority order (most specific / highest bonus first)
        # overlap_LN 12-16 has highest bonus, checked before london and new_york
        if 12 <= hour < 16:
            return SESSIONS["overlap_LN"]
        if 8 <= hour < 12:
            return SESSIONS["london"]
        if 7 <= hour < 8:
            return SESSIONS["pre_london"]
        if 13 <= hour < 20:
            return SESSIONS["new_york"]
        if 0 <= hour < 7:
            return SESSIONS["asia"]
        if 20 <= hour < 21:
            return SESSIONS["new_york"]  # late NY
        # 21-23 dead session (handles hour 21 and 22; 23 wraps to dead too)
        return SESSIONS["dead"]

    def get_psychological_levels(
        self, symbol: str, current_price: float
    ) -> List[PsychologicalLevel]:
        """Return psychological levels for the symbol with distance and stop-hunt flag."""
        levels_raw = PSYCH_LEVELS.get(symbol, [])
        result: List[PsychologicalLevel] = []

        for price in levels_raw:
            if price <= 0:
                continue
            distance_pct = abs(current_price - price) / price * 100.0
            stop_hunt_likely = distance_pct <= 0.5

            # Format label
            if price >= 1000:
                label = f"${price:,.0f}"
            elif price >= 1:
                label = f"{price:.4f}"
            else:
                label = f"{price:.6f}"

            result.append(
                PsychologicalLevel(
                    price=price,
                    label=label,
                    distance_pct=round(distance_pct, 4),
                    stop_hunt_likely=stop_hunt_likely,
                )
            )

        return result

    def detect_stop_hunt(
        self,
        symbol: str,
        current_price: float,
        prev_candle_high: float,
        prev_candle_low: float,
    ) -> bool:
        """
        True if price spiked past a psychological level (by >0.3%) then returned
        within the same candle range (current_price back inside prev candle range).
        """
        levels_raw = PSYCH_LEVELS.get(symbol, [])

        for level in levels_raw:
            spike_threshold = level * 0.003  # 0.3%

            # Bullish stop hunt: candle spiked below level then recovered above it
            if (
                prev_candle_low < level - spike_threshold
                and current_price > level
            ):
                return True

            # Bearish stop hunt: candle spiked above level then fell back below it
            if (
                prev_candle_high > level + spike_threshold
                and current_price < level
            ):
                return True

        return False

    def is_trade_blocked(self, utc_now: datetime = None) -> Tuple[bool, str]:
        """
        Returns (blocked, reason).
        Blocked on Friday 20:00+ UTC and during dead session hours (21:00-23:59).
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        # Friday night after 20:00 UTC
        if utc_now.weekday() == 4 and utc_now.hour >= 20:  # weekday 4 = Friday
            return True, "Friday night — markets closing, no trades"

        # Dead session hours 21:00-23:59 UTC
        if utc_now.hour >= 21:
            return True, "Dead session (21:00-23:59 UTC) — no liquidity"

        return False, ""

    def get_signal(
        self,
        symbol: str,
        current_price: float,
        utc_now: datetime = None,
        prev_candle_high: float = None,
        prev_candle_low: float = None,
    ) -> MicrostructureSignal:
        """Build a full MicrostructureSignal for the given symbol and price."""
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        # Session
        session = self.get_current_session(utc_now)

        # Day of week
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = day_names[utc_now.weekday()]
        day_bonus = 5 if day_of_week in ("Tuesday", "Wednesday", "Thursday") else 0

        # Psychological levels
        psych_levels = self.get_psychological_levels(symbol, current_price)
        nearest_level = (
            min(psych_levels, key=lambda lvl: lvl.distance_pct)
            if psych_levels else None
        )

        # Stop hunt detection
        stop_hunt = False
        if prev_candle_high is not None and prev_candle_low is not None:
            stop_hunt = self.detect_stop_hunt(
                symbol, current_price, prev_candle_high, prev_candle_low
            )

        # Trade blocking
        blocked, block_reason = self.is_trade_blocked(utc_now)
        should_trade = session.should_trade and not blocked

        # Total bonus
        if not should_trade:
            session_bonus = 0
            total_bonus = 0
        else:
            session_bonus = session.score_bonus
            stop_hunt_bonus = 10 if stop_hunt else 0
            total_bonus = min(15, session_bonus + day_bonus + stop_hunt_bonus)

        # Summary
        if not should_trade:
            summary = f"TRADE BLOCKED — {block_reason or 'Dead session'}"
        else:
            summary = (
                f"Session: {session.name} (+{session_bonus}pt) | "
                f"Day: {day_of_week} (+{day_bonus}pt) | "
                f"{'Stop hunt detected! ' if stop_hunt else ''}"
                f"Total bonus: {total_bonus}/15"
            )

        return MicrostructureSignal(
            current_session=session,
            session_bonus=session_bonus,
            day_of_week=day_of_week,
            day_bonus=day_bonus,
            psychological_levels=psych_levels,
            nearest_level=nearest_level,
            stop_hunt_detected=stop_hunt,
            should_trade=should_trade,
            total_bonus=total_bonus,
            summary=summary,
        )

    def score_adjustment(
        self,
        symbol: str,
        current_price: float,
        utc_now: datetime = None,
    ) -> int:
        """Return the total bonus score (0-15) for the current microstructure conditions."""
        signal = self.get_signal(symbol, current_price, utc_now)
        return signal.total_bonus

    def format_telegram(self, symbol: str, current_price: float, utc_now: datetime = None) -> str:
        """Format microstructure signal as a Telegram-friendly message."""
        signal = self.get_signal(symbol, current_price, utc_now)

        lines = [
            f"*Market Microstructure — {symbol}*",
            f"Session: `{signal.current_session.name}` (+{signal.session_bonus}pt)",
            f"Day: `{signal.day_of_week}` (+{signal.day_bonus}pt)",
            f"Trade allowed: {'✅ YES' if signal.should_trade else '❌ NO'}",
            f"Stop hunt: {'⚠️ DETECTED' if signal.stop_hunt_detected else '—'}",
        ]

        if signal.nearest_level:
            lvl = signal.nearest_level
            lines.append(
                f"Nearest psych level: `{lvl.label}` ({lvl.distance_pct:.2f}% away)"
            )

        lines.append(f"Microstructure bonus: `{signal.total_bonus}/15`")
        lines.append(signal.summary)

        return "\n".join(lines)
