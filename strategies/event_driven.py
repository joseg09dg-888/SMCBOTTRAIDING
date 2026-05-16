"""
Event-driven trading strategy for SMC Trading Bot.

Detects macro economic events (FOMC, NFP, BTC halvings) and adjusts
strategy signals accordingly. No external API calls — all calendars
are hardcoded.
"""
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Optional


class EventType(Enum):
    FOMC        = "fomc"
    NFP         = "nfp"
    CPI         = "cpi"
    HALVING     = "halving"
    LISTING     = "exchange_listing"
    LIQUIDATION = "liquidation_cascade"
    EARNINGS    = "earnings"
    OTHER       = "other"


class EventImpact(Enum):
    CRITICAL = "critical"   # immediate trade
    HIGH     = "high"       # prepare position
    MEDIUM   = "medium"     # monitor
    LOW      = "low"        # ignore


@dataclass
class EconomicEvent:
    name: str
    event_type: EventType
    impact: EventImpact
    scheduled_at: datetime
    symbol_bias: dict       # e.g. {"EURUSD": "sell", "XAUUSD": "buy"}
    expected_move_pct: float  # historical avg % move on this event
    win_rate: float           # historical win rate trading this event
    notes: str = ""


@dataclass
class EventSignal:
    event: EconomicEvent
    action: str             # "long" | "short" | "reduce_risk" | "wait"
    symbol: str
    pts_adjustment: int     # -20 to +20 for DecisionFilter
    urgency: str            # "immediate" | "prepare" | "monitor"
    risk_multiplier: float  # 0.0-1.0


class EventDrivenStrategy:
    """
    Event-driven trading: detect macro events and adjust strategy accordingly.
    Calendar is hardcoded (no external API calls).
    """

    # BTC halving dates
    HALVING_DATES = [
        datetime(2012, 11, 28, tzinfo=timezone.utc),
        datetime(2016, 7, 9, tzinfo=timezone.utc),
        datetime(2020, 5, 11, tzinfo=timezone.utc),
        datetime(2024, 4, 20, tzinfo=timezone.utc),
    ]

    # FOMC meetings 2026 (approximate dates)
    FOMC_2026 = [
        datetime(2026, 1, 29, 19, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 7, 18, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 17, 18, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 18, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 4, 19, 0, tzinfo=timezone.utc),
        datetime(2026, 12, 16, 19, 0, tzinfo=timezone.utc),
    ]

    # ── Halving ───────────────────────────────────────────────────────────

    def get_days_since_last_halving(self, as_of: datetime = None) -> int:
        """Returns days since most recent BTC halving."""
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        past_halvings = [h for h in self.HALVING_DATES if h <= as_of]
        if not past_halvings:
            # as_of is before any known halving; measure from the first one
            return (as_of - self.HALVING_DATES[0]).days

        last_halving = max(past_halvings)
        return (as_of - last_halving).days

    def get_halving_phase(self, days_since_halving: int) -> dict:
        """
        Returns dict with:
        - phase: "accumulation" | "bull_run" | "euphoria" | "distribution" | "bear"
        - days_range: (start, end)
        - bias: "bullish" | "very_bullish" | "reduce_longs" | "bearish" | "neutral"
        - pts: -10 to +10

        Phases:
        0-180:   accumulation, bullish +5
        181-365: bull_run, very_bullish +10
        366-500: euphoria, reduce_longs -3
        501-730: distribution, bearish -5
        730+:    bear, bearish -8
        """
        d = days_since_halving
        if d <= 180:
            return {
                "phase": "accumulation",
                "days_range": (0, 180),
                "bias": "bullish",
                "pts": 5,
            }
        elif d <= 365:
            return {
                "phase": "bull_run",
                "days_range": (181, 365),
                "bias": "very_bullish",
                "pts": 10,
            }
        elif d <= 500:
            return {
                "phase": "euphoria",
                "days_range": (366, 500),
                "bias": "reduce_longs",
                "pts": -3,
            }
        elif d <= 730:
            return {
                "phase": "distribution",
                "days_range": (501, 730),
                "bias": "bearish",
                "pts": -5,
            }
        else:
            return {
                "phase": "bear",
                "days_range": (730, 99999),
                "bias": "bearish",
                "pts": -8,
            }

    # ── FOMC ─────────────────────────────────────────────────────────────

    def get_next_fomc(self, as_of: datetime = None) -> Optional[datetime]:
        """Returns next FOMC date from FOMC_2026, or None if past all."""
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        future = [f for f in self.FOMC_2026 if f > as_of]
        return min(future) if future else None

    def is_fomc_window(self, as_of: datetime = None) -> bool:
        """True if within 1 hour before or after FOMC."""
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        window = timedelta(hours=1)
        for fomc in self.FOMC_2026:
            if abs((as_of - fomc).total_seconds()) <= window.total_seconds():
                return True
        return False

    # ── NFP ──────────────────────────────────────────────────────────────

    def get_nfp_dates_2026(self) -> list:
        """
        NFP = first Friday of each month at 12:30 UTC.
        Return list of all NFP dates for 2026.
        """
        nfp_dates = []
        for month in range(1, 13):
            first_day = date(2026, month, 1)
            day_of_week = first_day.weekday()   # 0=Monday, 4=Friday
            days_to_friday = (4 - day_of_week) % 7
            first_friday = first_day + timedelta(days=days_to_friday)
            nfp_dt = datetime(
                first_friday.year,
                first_friday.month,
                first_friday.day,
                12, 30,
                tzinfo=timezone.utc,
            )
            nfp_dates.append(nfp_dt)
        return nfp_dates

    def is_nfp_window(self, as_of: datetime = None) -> bool:
        """True if within 30 minutes of NFP release."""
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        window = timedelta(minutes=30)
        for nfp in self.get_nfp_dates_2026():
            if abs((as_of - nfp).total_seconds()) <= window.total_seconds():
                return True
        return False

    # ── Upcoming events ───────────────────────────────────────────────────

    def get_upcoming_events(
        self,
        as_of: datetime = None,
        days_ahead: int = 7,
    ) -> list:
        """
        Returns list of events in the next `days_ahead` days.
        Includes: FOMC, NFP, and a halving-phase summary event.
        """
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        end = as_of + timedelta(days=days_ahead)
        events = []

        # FOMC events
        for fomc in self.FOMC_2026:
            if as_of <= fomc <= end:
                events.append(
                    EconomicEvent(
                        name="FOMC Interest Rate Decision",
                        event_type=EventType.FOMC,
                        impact=EventImpact.CRITICAL,
                        scheduled_at=fomc,
                        symbol_bias={
                            "XAUUSD": "volatile",
                            "BTCUSDT": "volatile",
                            "EURUSD": "volatile",
                        },
                        expected_move_pct=2.5,
                        win_rate=0.55,
                        notes="Reduce risk 1h before and after announcement.",
                    )
                )

        # NFP events
        for nfp in self.get_nfp_dates_2026():
            if as_of <= nfp <= end:
                events.append(
                    EconomicEvent(
                        name="Non-Farm Payrolls",
                        event_type=EventType.NFP,
                        impact=EventImpact.HIGH,
                        scheduled_at=nfp,
                        symbol_bias={
                            "XAUUSD": "volatile",
                            "EURUSD": "volatile",
                        },
                        expected_move_pct=1.5,
                        win_rate=0.52,
                        notes="Avoid new positions 30min before release.",
                    )
                )

        # Halving phase as an ongoing event (represented as a single reference event)
        days_since = self.get_days_since_last_halving(as_of)
        phase_info = self.get_halving_phase(days_since)
        halving_event = EconomicEvent(
            name=f"BTC Halving Phase: {phase_info['phase'].replace('_', ' ').title()}",
            event_type=EventType.HALVING,
            impact=EventImpact.MEDIUM,
            scheduled_at=as_of,  # ongoing — use as_of as reference timestamp
            symbol_bias={"BTCUSDT": phase_info["bias"]},
            expected_move_pct=15.0,
            win_rate=0.65,
            notes=f"Day {days_since} post-halving. Bias: {phase_info['bias']}. PTS: {phase_info['pts']}",
        )
        events.append(halving_event)

        # Sort by scheduled_at
        events.sort(key=lambda e: e.scheduled_at)
        return events

    # ── Analyze event impact ──────────────────────────────────────────────

    def analyze_event_impact(
        self,
        event: EconomicEvent,
        symbol: str,
        as_of: datetime = None,
    ) -> EventSignal:
        """
        Given an event and symbol, determine action, pts_adjustment and risk_multiplier.
        """
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        bias = event.symbol_bias.get(symbol, "neutral")

        # Determine action from bias
        if bias in ("buy", "bullish", "very_bullish"):
            action = "long"
            base_pts = 10
        elif bias in ("sell", "bearish"):
            action = "short"
            base_pts = -10
        elif bias in ("volatile", "reduce_longs"):
            action = "reduce_risk"
            base_pts = -5
        else:
            action = "wait"
            base_pts = 0

        # Risk and urgency by impact level
        if event.impact == EventImpact.CRITICAL:
            risk_multiplier = 0.25
            urgency = "immediate"
            pts_adjustment = base_pts
        elif event.impact == EventImpact.HIGH:
            risk_multiplier = 0.5
            urgency = "prepare"
            pts_adjustment = int(base_pts * 0.75)
        elif event.impact == EventImpact.MEDIUM:
            risk_multiplier = 0.75
            urgency = "monitor"
            pts_adjustment = int(base_pts * 0.5)
        else:
            risk_multiplier = 1.0
            urgency = "monitor"
            pts_adjustment = 0

        # Clamp pts to [-20, +20]
        pts_adjustment = max(-20, min(20, pts_adjustment))

        return EventSignal(
            event=event,
            action=action,
            symbol=symbol,
            pts_adjustment=pts_adjustment,
            urgency=urgency,
            risk_multiplier=risk_multiplier,
        )

    # ── Risk adjustment ───────────────────────────────────────────────────

    def get_risk_adjustment(self, as_of: datetime = None) -> float:
        """
        Returns risk multiplier for current moment:
        - Normal: 1.0
        - Within 1h of FOMC: 0.25
        - Within 30min of NFP: 0.25
        - Day before CPI: 0.5  (CPI not in calendar yet, reserved)
        """
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        if self.is_fomc_window(as_of):
            return 0.25

        if self.is_nfp_window(as_of):
            return 0.25

        return 1.0

    # ── Halving signal ────────────────────────────────────────────────────

    def get_halving_signal_btc(self, as_of: datetime = None) -> EventSignal:
        """Return EventSignal based on current halving phase for BTC."""
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        days_since = self.get_days_since_last_halving(as_of)
        phase_info = self.get_halving_phase(days_since)
        bias = phase_info["bias"]

        event = EconomicEvent(
            name=f"BTC Halving Phase: {phase_info['phase'].replace('_', ' ').title()}",
            event_type=EventType.HALVING,
            impact=EventImpact.MEDIUM,
            scheduled_at=as_of,
            symbol_bias={"BTCUSDT": bias},
            expected_move_pct=15.0,
            win_rate=0.65,
            notes=f"Day {days_since} post-halving.",
        )

        # Determine action from bias string
        if bias in ("bullish", "very_bullish"):
            action = "long"
        elif bias in ("bearish",):
            action = "short"
        elif bias in ("reduce_longs",):
            action = "reduce_risk"
        else:
            action = "wait"

        pts = phase_info["pts"]
        # Clamp to [-20, +20]
        pts = max(-20, min(20, pts))

        return EventSignal(
            event=event,
            action=action,
            symbol="BTCUSDT",
            pts_adjustment=pts,
            urgency="monitor",
            risk_multiplier=1.0,
        )

    # ── Telegram formatter ────────────────────────────────────────────────

    def format_telegram(self, events: list) -> str:
        """HTML format for /eventos Telegram command."""
        if not events:
            return "<b>📅 Próximos eventos</b>\n\nNo hay eventos programados en el período seleccionado."

        impact_emoji = {
            EventImpact.CRITICAL: "🔴",
            EventImpact.HIGH:     "🟠",
            EventImpact.MEDIUM:   "🟡",
            EventImpact.LOW:      "🟢",
        }

        lines = ["<b>📅 Próximos eventos macro</b>\n"]
        for event in events:
            emoji = impact_emoji.get(event.impact, "⚪")
            ts = event.scheduled_at.strftime("%Y-%m-%d %H:%M UTC")
            biases = ", ".join(
                f"{sym}: <i>{b}</i>" for sym, b in event.symbol_bias.items()
            )
            lines.append(
                f"{emoji} <b>{event.name}</b>\n"
                f"   📆 {ts}\n"
                f"   📊 Impacto: {event.impact.value.upper()}\n"
                f"   📈 Sesgo: {biases}\n"
                f"   🎯 Win rate: {event.win_rate:.0%} | Mov. esperado: {event.expected_move_pct:.1f}%\n"
            )
            if event.notes:
                lines.append(f"   💬 {event.notes}\n")
            lines.append("")

        return "\n".join(lines).strip()
