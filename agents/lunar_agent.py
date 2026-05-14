"""LunarCycleAgent — lunar phase bias for trading decisions."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class LunarSignal:
    phase_pct: float        # 0.0=new moon, 0.5=full moon
    phase_name: str         # "nueva", "creciente", "llena", "menguante"
    bias: str               # "bullish", "bearish", "neutral"
    score_bonus: int        # 0-5
    eclipse_warning: bool
    summary: str


class LunarCycleAgent:
    """
    Calculates lunar phase and derives a trading bias.
    Uses ephem if available; degrades to neutral if not installed.
    """

    # Synodic month in days
    SYNODIC_MONTH = 29.53058868

    # Known new moon reference (J2000 epoch)
    _REF_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)

    def _phase_pct(self, as_of: Optional[datetime] = None) -> float:
        """Returns 0.0–1.0 phase fraction using ephem when available."""
        try:
            import ephem
            moon = ephem.Moon(as_of or datetime.now(timezone.utc))
            return float(moon.phase) / 100.0
        except Exception:
            pass
        # Fallback: simple synodic calculation
        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = (now - self._REF_NEW_MOON).total_seconds()
        cycle   = self.SYNODIC_MONTH * 86400
        return (elapsed % cycle) / cycle

    def get_current_phase(self, as_of: Optional[datetime] = None) -> LunarSignal:
        pct = self._phase_pct(as_of)

        if pct < 0.125 or pct > 0.875:
            name, bias, bonus = "nueva",     "bearish", 5
        elif pct < 0.375:
            name, bias, bonus = "creciente", "bullish", 5
        elif pct < 0.625:
            name, bias, bonus = "llena",     "bullish", 5
        else:
            name, bias, bonus = "menguante", "bearish", 5

        eclipse = self._check_eclipse(as_of)

        return LunarSignal(
            phase_pct=round(pct, 3),
            phase_name=name,
            bias=bias,
            score_bonus=bonus,
            eclipse_warning=eclipse,
            summary=(
                f"Luna {name} ({pct*100:.0f}%) — sesgo {bias}"
                + (" ⚠️ eclipse próximo" if eclipse else "")
            ),
        )

    def _check_eclipse(self, as_of: Optional[datetime] = None) -> bool:
        """Returns True if solar or lunar eclipse within 15 days."""
        try:
            import ephem
            now = as_of or datetime.now(timezone.utc)
            next_solar  = ephem.next_solar_eclipse(now)
            next_lunar  = ephem.next_lunar_eclipse(now)
            for ev in (next_solar, next_lunar):
                ev_dt = ephem.Date(ev).datetime().replace(tzinfo=timezone.utc)
                if abs((ev_dt - now).days) <= 15:
                    return True
        except Exception:
            pass
        return False

    def score_adjustment(self, trade_bias: str,
                          as_of: Optional[datetime] = None) -> int:
        """Returns +5 if lunar bias matches trade_bias, 0 otherwise."""
        signal = self.get_current_phase(as_of)
        return signal.score_bonus if signal.bias == trade_bias else 0

    def format_telegram(self, as_of: Optional[datetime] = None) -> str:
        s = self.get_current_phase(as_of)
        emoji = "🌕" if s.phase_name == "llena" else "🌑" if s.phase_name == "nueva" else "🌙"
        return (
            f"{emoji} *Luna — {s.phase_name.upper()}*\n"
            f"Fase: {s.phase_pct*100:.0f}% | Sesgo: {s.bias}\n"
            f"Bonus DecisionFilter: +{s.score_bonus} pts\n"
            + ("⚠️ Eclipse próximo — volatilidad elevada\n" if s.eclipse_warning else "")
            + s.summary
        )
