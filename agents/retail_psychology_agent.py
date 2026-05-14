"""
Module 10: Retail Psychology Agent
Identifies psychological price levels, stop-hunt patterns, liquidation zones,
and estimates retail sentiment to generate contrarian trade signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Psychological price levels per instrument
PSYCH_LEVELS_MAP: Dict[str, List[float]] = {
    "BTCUSDT":  [10000 * i for i in range(1, 15)],
    "ETHUSDT":  [500 * i for i in range(1, 15)],
    "EURUSD=X": [round(0.0500 * i, 4) for i in range(20, 30)],   # 1.0000-1.4500
    "GC=F":     [100 * i for i in range(15, 35)],                  # gold levels
    "^DJI":     [5000 * i for i in range(5, 20)],
}

COINGLASS_URL = "https://open-api.coinglass.com/public/v2/liquidation_map"


@dataclass
class PsychLevel:
    price: float
    label: str
    distance_pct: float    # distance from current price as %
    stop_concentration: str  # "high", "medium", "low"
    above_or_below: str    # "above" or "below"


@dataclass
class StopHuntSignal:
    detected: bool
    hunted_level: Optional[float]
    direction: str         # "bull_hunt" or "bear_hunt" or "none"
    confirmation_candles: int
    score_bonus: int       # +15 if detected, 0 otherwise


@dataclass
class LiquidationZone:
    price: float
    side: str              # "long_liquidations" or "short_liquidations"
    estimated_volume: float
    distance_pct: float


@dataclass
class RetailPsychologySignal:
    current_price: float
    nearest_psych_level: Optional[PsychLevel]
    psych_levels_above: List[PsychLevel]
    psych_levels_below: List[PsychLevel]
    stop_hunt: StopHuntSignal
    retail_long_pct: float    # estimated % of retail long (0-100)
    contrarian_bias: str      # opposite of retail majority
    liquidation_zones: List[LiquidationZone]
    nearest_liquidation: Optional[LiquidationZone]
    total_bonus: int          # 0-35
    summary: str


class RetailPsychologyAgent:
    """
    Analyses retail crowd behaviour at psychological price levels
    and generates a contrarian score bonus.
    """

    # ------------------------------------------------------------------
    # Psychological levels
    # ------------------------------------------------------------------

    def _format_label(self, price: float) -> str:
        if price >= 1000:
            return f"${price:,.0f}"
        elif price >= 1:
            return f"{price:.4f}"
        return f"{price:.6f}"

    def _stop_concentration(self, distance_pct: float) -> str:
        if distance_pct <= 0.5:
            return "high"
        elif distance_pct <= 2.0:
            return "medium"
        return "low"

    def get_psychological_levels(
        self,
        symbol: str,
        current_price: float,
        n_above: int = 3,
        n_below: int = 3,
    ) -> Tuple[List[PsychLevel], List[PsychLevel], Optional[PsychLevel]]:
        """
        Returns (levels_above, levels_below, nearest_level).
        Falls back to generic round-number grid if symbol not in PSYCH_LEVELS_MAP.
        """
        raw_levels = PSYCH_LEVELS_MAP.get(symbol)

        if raw_levels is None:
            # Generic fallback: multiples of 1% of current price rounded to 2 sig figs
            magnitude = 10 ** (len(str(int(current_price))) - 2)
            magnitude = max(magnitude, 1)
            base = round(current_price / magnitude) * magnitude
            raw_levels = [base + magnitude * i for i in range(-10, 11) if base + magnitude * i > 0]

        above: List[PsychLevel] = []
        below: List[PsychLevel] = []
        all_levels: List[PsychLevel] = []

        for price in sorted(raw_levels):
            if price <= 0:
                continue
            distance_pct = abs(current_price - price) / price * 100.0
            if price == current_price:
                continue  # exact match: neither strictly above nor below
            above_or_below = "above" if price > current_price else "below"
            conc = self._stop_concentration(distance_pct)
            lvl = PsychLevel(
                price=float(price),
                label=self._format_label(price),
                distance_pct=round(distance_pct, 4),
                stop_concentration=conc,
                above_or_below=above_or_below,
            )
            all_levels.append(lvl)
            if price > current_price:
                above.append(lvl)
            else:
                below.append(lvl)

        # Sort above ascending, below descending (nearest first)
        above_sorted = sorted(above, key=lambda l: l.distance_pct)[:n_above]
        below_sorted = sorted(below, key=lambda l: l.distance_pct)[:n_below]

        nearest: Optional[PsychLevel] = None
        if all_levels:
            nearest = min(all_levels, key=lambda l: l.distance_pct)

        return above_sorted, below_sorted, nearest

    # ------------------------------------------------------------------
    # Stop hunt detection
    # ------------------------------------------------------------------

    def detect_stop_hunt(self, df: pd.DataFrame, symbol: str) -> StopHuntSignal:
        """
        Look at last 3 candles for stop-hunt pattern:
        - Candle N-2 approaches a psych level
        - Candle N-1 wicks past level by >0.3% (sweep)
        - Candle N closes back on the original side
        bull_hunt: swept below level → long signal
        bear_hunt: swept above level → short signal
        """
        no_hunt = StopHuntSignal(
            detected=False, hunted_level=None,
            direction="none", confirmation_candles=0, score_bonus=0,
        )

        if df is None or len(df) < 3:
            return no_hunt

        # Accept both lowercase and OHLC capitalisations
        col_map = {c.lower(): c for c in df.columns}
        high_col = col_map.get("high")
        low_col = col_map.get("low")
        close_col = col_map.get("close")
        if not (high_col and low_col and close_col):
            return no_hunt

        raw_levels = PSYCH_LEVELS_MAP.get(symbol, [])
        if not raw_levels:
            return no_hunt

        last3 = df.iloc[-3:]
        candle_prev2_high = float(last3.iloc[0][high_col])
        candle_prev2_low  = float(last3.iloc[0][low_col])
        candle_sweep_high = float(last3.iloc[1][high_col])
        candle_sweep_low  = float(last3.iloc[1][low_col])
        candle_close_high = float(last3.iloc[2][high_col])
        candle_close_low  = float(last3.iloc[2][low_col])
        candle_close      = float(last3.iloc[2][close_col])

        for level in raw_levels:
            threshold = level * 0.003  # 0.3%

            # Bull hunt: wick swept BELOW level, close recovered ABOVE level
            if (
                candle_sweep_low < level - threshold
                and candle_close > level
            ):
                return StopHuntSignal(
                    detected=True,
                    hunted_level=float(level),
                    direction="bull_hunt",
                    confirmation_candles=1,
                    score_bonus=15,
                )

            # Bear hunt: wick swept ABOVE level, close fell BELOW level
            if (
                candle_sweep_high > level + threshold
                and candle_close < level
            ):
                return StopHuntSignal(
                    detected=True,
                    hunted_level=float(level),
                    direction="bear_hunt",
                    confirmation_candles=1,
                    score_bonus=15,
                )

        return no_hunt

    # ------------------------------------------------------------------
    # Liquidation zones
    # ------------------------------------------------------------------

    def get_liquidation_zones(
        self, symbol: str, current_price: float
    ) -> List[LiquidationZone]:
        """
        Attempt Coinglass API; fall back to synthetic ±5%, ±10%, ±15% zones.
        """
        try:
            import requests
            resp = requests.get(
                COINGLASS_URL,
                params={"symbol": symbol},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            zones: List[LiquidationZone] = []
            for item in data.get("data", []):
                price = float(item.get("price", 0))
                side  = item.get("side", "long_liquidations")
                vol   = float(item.get("volume", 0))
                dist  = abs(current_price - price) / price * 100.0
                zones.append(LiquidationZone(
                    price=price, side=side, estimated_volume=vol,
                    distance_pct=round(dist, 4),
                ))
            if zones:
                return zones
        except Exception as exc:
            logger.debug("Coinglass unavailable: %s — using synthetic zones", exc)

        # Synthetic fallback
        synthetic: List[LiquidationZone] = []
        for pct in (5.0, 10.0, 15.0):
            above_price = current_price * (1 + pct / 100)
            below_price = current_price * (1 - pct / 100)
            # Longs get liquidated below price, shorts above
            synthetic.append(LiquidationZone(
                price=round(below_price, 4),
                side="long_liquidations",
                estimated_volume=0.0,
                distance_pct=pct,
            ))
            synthetic.append(LiquidationZone(
                price=round(above_price, 4),
                side="short_liquidations",
                estimated_volume=0.0,
                distance_pct=pct,
            ))
        return synthetic

    # ------------------------------------------------------------------
    # Retail sentiment
    # ------------------------------------------------------------------

    def estimate_retail_sentiment(self, df: pd.DataFrame) -> float:
        """
        Estimate retail long % from 20-bar price momentum.
        Strong uptrend → 70-80% long (fade)
        Strong downtrend → 20-30% long (fade)
        Sideways → 50% (neutral)
        """
        if df is None or len(df) < 20:
            return 50.0

        col = "close" if "close" in df.columns else df.columns[-1]
        prices = df[col].values.astype(float)
        last20 = prices[-20:]

        pct_change = (last20[-1] - last20[0]) / last20[0] * 100.0

        if pct_change > 3.0:       # strong uptrend
            # Retail piling in long – random between 70 and 80
            return float(np.clip(70.0 + abs(pct_change) * 0.5, 70.0, 80.0))
        elif pct_change < -3.0:    # strong downtrend
            return float(np.clip(30.0 - abs(pct_change) * 0.5, 20.0, 30.0))
        else:                       # sideways
            return 50.0

    # ------------------------------------------------------------------
    # Combined signal
    # ------------------------------------------------------------------

    def get_signal(self, symbol: str, df: pd.DataFrame) -> RetailPsychologySignal:
        """Build a full RetailPsychologySignal."""
        col = "close" if "close" in df.columns else df.columns[-1]
        current_price = float(df[col].iloc[-1])

        levels_above, levels_below, nearest = self.get_psychological_levels(
            symbol, current_price
        )
        stop_hunt = self.detect_stop_hunt(df, symbol)
        liq_zones = self.get_liquidation_zones(symbol, current_price)

        nearest_liq: Optional[LiquidationZone] = (
            min(liq_zones, key=lambda z: z.distance_pct) if liq_zones else None
        )

        retail_long_pct = self.estimate_retail_sentiment(df)
        if retail_long_pct > 60:
            contrarian_bias = "bearish"   # fade the longs
        elif retail_long_pct < 40:
            contrarian_bias = "bullish"   # fade the shorts
        else:
            contrarian_bias = "neutral"

        # Score
        stop_bonus = stop_hunt.score_bonus
        psych_bonus = 10 if (nearest and nearest.distance_pct <= 0.5) else 0
        contrarian_bonus = 10 if contrarian_bias != "neutral" else 0
        total_bonus = int(np.clip(stop_bonus + psych_bonus + contrarian_bonus, 0, 35))

        parts = [
            f"Price={current_price:.4f}",
            f"RetailLong={retail_long_pct:.0f}% (fade→{contrarian_bias})",
            f"StopHunt={'YES ' + stop_hunt.direction if stop_hunt.detected else 'no'}",
            f"NearestPsych={nearest.label if nearest else 'N/A'}",
            f"Bonus=+{total_bonus}",
        ]
        summary = " | ".join(parts)

        return RetailPsychologySignal(
            current_price=current_price,
            nearest_psych_level=nearest,
            psych_levels_above=levels_above,
            psych_levels_below=levels_below,
            stop_hunt=stop_hunt,
            retail_long_pct=retail_long_pct,
            contrarian_bias=contrarian_bias,
            liquidation_zones=liq_zones,
            nearest_liquidation=nearest_liq,
            total_bonus=total_bonus,
            summary=summary,
        )

    def score_adjustment(self, symbol: str, df: pd.DataFrame, bias: str) -> int:
        """Return 0-35 score bonus aligned with *bias* direction."""
        sig = self.get_signal(symbol, df)
        # Only award full bonus when contrarian bias matches the requested bias
        if sig.contrarian_bias == bias or sig.contrarian_bias == "neutral":
            return sig.total_bonus
        # Partial: still credit stop-hunt if direction matches bias
        return sig.stop_hunt.score_bonus

    # ------------------------------------------------------------------
    # Telegram formatting
    # ------------------------------------------------------------------

    def format_telegram(self, symbol: str, df: pd.DataFrame) -> str:
        """Return a Telegram-ready retail psychology analysis string."""
        sig = self.get_signal(symbol, df)

        lines = [
            f"*Retail Psychology — {symbol}*",
            f"Price: `{sig.current_price:.4f}`",
            f"Retail long: `{sig.retail_long_pct:.0f}%` → contrarian: {sig.contrarian_bias.upper()}",
        ]

        if sig.nearest_psych_level:
            lvl = sig.nearest_psych_level
            lines.append(
                f"Nearest psych level: `{lvl.label}` "
                f"({lvl.distance_pct:.2f}% {lvl.above_or_below}, "
                f"stops={lvl.stop_concentration})"
            )

        if sig.psych_levels_above:
            above_labels = ", ".join(l.label for l in sig.psych_levels_above)
            lines.append(f"Resistance levels: `{above_labels}`")
        if sig.psych_levels_below:
            below_labels = ", ".join(l.label for l in sig.psych_levels_below)
            lines.append(f"Support levels: `{below_labels}`")

        sh = sig.stop_hunt
        if sh.detected:
            lines.append(
                f"Stop hunt: {sh.direction.upper()} at `{sh.hunted_level}` "
                f"(+{sh.score_bonus}pt)"
            )
        else:
            lines.append("Stop hunt: none detected")

        if sig.nearest_liquidation:
            lz = sig.nearest_liquidation
            lines.append(
                f"Nearest liq zone: `{lz.price:.2f}` "
                f"({lz.side}, {lz.distance_pct:.1f}% away)"
            )

        lines.append(f"Psych bonus: `+{sig.total_bonus}/35`")
        return "\n".join(lines)
