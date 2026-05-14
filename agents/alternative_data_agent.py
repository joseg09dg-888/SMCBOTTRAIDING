"""
Module 4: Alternative Data Agent
Fetches Fear & Greed Index and Google Trends to generate contrarian signals.
All network calls degrade gracefully when offline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrendsSignal:
    keyword: str
    current_value: int      # 0-100 (Google Trends scale)
    peak_value: int         # max in last 12 months
    pct_of_peak: float
    interpretation: str     # "euphoria" / "fear" / "neutral"
    score_bonus: int        # 0-5


@dataclass
class SentimentScore:
    source: str             # "google_trends", "app_rankings", "fear_greed"
    score: float            # -1.0 to +1.0
    label: str              # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
    contrarian_bias: str    # opposite of score for contrarian signal
    score_bonus: int


@dataclass
class AlternativeDataSignal:
    fear_greed_index: int           # 0-100 from alternative.me
    fear_greed_label: str
    google_trends_signals: List[TrendsSignal]
    overall_sentiment: float        # -1.0 to +1.0
    contrarian_bias: str            # trade against the crowd
    total_bonus: int                # 0-15
    summary: str


# ---------------------------------------------------------------------------
# Helper: label & contrarian from Fear & Greed value
# ---------------------------------------------------------------------------

def _fg_label(value: int) -> str:
    if value <= 25:
        return "extreme_fear"
    elif value <= 45:
        return "fear"
    elif value <= 55:
        return "neutral"
    elif value <= 75:
        return "greed"
    return "extreme_greed"


def _contrarian_bias(label: str) -> str:
    """Return the contrarian trade direction for a given crowd label."""
    mapping = {
        "extreme_fear": "bullish",
        "fear": "bullish",
        "neutral": "neutral",
        "greed": "bearish",
        "extreme_greed": "bearish",
    }
    return mapping.get(label, "neutral")


def _sentiment_score(value: int) -> float:
    """Normalise 0-100 F&G index to -1.0 … +1.0 (crowd mood, not contrarian)."""
    return round((value - 50) / 50.0, 4)


class AlternativeDataAgent:
    """
    Aggregates Fear & Greed Index and Google Trends to produce
    a contrarian score bonus (0-15 pts).
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

    def __init__(self):
        self._fg_cache: Optional[dict] = None
        self._trends_cache: List[TrendsSignal] = []

    # ------------------------------------------------------------------
    # Fear & Greed Index
    # ------------------------------------------------------------------

    def get_fear_greed(self) -> Optional[dict]:
        """
        Fetch the latest Fear & Greed index from alternative.me.
        Returns the raw API dict, or None if unavailable.
        """
        try:
            import requests
            resp = requests.get(self.FEAR_GREED_URL, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            entry = data["data"][0]
            result = {
                "value": int(entry["value"]),
                "value_classification": entry.get("value_classification", ""),
                "timestamp": entry.get("timestamp", ""),
            }
            self._fg_cache = result
            return result
        except Exception as exc:
            logger.warning("Fear & Greed fetch failed: %s — using cache", exc)
            return self._fg_cache

    # ------------------------------------------------------------------
    # Google Trends
    # ------------------------------------------------------------------

    def get_google_trends(self, keywords: List[str]) -> List[TrendsSignal]:
        """
        Use pytrends to fetch 12-month interest data for *keywords*.
        Returns [] if pytrends is not installed or network fails.
        """
        try:
            from pytrends.request import TrendReq  # optional dependency
        except ImportError:
            logger.info("pytrends not installed — skipping Google Trends")
            return []

        signals: List[TrendsSignal] = []
        try:
            pt = TrendReq(hl="en-US", tz=360)
            pt.build_payload(keywords, timeframe="today 12-m")
            df = pt.interest_over_time()

            for kw in keywords:
                if kw not in df.columns:
                    continue
                series = df[kw]
                current = int(series.iloc[-1])
                peak = int(series.max()) if len(series) > 0 else 1
                peak = peak if peak > 0 else 1
                pct = current / peak

                if pct >= 0.90:
                    interpretation = "euphoria"
                    bonus = 5
                elif pct <= 0.20:
                    interpretation = "fear"
                    bonus = 5
                else:
                    interpretation = "neutral"
                    bonus = 0

                signals.append(
                    TrendsSignal(
                        keyword=kw,
                        current_value=current,
                        peak_value=peak,
                        pct_of_peak=round(pct, 4),
                        interpretation=interpretation,
                        score_bonus=bonus,
                    )
                )

            self._trends_cache = signals
        except Exception as exc:
            logger.warning("Google Trends fetch failed: %s", exc)

        return signals

    # ------------------------------------------------------------------
    # Combined signal
    # ------------------------------------------------------------------

    def get_combined_signal(self, symbol: str, bias: str) -> AlternativeDataSignal:
        """
        Combine F&G + Google Trends into a contrarian AlternativeDataSignal.
        *bias* is the trade direction ("bullish"/"bearish") from the strategy.
        """
        fg_data = self.get_fear_greed()
        fg_value = fg_data["value"] if fg_data else 50
        fg_label = _fg_label(fg_value)
        contrarian = _contrarian_bias(fg_label)
        overall = _sentiment_score(fg_value)

        # Score: contrarian bonus when crowd is extreme AND agrees with trade bias
        fg_bonus = 0
        if fg_label == "extreme_fear" and bias == "bullish":
            fg_bonus = 15
        elif fg_label == "extreme_greed" and bias == "bearish":
            fg_bonus = 15
        elif fg_label in ("fear",) and bias == "bullish":
            fg_bonus = 7
        elif fg_label in ("greed",) and bias == "bearish":
            fg_bonus = 7

        # Google Trends bonus (up to 5 extra; not counted in base 15 cap for FG)
        keywords = [f"buy {symbol.lower()}", symbol.lower()]
        trends = self.get_google_trends(keywords)
        trends_bonus = sum(t.score_bonus for t in trends)
        # Contrarian: euphoria on trends → bad for bulls; fear on trends → good for bulls
        trends_contrarian_bonus = 0
        for t in trends:
            if t.interpretation == "euphoria" and bias == "bearish":
                trends_contrarian_bonus += t.score_bonus
            elif t.interpretation == "fear" and bias == "bullish":
                trends_contrarian_bonus += t.score_bonus

        total = min(fg_bonus + trends_contrarian_bonus, 15)

        summary_parts = [
            f"F&G={fg_value} ({fg_label}) → contrarian {contrarian}"
        ]
        if trends:
            summary_parts.append(
                f"Trends: {', '.join(t.keyword + '=' + t.interpretation for t in trends)}"
            )
        summary = " | ".join(summary_parts) + f" → bonus +{total}"

        return AlternativeDataSignal(
            fear_greed_index=fg_value,
            fear_greed_label=fg_label,
            google_trends_signals=trends,
            overall_sentiment=overall,
            contrarian_bias=contrarian,
            total_bonus=total,
            summary=summary,
        )

    def score_adjustment(self, symbol: str, bias: str) -> int:
        """Return the 0-15 contrarian score bonus for *symbol* given *bias*."""
        signal = self.get_combined_signal(symbol, bias)
        return signal.total_bonus

    # ------------------------------------------------------------------
    # Telegram formatting
    # ------------------------------------------------------------------

    def format_telegram(self, symbol: str) -> str:
        """Return a human-readable Telegram string with F&G and trends data."""
        fg = self._fg_cache
        fg_value = fg["value"] if fg else "N/A"
        fg_label = _fg_label(fg["value"]) if fg else "unknown"

        lines = [
            f"📊 *Alternative Data — {symbol}*",
            f"Fear & Greed: `{fg_value}` — {fg_label.replace('_', ' ').title()}",
        ]

        if self._trends_cache:
            for t in self._trends_cache:
                lines.append(
                    f"Trends '{t.keyword}': `{t.current_value}` "
                    f"({t.pct_of_peak*100:.0f}% of peak) → {t.interpretation}"
                )
        else:
            lines.append("Google Trends: no data")

        contrarian = _contrarian_bias(fg_label) if fg else "neutral"
        lines.append(f"Contrarian bias: *{contrarian.upper()}*")
        return "\n".join(lines)
