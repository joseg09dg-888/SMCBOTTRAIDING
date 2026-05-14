from datetime import datetime, date, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class FOMCEvent:
    date: str              # "2025-01-29"
    type: str              # "meeting", "speech", "minutes"
    description: str
    days_until: int        # days from today


@dataclass
class FEDSentimentResult:
    hawkish_score: int     # 1-10
    dovish_score: int      # 1-10
    net_sentiment: float   # hawkish_score - dovish_score, -9 to +9
    rate_hike_probability: float  # 0.0 to 1.0
    usd_bias: str          # "bullish", "bearish", "neutral"
    gold_bias: str         # opposite of USD
    crypto_bias: str
    score_bonus: int       # -20 to +10
    fomc_blackout: bool    # True if meeting within 24 hours
    next_fomc: Optional[FOMCEvent]
    summary: str


# Hardcoded 2025-2026 FOMC dates
FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
]
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]

ALL_FOMC_DATES = FOMC_DATES_2025 + FOMC_DATES_2026

# Keyword lists for sentiment analysis
HAWKISH_KEYWORDS = [
    "inflation", "hike", "restrictive", "tighten", "tightening",
    "rate increase", "hawkish", "higher for longer", "overshoot",
    "above target", "overheating",
]

DOVISH_KEYWORDS = [
    "cut", "easing", "support", "accommodative", "dovish",
    "recession", "slowdown", "below target", "pause", "pivot",
    "lower rates", "rate cut",
]


class FEDSentimentAgent:
    """
    Tracks FED/FOMC sentiment and its expected impact on USD, gold, and crypto.
    Uses keyword-based analysis without requiring an external API.
    """

    def __init__(self):
        self._cached_sentiment: Optional[FEDSentimentResult] = None

    def get_next_fomc(self, as_of: date = None) -> Optional[FOMCEvent]:
        """Return the next FOMC meeting date from today (or as_of date)."""
        if as_of is None:
            as_of = date.today()

        for date_str in ALL_FOMC_DATES:
            fomc_date = date.fromisoformat(date_str)
            if fomc_date >= as_of:
                days_until = (fomc_date - as_of).days
                return FOMCEvent(
                    date=date_str,
                    type="meeting",
                    description=f"FOMC Meeting — {date_str}",
                    days_until=days_until,
                )
        return None

    def is_fomc_blackout(self, as_of: date = None) -> bool:
        """True if an FOMC meeting is within 24 hours (0 or 1 day away)."""
        if as_of is None:
            as_of = date.today()

        next_fomc = self.get_next_fomc(as_of)
        if next_fomc is None:
            return False
        return next_fomc.days_until <= 1

    def analyze_sentiment(self, text: str) -> FEDSentimentResult:
        """
        Basic keyword analysis to determine FED sentiment.
        Counts hawkish vs dovish keywords and derives USD/gold/crypto bias.
        """
        text_lower = text.lower()

        hawkish_count = sum(1 for kw in HAWKISH_KEYWORDS if kw in text_lower)
        dovish_count = sum(1 for kw in DOVISH_KEYWORDS if kw in text_lower)

        # Score on 1-10 scale
        hawkish_score = min(10, max(1, 1 + hawkish_count * 2))
        dovish_score = min(10, max(1, 1 + dovish_count * 2))

        net_sentiment = float(hawkish_score - dovish_score)

        # Rate hike probability: higher when hawkish
        rate_hike_probability = min(1.0, max(0.0, 0.5 + net_sentiment * 0.05))

        # Bias determination
        if net_sentiment > 1:
            usd_bias = "bullish"
            gold_bias = "bearish"
            crypto_bias = "bearish"
        elif net_sentiment < -1:
            usd_bias = "bearish"
            gold_bias = "bullish"
            crypto_bias = "bullish"
        else:
            usd_bias = "neutral"
            gold_bias = "neutral"
            crypto_bias = "neutral"

        # Score bonus: hawkish → USD up, gold/crypto down; dovish → opposite
        if net_sentiment > 1:
            score_bonus = min(10, int(net_sentiment * 2))
        elif net_sentiment < -1:
            score_bonus = max(-20, int(net_sentiment * 2))
        else:
            score_bonus = 0

        # FOMC blackout check
        fomc_blackout = self.is_fomc_blackout()
        if fomc_blackout:
            score_bonus = -20

        next_fomc = self.get_next_fomc()

        summary = (
            f"FED Sentiment — Hawkish: {hawkish_score}/10 | Dovish: {dovish_score}/10 | "
            f"Net: {net_sentiment:+.1f} | USD: {usd_bias.upper()} | "
            f"Gold: {gold_bias.upper()} | Crypto: {crypto_bias.upper()}"
        )

        result = FEDSentimentResult(
            hawkish_score=hawkish_score,
            dovish_score=dovish_score,
            net_sentiment=net_sentiment,
            rate_hike_probability=rate_hike_probability,
            usd_bias=usd_bias,
            gold_bias=gold_bias,
            crypto_bias=crypto_bias,
            score_bonus=score_bonus,
            fomc_blackout=fomc_blackout,
            next_fomc=next_fomc,
            summary=summary,
        )

        # Cache the result
        self._cached_sentiment = result
        return result

    def get_cached_sentiment(self) -> FEDSentimentResult:
        """Returns last known sentiment. If no analysis has been run, returns neutral."""
        if self._cached_sentiment is not None:
            return self._cached_sentiment

        # Default neutral sentiment
        next_fomc = self.get_next_fomc()
        fomc_blackout = self.is_fomc_blackout()
        score_bonus = -20 if fomc_blackout else 0

        return FEDSentimentResult(
            hawkish_score=5,
            dovish_score=5,
            net_sentiment=0.0,
            rate_hike_probability=0.5,
            usd_bias="neutral",
            gold_bias="neutral",
            crypto_bias="neutral",
            score_bonus=score_bonus,
            fomc_blackout=fomc_blackout,
            next_fomc=next_fomc,
            summary="FED Sentiment — Neutral (no analysis available)",
        )

    def score_adjustment(
        self,
        trade_symbol: str,
        bias: str,
        as_of: date = None,
    ) -> int:
        """
        Returns score adjustment based on FED sentiment alignment with trade bias.

        Rules:
        - -20 if FOMC blackout (meeting within 24 hours)
        - +10 if FED sentiment aligns with trade direction:
          * USD symbols: hawkish FED → USD bullish
          * Gold/Crypto: dovish FED → bullish
        - 0 otherwise
        """
        if as_of is None:
            as_of = date.today()

        if self.is_fomc_blackout(as_of):
            return -20

        sentiment = self.get_cached_sentiment()

        # Determine expected bias from FED stance
        symbol_upper = trade_symbol.upper()
        is_gold = "GC" in symbol_upper or "XAU" in symbol_upper or "GOLD" in symbol_upper
        is_crypto = any(
            c in symbol_upper for c in ["BTC", "ETH", "USDT", "BNB", "SOL"]
        )
        is_usd = "USD" in symbol_upper and not is_crypto

        if is_usd:
            fed_bias = sentiment.usd_bias
        elif is_gold:
            fed_bias = sentiment.gold_bias
        elif is_crypto:
            fed_bias = sentiment.crypto_bias
        else:
            fed_bias = "neutral"

        if fed_bias != "neutral" and fed_bias == bias:
            return 10

        return 0

    def format_telegram(self) -> str:
        """Format FED sentiment as a Telegram-friendly message."""
        sentiment = self.get_cached_sentiment()

        bias_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}

        lines = [
            "*FED Sentiment Analysis*",
            f"Hawkish score: `{sentiment.hawkish_score}/10`",
            f"Dovish score: `{sentiment.dovish_score}/10`",
            f"Net sentiment: `{sentiment.net_sentiment:+.1f}`",
            f"Rate hike probability: `{sentiment.rate_hike_probability:.0%}`",
            f"USD bias: {bias_emoji.get(sentiment.usd_bias, '⚪')} `{sentiment.usd_bias.upper()}`",
            f"Gold bias: {bias_emoji.get(sentiment.gold_bias, '⚪')} `{sentiment.gold_bias.upper()}`",
            f"Crypto bias: {bias_emoji.get(sentiment.crypto_bias, '⚪')} `{sentiment.crypto_bias.upper()}`",
        ]

        if sentiment.fomc_blackout:
            lines.append("⛔ *FOMC BLACKOUT — No trading recommended*")
        elif sentiment.next_fomc:
            lines.append(
                f"Next FOMC: `{sentiment.next_fomc.date}` "
                f"({sentiment.next_fomc.days_until} days away)"
            )

        lines.append(f"Score bonus: `{sentiment.score_bonus:+d}`")

        return "\n".join(lines)
