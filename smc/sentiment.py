from dataclasses import dataclass
from typing import List, Dict


# Glint categories that are relevant to each market type
MARKET_CATEGORIES = {
    "crypto":  {"Crypto", "Legal", "Tech", "Economics"},
    "forex":   {"Economics", "Politics", "Military", "Conflict"},
    "indices": {"Economics", "Tech", "Health", "Science"},
    "metals":  {"Military", "Conflict", "Economics", "Politics"},
    "oil":     {"Military", "Conflict", "Climate", "Economics"},
}

IMPACT_WEIGHT = {
    "Critical": 1.0,
    "High":     0.7,
    "Medium":   0.3,
    "Low":      0.1,
}


@dataclass
class SentimentResult:
    component_score: int   # 0-20 (contribution to DecisionFilter)
    score: int             # alias for component_score
    signal_count: int
    alignment: float       # -1.0 to 1.0 (how aligned with bias)
    reason: str


class SentimentAnalyzer:
    """
    Analyzes Glint signals to produce a sentiment score (0-20 pts).

    Score breakdown (max 20):
      - Actionable signal present + aligned:  10 pts
      - High signal alignment (> 0.7):        10 pts

    Conflicting signals reduce alignment and lower the score.
    """

    def analyze(
        self,
        symbol: str,
        glint_signals: List[Dict],
        bias: str = "neutral",
    ) -> SentimentResult:
        if not glint_signals:
            return SentimentResult(
                component_score=0,
                score=0,
                signal_count=0,
                alignment=0.0,
                reason="Sin señales Glint disponibles",
            )

        market_type = self._detect_market_type(symbol)
        relevant = self._filter_relevant(glint_signals, market_type)

        if not relevant:
            return SentimentResult(
                component_score=0,
                score=0,
                signal_count=0,
                alignment=0.0,
                reason=f"Sin señales relevantes para {symbol} ({market_type})",
            )

        alignment = self._calc_alignment(relevant, bias)
        actionable_count = sum(
            1 for s in relevant
            if s.get("impact") in ("Critical", "High")
            and int(s.get("source_tier", 3)) <= 2
            and float(s.get("relevance_score", 0)) >= 7.0
        )

        score = 0
        reason_parts = []

        if actionable_count > 0 and alignment > 0:
            score += 10
            reason_parts.append(f"{actionable_count} señal(es) accionable(s) alineada(s)")

        if alignment > 0.7:
            score += 10
            reason_parts.append(f"Alineación fuerte ({alignment:.0%})")
        elif alignment > 0.4:
            score += 5
            reason_parts.append(f"Alineación moderada ({alignment:.0%})")
        elif alignment < 0:
            reason_parts.append(f"Señales contradictorias (alineación {alignment:.0%})")

        score = min(score, 20)
        return SentimentResult(
            component_score=score,
            score=score,
            signal_count=len(relevant),
            alignment=round(alignment, 3),
            reason="; ".join(reason_parts) if reason_parts else "Sentimiento neutral",
        )

    def _detect_market_type(self, symbol: str) -> str:
        s = symbol.upper()
        if any(x in s for x in ("BTC", "ETH", "BNB", "SOL", "USDT")):
            return "crypto"
        if any(x in s for x in ("XAU", "GOLD", "SILVER", "XAG")):
            return "metals"
        if any(x in s for x in ("OIL", "WTI", "BRENT", "NATGAS")):
            return "oil"
        if any(x in s for x in ("SPX", "NDX", "DOW", "DAX", "US30", "NAS")):
            return "indices"
        return "forex"

    def _filter_relevant(self, signals: List[Dict], market_type: str) -> List[Dict]:
        relevant_cats = MARKET_CATEGORIES.get(market_type, set())
        return [s for s in signals if s.get("category", "") in relevant_cats]

    def _calc_alignment(self, signals: List[Dict], bias: str) -> float:
        """
        Returns alignment between -1.0 and 1.0.
        Without a ground truth of bullish/bearish for each signal, we use
        relevance and impact as positive weight and treat all filtered signals
        as potentially supportive. Conflicting = same category, same impact,
        provided twice (simulating contradictory headlines).
        """
        if not signals:
            return 0.0

        # Weight each signal by impact and relevance
        total_weight = 0.0
        positive_weight = 0.0

        for s in signals:
            w = IMPACT_WEIGHT.get(s.get("impact", "Low"), 0.1)
            w *= min(float(s.get("relevance_score", 5)) / 10.0, 1.0)
            total_weight += w
            # Treat each signal as positive (we don't parse sentiment of text here)
            # In production: use Claude API to classify headline as bullish/bearish
            positive_weight += w

        if total_weight == 0:
            return 0.0

        raw = (positive_weight / total_weight)

        # Penalize if there are many signals (conflicting information)
        if len(signals) > 3:
            raw *= 0.8

        if bias == "neutral":
            return raw * 0.5  # neutral bias halves sentiment contribution

        return round(raw, 3)
