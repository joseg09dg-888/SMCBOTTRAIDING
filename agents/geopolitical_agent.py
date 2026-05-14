from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

try:
    import httpx
except ImportError:
    httpx = None

# Market impact by event category
CATEGORY_IMPACT: Dict[str, Dict[str, str]] = {
    "conflict":  {"XAUUSD": "bullish", "USOIL": "bullish", "USDJPY": "bullish", "BTCUSDT": "bullish"},
    "sanctions": {"BTCUSDT": "bullish", "EURUSD=X": "bearish"},
    "election":  {"EURUSD=X": "volatile", "GBPUSD=X": "volatile"},
    "disaster":  {"USOIL": "bullish", "NATGAS": "bullish"},
}

# Risk score thresholds
RISK_LABELS = {
    (1, 2): "low",
    (3, 4): "moderate",
    (5, 6): "elevated",
    (7, 8): "high",
    (9, 10): "critical",
}


def _risk_label(score: int) -> str:
    for (lo, hi), label in RISK_LABELS.items():
        if lo <= score <= hi:
            return label
    return "moderate"


@dataclass
class GeopoliticalEvent:
    title: str
    category: str          # "conflict", "sanctions", "election", "disaster"
    severity: int          # 1-10
    affected_markets: List[str]   # e.g. ["XAUUSD", "USOIL"]
    market_bias: Dict[str, str]   # e.g. {"XAUUSD": "bullish"}
    source: str
    timestamp: str


@dataclass
class GeopoliticalSignal:
    risk_score: int         # 1-10 global risk
    risk_label: str         # "low", "moderate", "elevated", "high", "critical"
    recent_events: List[GeopoliticalEvent]
    market_impact: Dict[str, str]  # {"XAUUSD": "bullish", ...}
    score_bonus: int        # -15 to +10
    trade_blocked: bool     # True if risk_score > 7
    summary: str


class GeopoliticalAgent:
    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch_events(self, max_events: int = 5) -> List[GeopoliticalEvent]:
        """
        Try GDELT API; return [] if network unavailable.
        Parses response for high-severity events (severity >= 6 treated as high).
        """
        try:
            if httpx is None:
                return []
            params = {
                "query": "war OR sanctions OR conflict OR disaster",
                "mode": "ArtList",
                "maxrecords": max_events,
                "format": "json",
                "sort": "DateDesc",
            }
            response = httpx.get(self.GDELT_URL, params=params, timeout=5.0)
            response.raise_for_status()
            data = response.json()

            events: List[GeopoliticalEvent] = []
            articles = data.get("articles", [])
            for article in articles[:max_events]:
                title = article.get("title", "Unknown event")
                category = self._classify_category(title)
                severity = self._estimate_severity(title, category)
                impact_map = CATEGORY_IMPACT.get(category, {})
                affected = list(impact_map.keys())
                events.append(GeopoliticalEvent(
                    title=title,
                    category=category,
                    severity=severity,
                    affected_markets=affected,
                    market_bias=dict(impact_map),
                    source="GDELT",
                    timestamp=article.get("seendate", datetime.utcnow().isoformat()),
                ))
            return events
        except Exception:
            return []

    def _classify_category(self, title: str) -> str:
        title_lower = title.lower()
        if any(w in title_lower for w in ("war", "conflict", "attack", "military", "strike")):
            return "conflict"
        if any(w in title_lower for w in ("sanction", "embargo")):
            return "sanctions"
        if any(w in title_lower for w in ("election", "vote", "referendum")):
            return "election"
        if any(w in title_lower for w in ("earthquake", "hurricane", "flood", "disaster", "storm")):
            return "disaster"
        return "conflict"  # default

    def _estimate_severity(self, title: str, category: str) -> int:
        """Heuristic severity based on keywords and category."""
        title_lower = title.lower()
        base = {"conflict": 7, "sanctions": 5, "election": 4, "disaster": 6}.get(category, 5)
        if any(w in title_lower for w in ("nuclear", "world war", "global")):
            base = min(base + 2, 10)
        if any(w in title_lower for w in ("minor", "local", "small")):
            base = max(base - 2, 1)
        return base

    def calculate_risk_score(self, events: List[GeopoliticalEvent]) -> int:
        """
        Average severity of recent events, capped 1-10.
        No events → risk_score = 3 (baseline).
        """
        if not events:
            return 3
        avg = sum(e.severity for e in events) / len(events)
        return max(1, min(10, round(avg)))

    def get_market_impact(self, symbol: str, events: List[GeopoliticalEvent]) -> str:
        """
        Returns "bullish", "bearish", "neutral", or "volatile" for the symbol
        based on active events.
        """
        votes: Dict[str, int] = {}
        for event in events:
            bias = event.market_bias.get(symbol)
            if bias:
                votes[bias] = votes.get(bias, 0) + event.severity

        if not votes:
            return "neutral"

        # Return the bias with the highest weighted score
        return max(votes, key=lambda k: votes[k])

    def get_signal(self, symbol: str = "") -> GeopoliticalSignal:
        """Returns full geopolitical signal; handles empty events list gracefully."""
        events = self.fetch_events()
        risk_score = self.calculate_risk_score(events)
        label = _risk_label(risk_score)
        trade_blocked = risk_score > 7

        # Build market impact map across all known symbols
        all_symbols = set()
        for cat_map in CATEGORY_IMPACT.values():
            all_symbols.update(cat_map.keys())
        if symbol:
            all_symbols.add(symbol)

        market_impact: Dict[str, str] = {}
        for sym in all_symbols:
            market_impact[sym] = self.get_market_impact(sym, events)

        # score_bonus: +10 if geopolitical favors direction, -15 if risk > 7
        score_bonus = self.score_adjustment(symbol, "bullish") if symbol else 0
        if trade_blocked:
            score_bonus = -15

        event_titles = [e.title[:60] for e in events[:3]]
        events_str = "; ".join(event_titles) if event_titles else "No significant events detected"

        summary = (
            f"Geopolitical Risk: {label.upper()} ({risk_score}/10) | "
            f"Trade blocked: {'YES' if trade_blocked else 'NO'} | "
            f"Recent: {events_str}"
        )

        return GeopoliticalSignal(
            risk_score=risk_score,
            risk_label=label,
            recent_events=events,
            market_impact=market_impact,
            score_bonus=score_bonus,
            trade_blocked=trade_blocked,
            summary=summary,
        )

    def score_adjustment(self, symbol: str, bias: str) -> int:
        """
        +10 if geopolitical signal favors the trade direction for this symbol.
        -15 if risk_score > 7 (high/critical environment).
        Returns int.
        """
        events = self.fetch_events()
        risk_score = self.calculate_risk_score(events)

        if risk_score > 7:
            return -15

        if symbol:
            impact = self.get_market_impact(symbol, events)
            if impact == bias:
                return 10

        return 0

    def format_telegram(self, symbol: str = "") -> str:
        """Structured Telegram output with geopolitical risk metrics."""
        signal = self.get_signal(symbol)

        risk_emoji = {
            "low": "🟢",
            "moderate": "🟡",
            "elevated": "🟠",
            "high": "🔴",
            "critical": "🚨",
        }.get(signal.risk_label, "⚪")

        event_lines = []
        for event in signal.recent_events[:3]:
            event_lines.append(
                f"  [{event.category.upper()}] {event.title[:70]} (severity: {event.severity}/10)"
            )
        events_section = "\n".join(event_lines) if event_lines else "  No significant events"

        impact_line = ""
        if symbol and symbol in signal.market_impact:
            impact_line = f"\nImpact on {symbol}: {signal.market_impact[symbol].upper()}"

        blocked_line = "TRADE BLOCKED (high geopolitical risk)" if signal.trade_blocked else "Trading permitted"

        lines = [
            f"GEOPOLITICAL SIGNAL",
            "",
            f"Risk Score: {signal.risk_score}/10 {risk_emoji} — {signal.risk_label.upper()}",
            f"Status: {blocked_line}",
            f"Score Bonus: {signal.score_bonus:+d}",
            impact_line,
            "",
            "Recent Events:",
            events_section,
            "",
            signal.summary,
        ]
        return "\n".join(line for line in lines)
