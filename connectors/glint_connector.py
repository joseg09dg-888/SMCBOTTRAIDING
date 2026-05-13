import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, Any

POLLING_INTERVAL    = 15   # seconds between polls when connected
RETRY_INTERVAL      = 300  # 5 minutes before retrying after full failure
MAX_SEEN_IDS        = 1000 # prevent unbounded memory growth

CATEGORY_INSTRUMENTS = {
    "Crypto":        ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    "Economics":     ["XAUUSD", "EURUSD", "US30", "SPX500"],
    "Politics":      ["XAUUSD", "EURUSD", "USDJPY"],
    "Military":      ["XAUUSD", "USOIL", "USDJPY"],
    "Tech":          ["NASDAQ", "TSLA", "AAPL", "NVDA"],
    "Conflict":      ["XAUUSD", "USOIL", "USDJPY"],
    "Climate":       ["USOIL", "NATGAS", "WHEAT"],
    "Health":        ["PHARMA", "SPX500"],
    "Legal":         ["BTC/USDT", "SPX500"],
    "Science":       [],
    "Sports":        [],
    "Entertainment": [],
}

IMPACT_URGENCY = {
    "Critical": "immediate",
    "High":     "immediate",
    "Medium":   "monitor",
    "Low":      "ignore",
}

IMPACT_EMOJI = {
    "Critical": "🚨",
    "High":     "🔥",
    "Medium":   "⚡",
    "Low":      "ℹ️",
}


@dataclass
class GlintSignal:
    signal_id: str
    category: str
    impact: str
    text: str
    source_tier: int
    relevance_score: float
    matched_market: str
    timestamp: str
    raw: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "GlintSignal":
        return cls(
            signal_id       = data.get("id", ""),
            category        = data.get("category", ""),
            impact          = data.get("impact", "Low"),
            text            = data.get("text", ""),
            source_tier     = int(data.get("source_tier", 3)),
            relevance_score = float(data.get("relevance_score", 0)),
            matched_market  = data.get("matched_market", ""),
            timestamp       = data.get("timestamp", ""),
            raw             = data,
        )

    def is_actionable(self) -> bool:
        return (
            self.impact in ("Critical", "High")
            and self.source_tier <= 2
            and self.relevance_score >= 7.0
        )

    def to_trading_context(self) -> Dict[str, Any]:
        instruments = CATEGORY_INSTRUMENTS.get(self.category, [])
        urgency = IMPACT_URGENCY.get(self.impact, "ignore")
        return {
            "signal_id":   self.signal_id,
            "headline":    self.text,
            "category":    self.category,
            "impact":      self.impact,
            "urgency":     urgency,
            "instruments": instruments,
            "relevance":   self.relevance_score,
            "timestamp":   self.timestamp,
            "action_hint": self._get_action_hint(),
        }

    def _get_action_hint(self) -> str:
        hints = {
            ("Economics", "Critical"): "Revisar XAUUSD, USD pairs — posible volatilidad extrema",
            ("Military",  "Critical"): "Risk-off: oro y yen pueden subir fuerte",
            ("Crypto",    "Critical"): "Alta volatilidad crypto — esperar confirmación en OB/FVG",
            ("Politics",  "High"):     "Monitorear USD y pares geopolíticos",
        }
        return hints.get(
            (self.category, self.impact),
            "Analizar en contexto SMC antes de operar",
        )

    def format_alert(self) -> str:
        emoji = IMPACT_EMOJI.get(self.impact, "")
        return (
            f"{emoji} *GLINT — {self.impact.upper()}*\n"
            f"Categoría: {self.category} | Tier {self.source_tier}\n"
            f"{self.text}\n"
            f"Mercado: {self.matched_market} | Relevancia: {self.relevance_score}/10\n"
            f"{self._get_action_hint()}\n"
            f"{self.timestamp}"
        )


class GlintConnector:
    """
    Polls Glint via HTTP every 15 s. Falls back to HTML scraping if the REST
    API is unavailable. Goes offline silently (logs once) if both fail, then
    retries every 5 minutes. The bot continues operating in all cases.
    """

    _API_URL  = "https://glint.trade/api/signals/latest"
    _FEED_URL = "https://glint.trade"

    def __init__(
        self,
        ws_url: str,          # kept for backward-compat with tests / callers
        session_token: str,
        on_signal: Optional[Callable[[GlintSignal], None]] = None,
        min_impact: str = "High",
    ):
        self.ws_url        = ws_url
        self.session_token = session_token
        self.on_signal     = on_signal
        self.min_impact    = min_impact
        self.connected     = False
        self._signals_received = 0
        self._offline      = False
        self._seen_ids: deque = deque(maxlen=MAX_SEEN_IDS)

    # ------------------------------------------------------------------
    # Public entry point — run as an asyncio Task; never raises/crashes.
    # ------------------------------------------------------------------
    async def connect(self):
        consecutive_failures = 0
        while True:
            try:
                signals = await self._poll_once()
                if signals is not None:
                    consecutive_failures = 0
                    if self._offline:
                        print("Glint reconectado — señales activas")
                        self._offline = False
                    self.connected = True
                    self._dispatch(signals)
                    await asyncio.sleep(POLLING_INTERVAL)
                else:
                    consecutive_failures += 1
                    self.connected = False
                    if not self._offline:
                        print("Glint en modo offline - bot opera sin noticias")
                        self._offline = True
                    await asyncio.sleep(RETRY_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(RETRY_INTERVAL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _poll_once(self) -> Optional[list]:
        signals = await self._fetch_api()
        if signals is None:
            signals = await self._fetch_scrape()
        return signals  # None means both methods failed

    async def _fetch_api(self) -> Optional[list]:
        try:
            import httpx
        except ImportError:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    self._API_URL,
                    headers={"Cookie": f"session={self.session_token}"},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                if isinstance(data, list):
                    return [GlintSignal.from_dict(item) for item in data]
                if isinstance(data, dict) and "signals" in data:
                    return [GlintSignal.from_dict(item) for item in data["signals"]]
        except Exception:
            pass
        return None

    async def _fetch_scrape(self) -> Optional[list]:
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    self._FEED_URL,
                    headers={
                        "Cookie": f"session={self.session_token}",
                        "User-Agent": "Mozilla/5.0",
                    },
                )
                if resp.status_code != 200:
                    return None
                soup = BeautifulSoup(resp.text, "html.parser")
                signals = []
                for item in soup.select("[data-signal], .signal-item, .news-item"):
                    raw = {
                        "id":              item.get("data-id", f"sc-{hash(item.get_text()[:50])}"),
                        "category":        item.get("data-category", "Economics"),
                        "impact":          item.get("data-impact", "Medium"),
                        "text":            item.get_text(strip=True)[:300],
                        "source_tier":     int(item.get("data-tier", 3)),
                        "relevance_score": float(item.get("data-relevance", 5.0)),
                        "matched_market":  item.get("data-market", ""),
                        "timestamp":       datetime.now(timezone.utc).isoformat(),
                    }
                    signals.append(GlintSignal.from_dict(raw))
                return signals if signals else None
        except Exception:
            pass
        return None

    def _dispatch(self, signals: list):
        for signal in signals:
            if signal.signal_id in self._seen_ids:
                continue
            self._seen_ids.append(signal.signal_id)
            self._signals_received += 1
            if self._should_process(signal) and self.on_signal:
                self.on_signal(signal)

    def _should_process(self, signal: GlintSignal) -> bool:
        impact_rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        min_rank    = impact_rank.get(self.min_impact, 3)
        sig_rank    = impact_rank.get(signal.impact, 1)
        return sig_rank >= min_rank

    def stats(self) -> Dict:
        return {
            "connected":        self.connected,
            "signals_received": self._signals_received,
            "ws_url":           self.ws_url,
            "mode":             "offline" if self._offline else "polling",
        }
