import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any

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
            signal_id      = data.get("id", ""),
            category       = data.get("category", ""),
            impact         = data.get("impact", "Low"),
            text           = data.get("text", ""),
            source_tier    = int(data.get("source_tier", 3)),
            relevance_score= float(data.get("relevance_score", 0)),
            matched_market = data.get("matched_market", ""),
            timestamp      = data.get("timestamp", ""),
            raw            = data,
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
    Connects to Glint's real-time WebSocket feed.
    Authenticates with a browser session token.
    Fires callbacks on actionable signals in < 30s.
    """

    def __init__(
        self,
        ws_url: str,
        session_token: str,
        on_signal: Optional[Callable[[GlintSignal], None]] = None,
        min_impact: str = "High",
    ):
        self.ws_url = ws_url
        self.session_token = session_token
        self.on_signal = on_signal
        self.min_impact = min_impact
        self.connected = False
        self._signals_received = 0
        self._ws = None

    async def connect(self):
        try:
            import websockets
        except ImportError:
            print("websockets not installed — install with: pip install websockets")
            return

        headers = {
            "Cookie": f"session={self.session_token}",
            "Origin": "https://glint.trade",
        }
        try:
            async with websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10,
            ) as ws:
                self._ws = ws
                self.connected = True
                print(f"Glint connected: {self.ws_url}")
                await self._listen(ws)
        except Exception as e:
            self.connected = False
            print(f"Glint error: {e}")
            await asyncio.sleep(5)
            await self.connect()

    async def _listen(self, ws):
        async for message in ws:
            try:
                data = json.loads(message)
                if data.get("type") == "signal":
                    signal = GlintSignal.from_dict(data.get("payload", data))
                    self._signals_received += 1
                    if self._should_process(signal) and self.on_signal:
                        self.on_signal(signal)
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"Error processing Glint signal: {e}")

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
        }
