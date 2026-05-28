"""AxiVisionAgent — Claude Vision reads the MT5 screen and extracts trading state."""
from __future__ import annotations

import base64
import io
import json

import anthropic
import mss
from PIL import Image

from core.config import config

_PROMPT = """You are an expert SMC trader analyzing an MT5 terminal screenshot.
Extract the following information and respond ONLY with valid JSON — no markdown, no explanation:
{
  "balance": <float, account balance>,
  "patrimonio": <float, equity>,
  "posiciones": [
    {"symbol": "<string>", "direction": "<BUY|SELL>", "volume": <float>, "pnl": <float>}
  ],
  "alertas": ["<string>"],
  "setup_visible": <true|false, is there a clear SMC setup on any visible chart>,
  "accion_recomendada": "<string, what action should the bot take now>"
}
If a field is not visible, use 0 for numbers, [] for lists, and "" for strings."""


class AxiVisionAgent:
    """Captures the MT5 screen and uses Claude Vision to extract trading state."""

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def capture_mt5_screen(self) -> str:
        """Capture the primary monitor and return a base64-encoded JPEG."""
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode()

    def analyze_mt5_screen(self) -> dict:
        """Capture screen and ask Claude Vision to extract MT5 state."""
        img_b64 = self.capture_mt5_screen()
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }],
        )
        text = response.content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw": text}

    def should_close_position(self, symbol: str, pnl: float) -> bool:
        """Return True if position loss exceeds the emergency close threshold."""
        return pnl < -500

    def monitor_and_protect(self) -> dict:
        """Analyze screen and build alerts for losing positions."""
        analysis = self.analyze_mt5_screen()
        alerts: list[str] = []

        for pos in analysis.get("posiciones", []):
            pnl = pos.get("pnl", 0)
            symbol = pos.get("symbol", "")
            if pnl < -500:
                alerts.append(f"CERRAR {symbol} -- perdida critica ${abs(pnl):.0f}")
            elif pnl < -100:
                alerts.append(f"{symbol} perdiendo ${abs(pnl):.0f}")

        return {
            "analysis": analysis,
            "alerts": alerts,
            "balance": analysis.get("balance", 0),
        }
