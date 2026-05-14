"""
core/continuous_learning.py
----------------------------
Continuous learning engine for the trading agent.

Design constraints:
- No ChromaDB — lessons stored in-memory list.
- No real YouTube API — check_youtube_live uses HTTP with try/except.
- No Claude API — analysis is deterministic.
- Telegram imports are optional (try/except).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TradeLesson:
    symbol: str
    direction: str          # "long" | "short"
    entry: float
    exit_price: float
    pnl: float
    reason: str             # motivo de ganancia/pérdida
    tags: list              # ["OB", "FVG", "BOS", ...]
    timestamp: datetime


@dataclass
class YouTubeChannel:
    name: str
    channel_id: str         # YouTube channel ID
    is_live: bool
    last_checked: datetime


@dataclass
class AdjustmentSuggestion:
    component: str          # "smc" | "ml" | "sentiment" | "energy"
    current_value: int
    suggested_value: int
    reason: str
    win_rate_evidence: float   # e.g. 0.78


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ContinuousLearningEngine:
    """
    Continuous learning engine: records trade lessons, analyses patterns,
    monitors YouTube channels and generates parameter-adjustment suggestions.
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    CHANNELS: list[YouTubeChannel] = [
        YouTubeChannel(
            name="ICT",
            channel_id="UCqGE4ZTvCEVimRBHs6Yqw5g",
            is_live=False,
            last_checked=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        YouTubeChannel(
            name="Rayner Teo",
            channel_id="UCf-aBGkFwUBbHVkLXR7Yjiw",
            is_live=False,
            last_checked=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        YouTubeChannel(
            name="Trader Dante",
            channel_id="UCl_HTqnwKyPrL6kRVoVMfcA",
            is_live=False,
            last_checked=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        YouTubeChannel(
            name="Anton Kreil",
            channel_id="UCaJWTbHmBCSV2SKcJgNBSBQ",
            is_live=False,
            last_checked=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    ]

    FAMOUS_CRASHES: list[dict] = [
        {
            "name": "Black Monday 1987",
            "date": "1987-10-19",
            "move_pct": -22.6,
            "asset": "DJIA",
            "smc_signals": ["Distribution phase visible on weekly", "Bearish OB on daily", "Liquidity sweep above highs"],
            "lesson": "Largest single-day percentage drop in DJIA history. Portfolio insurance and program trading amplified the crash.",
        },
        {
            "name": "Flash Crash 2010",
            "date": "2010-05-06",
            "move_pct": -9.2,
            "asset": "DJIA",
            "smc_signals": ["Extreme order-flow imbalance", "FVG left unfilled on 1m", "Rapid liquidity void"],
            "lesson": "HFT and ETF liquidity collapse caused a trillion-dollar drop and partial recovery in minutes.",
        },
        {
            "name": "COVID Crash 2020",
            "date": "2020-03-12",
            "move_pct": -26.7,
            "asset": "S&P500",
            "smc_signals": ["Weekly bearish BOS", "Monthly FVG never filled", "Institutional distribution Jan-Feb"],
            "lesson": "Pandemic panic selling. Fastest bear market in history. Recovered fully within months.",
        },
        {
            "name": "Luna Crash 2022",
            "date": "2022-05-09",
            "move_pct": -98.0,
            "asset": "LUNA/USD",
            "smc_signals": ["Death spiral algorithm visible in order book", "UST depeg 1h signal", "Liquidity wipeout on 15m"],
            "lesson": "Algorithmic stablecoin failure. UST depeg triggered death spiral. Wiped ~$40B in days.",
        },
        {
            "name": "FTX Collapse 2022",
            "date": "2022-11-08",
            "move_pct": -25.0,
            "asset": "BTC/USD",
            "smc_signals": ["Exchange outflow anomaly", "BTC daily bearish OB rejection", "Market-wide liquidity drain"],
            "lesson": "FTX exchange insolvency. Binance offer to acquire then withdrawal caused contagion.",
        },
        {
            "name": "Dotcom Crash 2000",
            "date": "2000-03-10",
            "move_pct": -78.0,
            "asset": "NASDAQ",
            "smc_signals": ["Multi-year distribution phase", "Failed higher-high on weekly", "Volume divergence on monthly"],
            "lesson": "Speculation bubble in internet companies. NASDAQ fell 78% over 2.5 years.",
        },
        {
            "name": "Lehman 2008",
            "date": "2008-09-15",
            "move_pct": -45.0,
            "asset": "S&P500",
            "smc_signals": ["Banking sector bearish BOS on weekly", "Credit default swap signals", "Monthly OB breakdown"],
            "lesson": "Lehman Brothers bankruptcy triggered global financial crisis. Housing bubble collapse.",
        },
    ]

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        telegram_bot=None,
        suggestion_interval_hours: int = 6,
    ) -> None:
        self._telegram_bot = telegram_bot
        self._suggestion_interval_hours = suggestion_interval_hours
        self._lessons: list[TradeLesson] = []
        # Map tag -> list of win booleans for fast win-rate computation
        self._tag_results: dict[str, list[bool]] = {}

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def record_trade(self, lesson: TradeLesson) -> None:
        """Save lesson to in-memory list and update tag win-rate index."""
        self._lessons.append(lesson)
        win = lesson.pnl > 0
        for tag in lesson.tags:
            self._tag_results.setdefault(tag, []).append(win)

    def analyze_trade_lesson(self, lesson: TradeLesson) -> dict:
        """
        Deterministic analysis of a trade lesson.
        Returns dict with keys: 'cause', 'tags', 'improvement'.
        """
        if lesson.pnl > 0:
            cause = (
                f"Ganancia de {lesson.pnl:.2f} en {lesson.direction.upper()} "
                f"— profit confirmado por {lesson.reason}"
            )
            improvement = "Mantener configuración actual para este setup."
        else:
            cause = (
                f"Perdida de {abs(lesson.pnl):.2f} en {lesson.direction.upper()} "
                f"— loss causado por {lesson.reason}"
            )
            improvement = "Revisar entrada y confirmación de señal antes del próximo trade."

        tag_insights = []
        for tag in lesson.tags:
            rate = self.get_win_rate_by_tag(tag)
            tag_insights.append(f"{tag}: {rate:.0%} win rate histórico")

        return {
            "cause": cause,
            "tags": tag_insights,
            "improvement": improvement,
            "pnl": lesson.pnl,
            "direction": lesson.direction,
        }

    def get_recent_lessons(self, n: int = 10) -> list[TradeLesson]:
        """Return the last *n* lessons (most-recently added last)."""
        return self._lessons[-n:]

    def get_win_rate_by_tag(self, tag: str) -> float:
        """Return fraction of winning trades that carry *tag*. 0.0 if no trades."""
        results = self._tag_results.get(tag, [])
        if not results:
            return 0.0
        return sum(results) / len(results)

    def generate_adjustment_suggestion(self) -> Optional[AdjustmentSuggestion]:
        """
        If any tag has >= 5 trades and win rate > 70%, suggest increasing the
        corresponding component weight.  Returns None if evidence is insufficient.
        """
        # Map tags to components
        tag_to_component = {
            "OB": "smc",
            "FVG": "smc",
            "BOS": "smc",
            "CHoCH": "smc",
            "ML": "ml",
            "AI": "ml",
            "SENT": "sentiment",
            "NEWS": "sentiment",
            "ENERGY": "energy",
        }

        best_tag: Optional[str] = None
        best_rate: float = 0.0

        for tag, results in self._tag_results.items():
            if len(results) < 5:
                continue
            rate = sum(results) / len(results)
            if rate > 0.70 and rate > best_rate:
                best_tag = tag
                best_rate = rate

        if best_tag is None:
            return None

        component = tag_to_component.get(best_tag, "smc")
        trade_count = len(self._tag_results[best_tag])

        return AdjustmentSuggestion(
            component=component,
            current_value=30,
            suggested_value=40,
            reason=(
                f"Tag '{best_tag}' tiene {best_rate:.0%} win rate "
                f"en {trade_count} trades recientes"
            ),
            win_rate_evidence=best_rate,
        )

    async def format_suggestion_telegram(self, suggestion: AdjustmentSuggestion) -> str:
        """Format an AdjustmentSuggestion as a Telegram-friendly approval message."""
        return (
            f"💡 SUGERENCIA DE AJUSTE\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Componente : {suggestion.component.upper()}\n"
            f"Valor actual   : {suggestion.current_value}\n"
            f"Valor sugerido : {suggestion.suggested_value}\n"
            f"Win rate evidencia: {suggestion.win_rate_evidence:.0%}\n"
            f"\n📊 Razón: {suggestion.reason}\n"
            f"\n¿Aprobar ajuste? Responde /aprobar o /rechazar"
        )

    def get_crash_analysis(self, crash_name: str) -> dict:
        """Return historical crash analysis dict. Returns {'error': ...} if unknown."""
        for crash in self.FAMOUS_CRASHES:
            if crash["name"].lower() == crash_name.lower():
                return dict(crash)  # return a copy
        return {"error": f"Crash '{crash_name}' no encontrado en la base de datos histórica."}

    def check_youtube_live(self, channel: YouTubeChannel) -> bool:
        """
        Attempt to check if a YouTube channel is live via HTTP.
        Always returns bool — network errors → False.
        """
        if _requests is None:
            return False

        url = f"https://www.youtube.com/channel/{channel.channel_id}/live"
        try:
            resp = _requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                text = resp.text
                # YouTube embeds isLiveBroadcast or "LIVE" indicators in page HTML
                live_indicators = [
                    '"isLiveBroadcast":true',
                    "isLiveBroadcast",
                    '"style":"LIVE"',
                    '"broadcastType":"LIVE"',
                ]
                return any(ind in text for ind in live_indicators)
            return False
        except Exception:
            return False

    async def run_study_cycle(self) -> None:
        """
        Asyncio task that every N hours:
        1. Checks YouTube channels for live status.
        2. Generates an adjustment suggestion if evidence exists.
        3. Sends a report via telegram_bot.send_glint_alert().
        """
        while True:
            try:
                # 1. Check live channels
                live_channels: list[str] = []
                for channel in self.CHANNELS:
                    is_live = self.check_youtube_live(channel)
                    channel.is_live = is_live
                    channel.last_checked = datetime.now(timezone.utc)
                    if is_live:
                        live_channels.append(channel.name)

                # 2. Generate suggestion
                suggestion = self.generate_adjustment_suggestion()

                # 3. Build report
                report_lines = ["📚 CICLO DE ESTUDIO CONTINUO"]
                if live_channels:
                    report_lines.append(f"🔴 En vivo: {', '.join(live_channels)}")
                else:
                    report_lines.append("⚪ Ningún canal en vivo ahora.")

                report_lines.append(self.get_study_report())

                if suggestion:
                    suggestion_text = await self.format_suggestion_telegram(suggestion)
                    report_lines.append(suggestion_text)

                full_report = "\n".join(report_lines)

                # 4. Send via Telegram
                if self._telegram_bot is not None:
                    await self._telegram_bot.send_glint_alert(full_report)
                else:
                    print(full_report)

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                print(f"[ContinuousLearning] Error en ciclo: {exc}")

            await asyncio.sleep(self._suggestion_interval_hours * 3600)

    def get_study_report(self) -> str:
        """Return a plain-text summary of the current learning state."""
        total = len(self._lessons)
        wins = sum(1 for l in self._lessons if l.pnl > 0)
        losses = total - wins
        overall_wr = (wins / total * 100) if total > 0 else 0.0

        lines = [
            f"📈 Reporte de Aprendizaje Continuo",
            f"Total trades registrados: {total}",
            f"Ganadores: {wins} | Perdedores: {losses}",
            f"Win rate global: {overall_wr:.1f}%",
        ]

        if self._tag_results:
            lines.append("Win rate por tag:")
            for tag, results in sorted(self._tag_results.items()):
                rate = sum(results) / len(results)
                lines.append(f"  {tag}: {rate:.0%} ({len(results)} trades)")

        live_count = sum(1 for ch in self.CHANNELS if ch.is_live)
        lines.append(f"Canales YouTube en vivo: {live_count}/{len(self.CHANNELS)}")

        return "\n".join(lines)
