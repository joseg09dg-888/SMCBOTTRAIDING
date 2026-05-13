# dashboard/telegram_commander.py
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Callable, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


class BotMode(Enum):
    AUTO   = "auto"
    SEMI   = "semi"
    PAUSED = "paused"
    HYBRID = "hybrid"


@dataclass
class CommandResult:
    success: bool
    message: str
    action: str = ""


@dataclass
class BotStatus:
    mode: BotMode = BotMode.HYBRID
    paused: bool = False
    capital: float = 1000.0
    balance: float = 1000.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    open_positions: int = 0
    wins_today: int = 0
    losses_today: int = 0
    win_rate: float = 0.0
    drawdown: float = 0.0
    last_trade_symbol: str = ""
    last_trade_pnl: float = 0.0
    score_history: List[int] = field(default_factory=list)
    mode_history: List[dict] = field(default_factory=list)


COMMANDS = {
    "/auto":      "Activa modo 100% automatico",
    "/semi":      "Activa modo semi-auto (pide aprobacion)",
    "/pause":     "Pausa el bot completamente",
    "/resume":    "Reanuda el bot",
    "/status":    "Estado completo del bot",
    "/positions": "Lista posiciones abiertas con P&L en vivo",
    "/close_all": "Cierra todas las posiciones",
    "/train":     "Estado del curriculum de entrenamiento",
    "/scores":    "Ultimos 10 scores del DecisionFilter",
    "/risk":      "Estado del riesgo (correlaciones, sesion, volatilidad)",
    "/youtube":   "Estado del aprendizaje YouTube",
}


class TelegramCommander:
    """
    Full Telegram command interface for the SMC Trading Bot.
    Handles all /commands and routes to the appropriate bot actions.
    """

    def __init__(self, bot_token: str = "", chat_id: str = "",
                 on_close_all: Optional[Callable] = None):
        self.bot_token  = bot_token
        self.chat_id    = chat_id
        self.on_close_all = on_close_all
        self.state      = BotStatus()
        self._bot       = None
        self._app       = None

    def handle_command(self, command: str) -> CommandResult:
        """Synchronous command handler — used in tests and fallback mode."""
        cmd = command.strip().lower().split()[0]

        handlers: Dict[str, Callable] = {
            "/auto":      self._cmd_auto,
            "/semi":      self._cmd_semi,
            "/pause":     self._cmd_pause,
            "/resume":    self._cmd_resume,
            "/status":    self._cmd_status,
            "/positions": self._cmd_positions,
            "/close_all": self._cmd_close_all,
            "/train":     self._cmd_train,
            "/scores":    self._cmd_scores,
            "/risk":      self._cmd_risk,
            "/youtube":   self._cmd_youtube,
        }

        handler = handlers.get(cmd)
        if handler is None:
            return CommandResult(
                success=False,
                message=f"Comando desconocido: {cmd}. Usa /status para ver opciones.",
                action="unknown",
            )
        return handler()

    # ── Command handlers ──────────────────────────────────────────────────

    def _cmd_auto(self) -> CommandResult:
        self.state.mode = BotMode.AUTO
        self.state.paused = False
        self._log_mode_change("auto", "Activado por comando /auto")
        return CommandResult(
            success=True,
            message="Modo AUTO activado. Operare solo sin pedir confirmacion.",
            action="mode_change",
        )

    def _cmd_semi(self) -> CommandResult:
        self.state.mode = BotMode.SEMI
        self.state.paused = False
        self._log_mode_change("semi", "Activado por comando /semi")
        return CommandResult(
            success=True,
            message="Modo SEMI activado. Te pedire confirmacion antes de cada trade.",
            action="mode_change",
        )

    def _cmd_pause(self) -> CommandResult:
        self.state.paused = True
        self._log_mode_change("paused", "Pausado por comando /pause")
        return CommandResult(
            success=True,
            message="Bot pausado. No abrire nuevas posiciones hasta /resume.",
            action="pause",
        )

    def _cmd_resume(self) -> CommandResult:
        self.state.paused = False
        self._log_mode_change(self.state.mode.value, "Reanudado por comando /resume")
        return CommandResult(
            success=True,
            message=f"Bot reanudado. Modo activo: {self.state.mode.value.upper()}",
            action="resume",
        )

    def _cmd_status(self) -> CommandResult:
        s = self.state
        win_total = s.wins_today + s.losses_today
        win_rate_str = f"{s.wins_today}W / {s.losses_today}L ({s.win_rate:.1f}%)" if win_total > 0 else "Sin trades hoy"
        pnl_sign = "+" if s.daily_pnl >= 0 else ""
        status_msg = (
            f"SMC Bot Status\n"
            f"Modo: {s.mode.value.upper()} | {'PAUSADO' if s.paused else 'ACTIVO'}\n"
            f"Capital: ${s.capital:,.2f} | Balance: ${s.balance:,.2f}\n"
            f"Posiciones abiertas: {s.open_positions}\n"
            f"P&L hoy: {pnl_sign}${s.daily_pnl:.2f}\n"
            f"P&L total: {'+' if s.total_pnl >= 0 else ''}${s.total_pnl:.2f}\n"
            f"Trades hoy: {win_rate_str}\n"
            f"Win rate total: {s.win_rate:.1f}%\n"
            f"Drawdown actual: {s.drawdown:.1f}%\n"
            f"Ultimo trade: {s.last_trade_symbol} {'+' if s.last_trade_pnl >= 0 else ''}${s.last_trade_pnl:.2f}"
        )
        return CommandResult(success=True, message=status_msg, action="status")

    def _cmd_positions(self) -> CommandResult:
        if self.state.open_positions == 0:
            return CommandResult(success=True, message="Sin posiciones abiertas actualmente.", action="positions")
        return CommandResult(
            success=True,
            message=f"Posiciones abiertas: {self.state.open_positions}\n(Conecta Market Connector para ver detalle)",
            action="positions",
        )

    def _cmd_close_all(self) -> CommandResult:
        if self.on_close_all:
            self.on_close_all()
        return CommandResult(
            success=True,
            message="Cerrando todas las posiciones...",
            action="close_all",
        )

    def _cmd_train(self) -> CommandResult:
        return CommandResult(
            success=True,
            message="Curriculum: 0/6 hitos completados. Usa training/curriculum.py para ver detalle.",
            action="train",
        )

    def _cmd_scores(self) -> CommandResult:
        history = self.state.score_history[-10:]
        if not history:
            return CommandResult(success=True, message="Sin scores registrados aun.", action="scores")
        score_lines = "\n".join(f"  Score {i+1}: {s}/100" for i, s in enumerate(history))
        avg = sum(history) / len(history)
        return CommandResult(
            success=True,
            message=f"Ultimos {len(history)} scores del DecisionFilter:\n{score_lines}\nPromedio: {avg:.1f}/100",
            action="scores",
        )

    def _cmd_risk(self) -> CommandResult:
        s = self.state
        return CommandResult(
            success=True,
            message=(
                f"Estado de Riesgo:\n"
                f"  Drawdown actual: {s.drawdown:.1f}%\n"
                f"  Posiciones abiertas: {s.open_positions}\n"
                f"  P&L del dia: {'+' if s.daily_pnl >= 0 else ''}${s.daily_pnl:.2f}\n"
                f"  Estado: {'OK' if s.drawdown < 3 else 'ELEVADO'}"
            ),
            action="risk",
        )

    def _cmd_youtube(self) -> CommandResult:
        return CommandResult(
            success=True,
            message="YouTube Trainer: 0 estrategias cargadas. Ejecuta YouTubeTrainer.process_video() para aprender.",
            action="youtube",
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_mode_change(self, mode: str, reason: str):
        self.state.mode_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "reason": reason,
        })

    async def send_message(self, text: str):
        if not HAS_TELEGRAM or not self.bot_token:
            print(f"[Telegram] {text[:100]}")
            return
        try:
            bot = Bot(token=self.bot_token)
            await bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def update_state(self, **kwargs):
        """Update bot state from supervisor."""
        for k, v in kwargs.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)
