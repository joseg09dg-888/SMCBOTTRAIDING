# dashboard/telegram_commander.py
import asyncio
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
    "/history":          "Análisis histórico de un símbolo. Ej: /history BTC",
    "/memory":           "Estado de memoria y accuracy de todos los agentes",
    "/health":           "Health check de los 21 agentes del bot",
    "/energy":           "Lectura energetica del mercado. Ej: /energy BTC",
    "/reporte_semanal":  "Genera reporte semanal ahora",
    "/reporte_mensual":  "Genera reporte mensual ahora",
    "/criterios":        "Muestra criterios para ir a cuenta real",
    "/proyeccion":       "Proyeccion de la proxima semana",
    "/vision":           "Activa/desactiva vision de pantalla",
    "/screenshot":       "Captura y analiza pantalla ahora",
    "/mirror":           "Activa/desactiva modo espejo",
    "/analysis":         "Análisis SMC completo del mercado. Ej: /analysis BTC",
    "/onchain":          "Métricas on-chain actuales (flujos ballenas, exchange netflow)",
    "/lunar":            "Análisis de ciclos lunares y su correlación con el mercado",
    "/elliott":          "Conteo de ondas de Elliott en el símbolo activo",
    "/edge":             "Statistical edge y winrate histórico del sistema",
}


class TelegramCommander:
    """
    Full Telegram command interface for the SMC Trading Bot.
    Handles all /commands and routes to the appropriate bot actions.
    Call start_polling() as an asyncio task to activate command listening.
    """

    # Class-level defaults so tests that use __new__() don't get AttributeError
    on_mode_change = None
    on_callback    = None
    on_close_all   = None
    on_history     = None
    on_memory      = None

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        on_close_all: Optional[Callable] = None,
        on_mode_change: Optional[Callable[[str], None]] = None,
        on_callback: Optional[Callable] = None,
        on_history: Optional[Callable[[str], str]] = None,
        on_memory: Optional[Callable[[], str]] = None,
    ):
        self.bot_token      = bot_token
        self.chat_id        = chat_id
        self.on_close_all   = on_close_all
        self.on_mode_change = on_mode_change
        self.on_callback    = on_callback
        self.on_history     = on_history
        self.on_memory      = on_memory
        self.state          = BotStatus()
        self._bot           = None
        self._app           = None

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
            "/history":   self._cmd_history,
            "/memory":           self._cmd_memory,
            "/health":           self._cmd_health,
            "/energy":           self._cmd_energy,
            "/reporte_semanal":  self._cmd_reporte_semanal,
            "/reporte_mensual":  self._cmd_reporte_mensual,
            "/criterios":        self._cmd_criterios,
            "/proyeccion":       self._cmd_proyeccion,
            "/vision":           self._cmd_vision,
            "/screenshot":       self._cmd_screenshot,
            "/mirror":           self._cmd_mirror,
            "/analysis":         self._cmd_analysis,
            "/onchain":          self._cmd_onchain,
            "/lunar":            self._cmd_lunar,
            "/elliott":          self._cmd_elliott,
            "/edge":             self._cmd_edge,
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
        if self.on_mode_change:
            self.on_mode_change("auto")
        return CommandResult(
            success=True,
            message="Modo AUTO activado. Operare solo sin pedir confirmacion.",
            action="mode_change",
        )

    def _cmd_semi(self) -> CommandResult:
        self.state.mode = BotMode.SEMI
        self.state.paused = False
        self._log_mode_change("semi", "Activado por comando /semi")
        if self.on_mode_change:
            self.on_mode_change("semi")
        return CommandResult(
            success=True,
            message="Modo SEMI activado. Te pedire confirmacion antes de cada trade.",
            action="mode_change",
        )

    def _cmd_pause(self) -> CommandResult:
        self.state.paused = True
        self._log_mode_change("paused", "Pausado por comando /pause")
        if self.on_mode_change:
            self.on_mode_change("paused")
        return CommandResult(
            success=True,
            message="Bot pausado. No abrire nuevas posiciones hasta /resume.",
            action="pause",
        )

    def _cmd_resume(self) -> CommandResult:
        self.state.paused = False
        self._log_mode_change(self.state.mode.value, "Reanudado por comando /resume")
        if self.on_mode_change:
            self.on_mode_change(self.state.mode.value)
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

    def _cmd_health(self) -> CommandResult:
        from core.agent_health_check import AgentHealthCheck
        checker = AgentHealthCheck()
        report = checker.run_full_check()
        return CommandResult(
            success=True,
            message=report.format_telegram(),
            action="health",
        )

    def _cmd_history(self) -> CommandResult:
        if self.on_history:
            try:
                text = self.on_history("BTC")
            except Exception as e:
                text = f"Error al obtener historial: {e}"
        else:
            text = (
                "Historial: agente histórico no conectado.\n"
                "Usa /history BTC para análisis por símbolo."
            )
        return CommandResult(success=True, message=text, action="history")

    def _cmd_memory(self) -> CommandResult:
        if self.on_memory:
            try:
                text = self.on_memory()
            except Exception as e:
                text = f"Error al obtener memoria: {e}"
        else:
            text = (
                "Memoria del bot: sin conexion a AgentMemoryManager.\n"
                "Reinicia el bot con memoria activa para ver estadisticas."
            )
        return CommandResult(success=True, message=text, action="memory")

    def _cmd_energy(self) -> CommandResult:
        from agents.energy_frequency_agent import EnergyFrequencyAgent
        agent = EnergyFrequencyAgent()
        reading = agent.analyze("BTC", price=0.0)
        return CommandResult(
            success=True,
            message=reading.format_telegram(),
            action="energy",
        )

    def _cmd_reporte_semanal(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date, timedelta
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        days_since_monday = today.weekday()
        week_start = today - timedelta(days=days_since_monday)
        stats = agent.calculate_weekly_stats(week_start)
        summary = agent.generate_telegram_summary(stats)
        return CommandResult(success=True, message=summary, action="reporte_semanal")

    def _cmd_reporte_mensual(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        stats = agent.calculate_monthly_stats(today.year, today.month)
        summary = agent.generate_telegram_summary(stats)
        return CommandResult(success=True, message=summary, action="reporte_mensual")

    def _cmd_criterios(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        agent = ReportAgent(capital=self.state.capital)
        msg = agent.generate_criteria_message()
        return CommandResult(success=True, message=msg, action="criterios")

    def _cmd_proyeccion(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        agent = ReportAgent(capital=self.state.capital)
        msg = agent.generate_projection_message()
        return CommandResult(success=True, message=msg, action="proyeccion")

    def _cmd_vision(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        new_state = agent.toggle()
        status = "activada" if new_state else "desactivada"
        return CommandResult(success=True, message=f"Vision de pantalla {status}.", action="vision")

    def _cmd_screenshot(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        cap = agent.capture_full_screen() or agent.create_mock_capture()
        analysis = agent.analyze_capture(cap)
        msg = agent.build_alert_message(analysis, "full")
        return CommandResult(success=True, message=msg, action="screenshot")

    def _cmd_mirror(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        if not agent._mirror_active:
            agent.start_mirror_mode()
            msg = "Modo espejo ACTIVADO. El bot aprende de tus operaciones."
        else:
            session = agent.stop_mirror_mode()
            actions = session.actions_recorded if session else 0
            msg = f"Modo espejo DESACTIVADO. Acciones grabadas: {actions}"
        return CommandResult(success=True, message=msg, action="mirror")

    def _cmd_analysis(self) -> CommandResult:
        return CommandResult(
            success=True,
            message=(
                "Análisis SMC: Para análisis completo pasa un símbolo.\n"
                "Ej: /analysis BTC\n"
                "(Agente de análisis no conectado en modo standalone)"
            ),
            action="analysis",
        )

    def _cmd_onchain(self) -> CommandResult:
        return CommandResult(
            success=True,
            message=(
                "On-Chain Metrics:\n"
                "  Exchange Netflow: sin datos en tiempo real\n"
                "  Whale Flows: sin datos en tiempo real\n"
                "Conecta OnchainAgent para métricas en vivo."
            ),
            action="onchain",
        )

    def _cmd_lunar(self) -> CommandResult:
        return CommandResult(
            success=True,
            message=(
                "Análisis Lunar:\n"
                "  Ciclo lunar: disponible vía LunarAgent\n"
                "  Correlación histórica: sin datos cargados\n"
                "Conecta LunarAgent para lectura completa."
            ),
            action="lunar",
        )

    def _cmd_elliott(self) -> CommandResult:
        return CommandResult(
            success=True,
            message=(
                "Ondas de Elliott:\n"
                "  Conteo activo: sin datos de mercado en tiempo real\n"
                "  Pasa un símbolo al supervisor para análisis completo.\n"
                "Conecta ElliottAgent para conteo en vivo."
            ),
            action="elliott",
        )

    def _cmd_edge(self) -> CommandResult:
        return CommandResult(
            success=True,
            message=(
                "Statistical Edge del Sistema:\n"
                "  Winrate histórico: sin trades registrados aún\n"
                "  Expectancy: N/A\n"
                "  Sharpe Ratio: N/A\n"
                "Ejecuta /history para ver datos por símbolo."
            ),
            action="edge",
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

    async def start_polling(self):
        """
        Starts listening for Telegram commands as an asyncio task.
        Reconnects automatically every 30 s on any connection failure (silent).
        Compatible with python-telegram-bot v20+.
        """
        if not HAS_TELEGRAM or not self.bot_token:
            print("[Telegram] Sin token — polling de comandos desactivado")
            await asyncio.Event().wait()
            return

        while True:
            try:
                app = Application.builder().token(self.bot_token).build()

                for cmd in COMMANDS:
                    app.add_handler(CommandHandler(cmd.lstrip("/"), self._make_handler(cmd)))

                if self.on_callback:
                    app.add_handler(CallbackQueryHandler(self.on_callback))

                async with app:
                    await app.start()
                    await app.updater.start_polling(drop_pending_updates=True)
                    await self.send_message(
                        "🤖 Bot online — comandos activos: /status /auto /semi /pause /resume "
                        "/positions /scores /risk /train /youtube"
                    )
                    print("[Telegram] Polling activo — escuchando comandos")
                    await asyncio.Event().wait()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Telegram polling perdido, reconectando en 30s: %s", exc)
                await asyncio.sleep(30)

    def _make_handler(self, cmd: str):
        """Returns a PTB-compatible async handler for the given command."""
        if cmd == "/history":
            return self._make_history_handler()

        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not update.message:
                return
            result = self.handle_command(cmd)
            parse = "Markdown" if any(c in result.message for c in ("*", "_", "`")) else None
            try:
                await update.message.reply_text(result.message, parse_mode=parse)
            except Exception as e:
                logger.error(f"Reply error for {cmd}: {e}")
        return handler

    def _make_history_handler(self):
        """Handler for /history [symbol] — calls on_history callback."""
        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not update.message:
                return
            symbol = (context.args[0].upper() if context.args else "BTC")
            if self.on_history:
                try:
                    text = self.on_history(symbol)
                except Exception as e:
                    text = f"Error generando historial para {symbol}: {e}"
            else:
                text = f"Agente histórico no disponible. Reinicia el bot."
            parse = "Markdown" if "*" in text else None
            try:
                await update.message.reply_text(text, parse_mode=parse)
            except Exception as e:
                logger.error(f"History reply error: {e}")
        return handler

    def update_state(self, **kwargs):
        """Update bot state from supervisor."""
        for k, v in kwargs.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)
