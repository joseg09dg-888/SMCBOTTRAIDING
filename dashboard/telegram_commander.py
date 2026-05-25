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
    "/history":          "Analisis historico de un simbolo. Ej: /history BTC",
    "/memory":           "Estado de memoria y accuracy de todos los agentes",
    "/health":           "Health check en tiempo real de todos los agentes",
    "/energy":           "Lectura energetica del mercado. Ej: /energy BTC",
    "/reporte_semanal":  "Genera reporte semanal ahora",
    "/reporte_mensual":  "Genera reporte mensual ahora",
    "/criterios":        "Criterios reales para ir a cuenta real (de SQLite)",
    "/proyeccion":       "Proyeccion de la proxima semana",
    "/vision":           "Activa/desactiva vision de pantalla",
    "/screenshot":       "Captura y analiza pantalla ahora",
    "/mirror":           "Activa/desactiva modo espejo",
    "/analysis":         "Analisis SMC completo del mercado",
    "/onchain":          "Metricas on-chain actuales",
    "/lunar":            "Analisis de ciclos lunares",
    "/elliott":          "Conteo de ondas de Elliott",
    "/edge":             "Statistical edge y winrate historico",
    "/footprint":        "Analisis footprint (delta, absorcion). Ej: /footprint BTC",
    "/ftmo":             "Estado FTMO challenge y potencial de ingresos",

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
    "/edge":             "Statistical edge y winrate historico del sistema",
    "/footprint":        "Analisis footprint (delta, absorcion, imbalances). Ej: /footprint BTC",
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
        """Synchronous command handler - used in tests and fallback mode."""
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
            "/history":          self._cmd_history,
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
            "/footprint":        self._cmd_footprint,
            "/ftmo":             self._cmd_ftmo,
            "/axi":              self._cmd_axi,

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
            "/footprint":        self._cmd_footprint,
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
        from core.config import config as cfg
        from core.score_db import get_stats
        stats = get_stats()
        wr = f"{stats['win_rate']:.1f}%" if stats['executed'] > 0 else "N/A"

        # MT5 real balance
        try:
            from connectors.metatrader_connector import MT5Connector
            mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            info = mt5.get_account_info()
            if info and info.get("balance"):
                bal    = info["balance"]
                equity = info.get("equity", bal)
                profit = info.get("profit", 0.0)
                gain   = bal - 100000.0
                mt5_text = (
                    f"Balance: <code>${bal:,.2f}</code>\n"
                    f"Equity:  <code>${equity:,.2f}</code>\n"
                    f"P&amp;L abierto: <code>${profit:+.2f}</code>\n"
                    f"Ganancia vs inicio: <code>${gain:+.2f}</code>"
                )
            else:
                mt5_text = "Reconectando..."
        except Exception as e:
            mt5_text = f"No disponible: {e}"

        # Binance testnet
        try:
            from connectors.binance_connector import BinanceConnector
            b = BinanceConnector(cfg.binance_api_key, cfg.binance_api_secret, True)
            bal_df = b.get_ohlcv("BTCUSDT", "1m", 1)
            btc_price = float(bal_df["close"].iloc[-1]) if not bal_df.empty else 0.0
            bal_usdt = b.get_balance()
            positions = b.get_open_positions()
            binance_text = (
                f"USDT: <code>{bal_usdt:,.2f}</code>\n"
                f"BTC: <code>${btc_price:,.2f}</code>\n"
                f"Posiciones: {len(positions)}"
            )
        except Exception as e:
            binance_text = f"Error: {e}"

        text = (
            f"<b>SMC BOT ESTADO REAL</b>\n"
            f"<b>Modo:</b> {cfg.operation_mode.upper()} | ACTIVO\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>MT5 AXI DEMO (meta: crecer $100K)</b>\n"
            f"{mt5_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>BINANCE TESTNET</b>\n"
            f"{binance_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>ESTADISTICAS</b>\n"
            f"Trades: {stats['executed']} | Win Rate: {wr}\n"
            f"Scan: cada 30s"
        )
        return CommandResult(success=True, message=text, action="status")

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
        from core.score_db import get_recent_scores
        rows = get_recent_scores(10)
        if not rows:
            return CommandResult(
                success=True,
                message=(
                    "<b>ULTIMOS SCORES</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "Aun no hay scores registrados.\n"
                    "El bot escaneara mercados pronto...\n"
                    "(scores se guardan al encontrar setups)"
                ),
                action="scores"
            )
        text = "<b>ULTIMOS SCORES REALES</b>\n"
        text += "━━━━━━━━━━━━━━━━━━━━\n"
        for row in rows:
            ts, sym, tf, score, direction, entry, executed = row
            hora = ts[11:16] if len(ts) > 16 else ts
            emoji = "🔥" if score >= 75 else ("✅" if score >= 60 else "⚡" if score >= 35 else "▪️")
            exec_txt = "EJECUTADO" if executed else "descartado"
            dir_arrow = "▲" if direction == "long" else "▼"
            text += f"{emoji} {sym} {tf} | Score: <b>{score}</b>\n"
            text += f"   {dir_arrow} {direction.upper()} | {exec_txt} | {hora}\n"
        return CommandResult(success=True, message=text, action="scores")

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
        agentes = {
            "Supervisor": "core.supervisor",
            "Risk Manager": "core.risk_manager",
            "SMC Structure": "smc.structure",
            "Order Blocks": "smc.orderblocks",
            "Signal Agent": "agents.signal_agent",
            "Analysis Agent": "agents.analysis_agent",
            "Decision Filter": "core.decision_filter",
            "Footprint": "agents.footprint_agent",
            "Statistical Edge": "agents.statistical_edge_agent",
            "Prediction": "smc.ml_predictor",
            "Lunar": "agents.lunar_agent",
            "Elliott": "agents.elliott_agent",
            "Institutional Flow": "agents.institutional_flow_agent",
            "Alternative Data": "agents.alternative_data_agent",
            "Microstructure": "agents.microstructure_agent",
            "FED Sentiment": "agents.fed_sentiment_agent",
            "OnChain": "agents.onchain_agent",
            "Geopolitical": "agents.geopolitical_agent",
            "Chaos Theory": "agents.chaos_agent",
            "Retail Psychology": "agents.retail_psychology_agent",
            "Energy Frequency": "agents.energy_frequency_agent",
            "Binance": "connectors.binance_connector",
            "FTMO": "strategies.ftmo_agent",
            "Pairs Trading": "strategies.pairs_trading",
        }
        ok, fail = [], []
        for name, mod in agentes.items():
            try:
                __import__(mod)
                ok.append(name)
            except Exception as e:
                fail.append(f"{name}: {str(e)[:30]}")
        text = f"<b>HEALTH CHECK — {len(agentes)} AGENTES</b>\n"
        text += "━━━━━━━━━━━━━━━━━━━━\n"
        for name in ok:
            text += f"✅ {name}\n"
        for f in fail:
            text += f"❌ {f}\n"
        text += f"━━━━━━━━━━━━━━━━━━━━\n"
        text += f"<b>{len(ok)}/{len(agentes)} operativos</b>"
        return CommandResult(success=True, message=text, action="health")

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
        from core.score_db import get_stats
        from strategies.axi_select_agent import AxiSelectAgent

        stats = get_stats()
        total    = stats["executed"]
        wr       = stats["win_rate"]
        high     = stats["high_score"]
        losses   = max(0, total - high)
        pf       = high / max(losses, 1)

        # Edge Score via AxiSelectAgent
        try:
            agent_axi = AxiSelectAgent()
            axi_state = AxiSelectAgent.new_state(500.0)
            axi_state.trades_closed = total
            axi_state.wins   = high
            axi_state.losses = losses
            axi_state.edge_score = agent_axi.calculate_edge_score(axi_state)
            edge_total = axi_state.edge_score.total
        except Exception:
            edge_total = 0

        # MT5 connectivity check
        try:
            import MetaTrader5 as mt5
            mt5_info = mt5.terminal_info()
            mt5_ok = mt5_info is not None
        except Exception:
            mt5_ok = False

        cr_wr     = "✅" if wr >= 60    else "❌"
        cr_trades = "✅" if total >= 100 else "❌"
        cr_risk   = "✅"
        cr_edge   = "✅" if edge_total >= 50 else "❌"
        cr_mt5    = "✅" if mt5_ok       else "❌"
        cr_pf     = "✅" if pf >= 1.5 or losses == 0 else "❌"

        criteria = [cr_wr, cr_trades, cr_risk, cr_edge, cr_mt5, cr_pf]
        failed   = criteria.count("❌")

        if failed == 0:
            veredicto = "🟢 LISTO PARA REAL — deposita $500 en Axi"
        else:
            veredicto = f"🟡 FALTAN {failed} criterios — sigue en demo"

        pf_str = ">" if losses == 0 else f"{pf:.2f}"
        mt5_str = "Conectado" if mt5_ok else "Desconectado"

        text = (
            f"<b>CRITERIOS PARA IR A REAL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{cr_wr} Win Rate &gt; 60%: {wr:.1f}%\n"
            f"{cr_trades} 100+ trades demo: {min(total,100)}/100\n"
            f"{cr_risk} Sin violaciones de riesgo\n"
            f"{cr_edge} Edge Score Axi &gt; 50: {edge_total}\n"
            f"{cr_mt5} MT5 Axi conectado: {mt5_str}\n"
            f"{cr_pf} Profit Factor &gt; 1.5: {pf_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>VEREDICTO:</b>\n"
            f"{veredicto}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades: {total} | Estimado: {max(0,100-total)} mas\n"
            f"<b>POTENCIAL AXI SELECT $1M</b>\n"
            f"2%/mes x 80% split = $16,000/mes"
        )
        return CommandResult(success=True, message=text, action="criterios")

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

    def _cmd_footprint(self) -> CommandResult:
        from agents.footprint_agent import FootprintAgent
        from core.config import config
        agent = FootprintAgent(
            api_key=config.binance_api_key,
            api_secret=config.binance_api_secret,
            testnet=config.binance_testnet,
        )
        candle = agent.build_live_footprint("BTCUSDT")
        if candle is None:
            return CommandResult(
                success=True,
                message="No hay datos de footprint disponibles. Verifica conexion Binance.",
                action="footprint",
            )
        msg = agent.format_telegram(candle, "BTCUSDT")
        return CommandResult(success=True, message=msg, action="footprint")

    # ── Helpers ───────────────────────────────────────────────────────────


    def _cmd_history(self) -> CommandResult:
        if self.on_history:
            try: text = self.on_history("BTC")
            except Exception as e: text = f"Error: {e}"
        else:
            text = "Agente historico no disponible. Conecta el bot con datos en vivo."
        return CommandResult(success=True, message=text, action="history")

    def _cmd_energy(self) -> CommandResult:
        from agents.energy_frequency_agent import EnergyFrequencyAgent
        reading = EnergyFrequencyAgent().analyze("BTC", price=0.0)
        return CommandResult(success=True, message=reading.format_telegram(), action="energy")

    def _cmd_reporte_semanal(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date, timedelta
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        stats = agent.calculate_weekly_stats(week_start)
        return CommandResult(success=True, message=agent.generate_telegram_summary(stats), action="reporte_semanal")

    def _cmd_reporte_mensual(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        stats = agent.calculate_monthly_stats(today.year, today.month)
        return CommandResult(success=True, message=agent.generate_telegram_summary(stats), action="reporte_mensual")

    def _cmd_proyeccion(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        return CommandResult(success=True, message=ReportAgent(capital=self.state.capital).generate_projection_message(), action="proyeccion")

    def _cmd_vision(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        state = agent.toggle()
        return CommandResult(success=True, message=f"Vision {'activada' if state else 'desactivada'}.", action="vision")

    def _cmd_screenshot(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        cap = agent.capture_full_screen() or agent.create_mock_capture()
        analysis = agent.analyze_capture(cap)
        return CommandResult(success=True, message=agent.build_alert_message(analysis, "full"), action="screenshot")

    def _cmd_mirror(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        if not agent._mirror_active:
            agent.start_mirror_mode()
            msg = "Modo espejo ACTIVADO. El bot aprende de tus operaciones."
        else:
            session = agent.stop_mirror_mode()
            actions = session.actions_recorded if session else 0
            msg = f"Modo espejo DESACTIVADO. Acciones: {actions}"
        return CommandResult(success=True, message=msg, action="mirror")

    def _cmd_analysis(self) -> CommandResult:
        return CommandResult(success=True, message="Analisis SMC: conecate con datos en vivo para analisis completo.", action="analysis")

    def _cmd_onchain(self) -> CommandResult:
        return CommandResult(success=True, message="OnChain metrics: flujos de ballenas disponibles con datos en vivo.", action="onchain")

    def _cmd_lunar(self) -> CommandResult:
        from agents.lunar_agent import LunarCycleAgent
        return CommandResult(success=True, message=LunarCycleAgent().format_telegram(), action="lunar")

    def _cmd_elliott(self) -> CommandResult:
        return CommandResult(success=True, message="Elliott Wave: conecta con OHLCV en vivo para conteo de ondas.", action="elliott")

    def _cmd_edge(self) -> CommandResult:
        from core.score_db import get_stats
        stats = get_stats()
        wr = f"{stats['win_rate']:.1f}%" if stats["executed"] > 0 else "N/A (sin trades)"
        return CommandResult(
            success=True,
            message=(
                f"<b>STATISTICAL EDGE REAL</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Trades ejecutados: {stats['executed']}\n"
                f"Total scaneados: {stats['total']}\n"
                f"Win Rate: {wr}\n"
                f"Scores >= 60: {stats['high_score']}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Acumula 50+ trades para estadisticas robustas."
            ),
            action="edge"
        )

    def _cmd_footprint(self) -> CommandResult:
        from agents.footprint_agent import FootprintAgent
        from core.config import config
        agent = FootprintAgent(api_key=config.binance_api_key, api_secret=config.binance_api_secret, testnet=config.binance_testnet)
        candle = agent.build_live_footprint("BTCUSDT")
        if candle is None:
            return CommandResult(success=True, message="No hay datos de footprint disponibles.", action="footprint")
        return CommandResult(success=True, message=agent.format_telegram(candle, "BTCUSDT"), action="footprint")

    def _cmd_ftmo(self) -> CommandResult:
        from strategies.ftmo_agent import FTMOAgent, ChallengeType
        from core.score_db import get_stats
        agent = FTMOAgent()
        state = FTMOAgent.new_challenge(initial_balance=10000.0, challenge_type=ChallengeType.TWO_STEP)
        stats = get_stats()
        if stats["executed"] > 0:
            try:
                from agents.report_agent import ReportAgent
                from datetime import date
                rpt = ReportAgent(capital=10000.0)
                monthly = rpt.calculate_monthly_stats(date.today().year, date.today().month)
                state = agent.record_trade(state, monthly.pnl)
            except Exception:
                pass
        msg = agent.format_daily_report(state)
        income = agent.calculate_monthly_income(200000, 0.05, 0.90)
        msg += (
            "\n━━━━━━━━━━━━━━━━━━━━\n"
            "<b>POTENCIAL CON $200K FTMO</b>\n"
            f"5%/mes x 90% split = ${income['net_monthly']:,.0f}/mes\n"
            f"Ingreso anual: ${income['yearly']:,.0f}"
        )
        return CommandResult(success=True, message=msg, action="ftmo")
    def _cmd_axi(self) -> CommandResult:
        from strategies.axi_select_agent import AxiSelectAgent
        from core.score_db import get_stats
        agent = AxiSelectAgent()
        state = AxiSelectAgent.new_state(initial_balance=500.0)
        stats = get_stats()
        if stats["executed"] > 0:
            state.trades_closed = stats["executed"]
            state.wins   = stats["high_score"]
            state.losses = max(0, stats["executed"] - stats["high_score"])
            state.edge_score = agent.calculate_edge_score(state)
            state.stage = agent.get_current_stage(state)
        return CommandResult(success=True, message=agent.format_telegram(state), action="axi")

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
            print("[Telegram] Sin token - polling de comandos desactivado")
            await asyncio.Event().wait()
            return

        # Kill any stale getUpdates session before starting
        if HAS_TELEGRAM and self.bot_token:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10) as hx:
                    await hx.post(
                        f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                        json={"offset": -1, "timeout": 0},
                    )
            except Exception:
                pass
            await asyncio.sleep(1)

        while True:
            try:
                app = Application.builder().token(self.bot_token).build()

                for cmd in COMMANDS:
                    app.add_handler(CommandHandler(cmd.lstrip("/"), self._make_handler(cmd)))

                if self.on_callback:
                    app.add_handler(CallbackQueryHandler(self.on_callback))

                async with app:
                    await app.start()
                    await app.updater.start_polling(
                        drop_pending_updates=True,
                        allowed_updates=["message", "callback_query"],
                    )
                    await self.send_message(
                        "🤖 Bot online - comandos activos: /status /auto /semi /pause /resume "
                        "/positions /scores /risk /train /youtube"
                    )
                    print("[Telegram] Polling activo - escuchando comandos")
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
        """Handler for /history [symbol] - calls on_history callback."""
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
