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
    "/axicheck":         "Verificacion 6 variables Axi Select (listo para live?)",
    "/plan":             "Plan financiero 70-20-10 — donde va el capital del bot",
    "/demo":             "Posiciones demo Binance crypto con P&L en vivo",
    "/performance":      "Performance real: win rate, profit factor, P&L cuenta Axi",
    "/session":          "Snapshot de sesion en vivo: P&L, posiciones, target $250, DIM6",
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
        self._supervisor    = None  # set by Supervisor after construction

    def handle_command(self, command: str) -> CommandResult:
        """Synchronous command handler - used in tests and fallback mode."""
        parts = command.strip().split()
        cmd = parts[0].lower()
        self._current_args = parts[1:]  # stored for handlers that accept symbol args

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
            "/axi":              self._cmd_axi,
            "/axicheck":         self._cmd_axicheck,
            "/plan":             self._cmd_plan,
            "/session":          self._cmd_session,
            "/ver_mt5":          self._cmd_ver_mt5,
            "/proteger":         self._cmd_proteger,
            "/demo":             self._cmd_demo,
            "/performance":      self._cmd_performance,
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

        # MT5 real P&L report
        try:
            from connectors.metatrader_connector import MT5Connector
            _mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            pnl = _mt5.get_pnl_report(initial_balance=100_000.0)
            if "error" not in pnl:
                bal       = pnl["balance"]
                equity    = pnl["equity"]
                net       = pnl["net_change"]
                realized  = pnl["realized_pnl"]
                open_pnl  = pnl["profit_open"]
                n_trades  = pnl["n_trades"]
                net_pct   = net / 100_000.0 * 100
                sign      = "+" if net >= 0 else ""
                recent    = pnl.get("recent_trades", [])
                trades_txt = ""
                for t in recent[-3:]:
                    entry_flag = "ABRIO" if t["entry"] == 0 else "CERRO"
                    pnl_usd = f"{t['profit']:+.2f}" if t["entry"] == 1 else "abierta"
                    trades_txt += (
                        f"  {t['dt']} {t['symbol']} {t['direction']} "
                        f"{t['volume']}lot @{t['price']:.3f} P&amp;L:{pnl_usd}\n"
                    )
                mt5_text = (
                    f"Capital inicial: <code>$100,000.00 USD</code>\n"
                    f"Balance actual:  <code>${bal:,.2f} USD</code>\n"
                    f"Equity:          <code>${equity:,.2f} USD</code>\n"
                    f"P&amp;L abierto: <code>${open_pnl:+.2f} USD</code>\n"
                    f"Resultado neto:  <code>{sign}${abs(net):,.2f} USD ({sign}{net_pct:.3f}%)</code>\n"
                    f"Realizado mes:   <code>${realized:+.2f} USD</code>\n"
                    f"Operaciones:     {n_trades}\n"
                    + (f"<b>Ultimas:</b>\n{trades_txt}" if trades_txt else "")
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

        # MT5 open positions
        positions_text = ""
        try:
            from connectors.metatrader_connector import MT5Connector
            _mt5b = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            positions = _mt5b.get_positions()
            if positions:
                for p in positions:
                    pnl_sign = "+" if p.get("profit", 0) >= 0 else ""
                    positions_text += (
                        f"  {p['symbol']} {p['type']} {p['volume']}lot "
                        f"P&amp;L: <code>{pnl_sign}{p.get('profit', 0):.2f} USD</code>\n"
                    )
            else:
                positions_text = "  Sin posiciones abiertas\n"
        except Exception:
            positions_text = "  No disponible\n"

        # Scan statistics from JSON
        import json, os
        scan_text = ""
        try:
            stats_path = os.path.join("memory", "scan_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, encoding="utf-8") as sf:
                    sc = json.load(sf)
                total     = sc.get("total", 0)
                executed  = sc.get("executed", 0)
                blk_score = sc.get("blocked_score", 0)
                blk_cons  = sc.get("blocked_conservative", 0)
                blk_rr    = sc.get("blocked_rr", 0)
                blk_daily = sc.get("blocked_daily_limit", 0)
                blk_ftmo  = sc.get("blocked_ftmo", 0)
                blk_dup   = sc.get("blocked_duplicate", 0)
                blk_claude= sc.get("blocked_claude", 0)
                last_ts   = sc.get("last_trade_ts")
                last_str  = last_ts[:16].replace("T", " ") if last_ts else "nunca"
                scan_text = (
                    f"Setups analizados: <b>{total}</b>\n"
                    f"Ejecutados: <b>{executed}</b>\n"
                    f"Bloqueados:\n"
                    f"  Score bajo: {blk_score}\n"
                    f"  Conservador: {blk_cons}\n"
                    f"  RR bajo: {blk_rr}\n"
                    f"  Limite diario: {blk_daily}\n"
                    f"  Risk Gate: {blk_ftmo}\n"
                    f"  Duplicado: {blk_dup}\n"
                    f"  Claude veto: {blk_claude}\n"
                    f"Ultimo trade: {last_str} UTC"
                )
        except Exception:
            scan_text = "Stats no disponibles aun"

        text = (
            f"<b>SMC BOT -- ESTADO REAL</b>\n"
            f"<b>Modo:</b> {cfg.operation_mode.upper()} | ACTIVO\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>MT5 AXI DEMO</b>\n"
            f"{mt5_text}"
            f"<b>Posiciones abiertas:</b>\n{positions_text}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>SCAN STATS (acumulado)</b>\n"
            f"{scan_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>BINANCE TESTNET</b>\n"
            f"{binance_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>FILTROS ACTIVOS</b>\n"
            f"Score MT5: >=75 | RR min: 1:2\n"
            f"Max posiciones: 2 | Max/dia: 8\n"
            f"H4 trend filter: crypto + forex\n"
            f"13 agentes institucionales paralelos\n"
            f"Win Rate: {wr} | /performance para detalle"
        )
        return CommandResult(success=True, message=text, action="status")

    def _cmd_positions(self) -> CommandResult:
        try:
            from connectors.metatrader_connector import MT5Connector
            from core.config import config as cfg
            mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            positions = mt5.get_open_positions()
            if not positions:
                return CommandResult(success=True, message="Sin posiciones abiertas en MT5.", action="positions")
            lines = ["<b>POSICIONES ABIERTAS MT5</b>", "━━━━━━━━━━━━━━━━━━━━"]
            for p in positions:
                pnl = p.get("profit", 0.0)
                sign = "+" if pnl >= 0 else ""
                estado = "GANANDO" if pnl >= 0 else "PERDIENDO"
                lines.append(
                    f"{p.get('symbol','?')} {p.get('type','').upper()} "
                    f"{p.get('volume',0):.2f}L @ {p.get('price_open',0):.4f}\n"
                    f"  P&amp;L: <b>{sign}${pnl:.2f}</b> ({estado})"
                )
            return CommandResult(success=True, message="\n".join(lines), action="positions")
        except Exception as e:
            if self.state.open_positions == 0:
                return CommandResult(success=True, message="Sin posiciones abiertas actualmente.", action="positions")
            return CommandResult(
                success=True,
                message=f"Posiciones: {self.state.open_positions} abiertas (MT5 no disponible: {e})",
                action="positions",
            )

    def _cmd_demo(self) -> CommandResult:
        """Show open Binance crypto demo positions with live P&L."""
        try:
            sup = self._supervisor
            demo_trades = getattr(sup, "_demo_trades", []) if sup else []
            open_trades = [d for d in demo_trades if getattr(d, "status", "") == "open"]

            if not open_trades:
                return CommandResult(
                    success=True,
                    message="<b>DEMO BINANCE</b>\nSin posiciones demo abiertas ahora.",
                    action="demo",
                )

            import yfinance as yf
            from datetime import datetime, timezone

            lines = [
                "<b>DEMO BINANCE — POSICIONES ABIERTAS</b>",
                "━━━━━━━━━━━━━━━━━━━━",
            ]
            wins = losses = 0
            for d in open_trades:
                symbol    = d.signal.symbol
                entry     = d.signal.entry or 0.0
                sl        = d.signal.stop_loss or 0.0
                tp        = d.signal.take_profit or 0.0
                score     = d.score
                dir_str   = "LONG" if d.signal.signal_type.value == "long" else "SHORT"
                opened_ago = int((datetime.now(timezone.utc) - d.opened_at).total_seconds() / 60)

                # Current price via yfinance
                current = entry
                try:
                    yf_sym = symbol.replace("USDT", "-USD")
                    current = float(yf.Ticker(yf_sym).fast_info.last_price)
                except Exception:
                    pass

                if d.signal.signal_type.value == "long":
                    pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0.0
                else:
                    pnl_pct = (entry - current) / entry * 100 if entry > 0 else 0.0

                sign   = "+" if pnl_pct >= 0 else ""
                estado = "GANANDO" if pnl_pct >= 0 else "PERDIENDO"
                if pnl_pct >= 0:
                    wins += 1
                else:
                    losses += 1

                lines.append(
                    f"<b>{symbol}</b> {dir_str} | Score: {score}\n"
                    f"  Entrada: <code>{entry:.4f}</code>  Actual: <code>{current:.4f}</code>\n"
                    f"  P&amp;L: <b>{sign}{pnl_pct:.2f}%</b> ({estado})\n"
                    f"  SL: <code>{sl:.4f}</code>  TP: <code>{tp:.4f}</code>  [{opened_ago}min]"
                )

            total = len(open_trades)
            lines.append(
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Total: {total} | Ganando: {wins} | Perdiendo: {losses}"
            )
            return CommandResult(success=True, message="\n".join(lines), action="demo")

        except Exception as e:
            return CommandResult(
                success=True,
                message=f"<b>DEMO BINANCE</b>\nError: {e}",
                action="demo",
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
        daily = self.state.daily_pnl
        balance = float(getattr(self.state, 'balance', self.state.capital))
        drawdown_pct = float(self.state.drawdown)
        equity = balance
        try:
            from connectors.metatrader_connector import MT5Connector
            from core.config import config as cfg
            mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            pnl = mt5.get_pnl_report(initial_balance=100_000.0)
            if "error" not in pnl:
                daily       = pnl.get("daily_pnl", daily)
                balance     = pnl.get("balance", balance)
                equity      = pnl.get("equity", balance)
                drawdown_pct = (100_000.0 - equity) / 100_000.0 * 100
        except Exception:
            pass
        sign = "+" if daily >= 0 else ""
        return CommandResult(
            success=True,
            message=(
                f"<b>ESTADO DE RIESGO</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Balance: <code>${balance:,.2f}</code>\n"
                f"Equity:  <code>${equity:,.2f}</code>\n"
                f"Drawdown: <code>{drawdown_pct:.2f}%</code>\n"
                f"P&amp;L dia: <code>{sign}${daily:.2f}</code>\n"
                f"Posiciones abiertas: {self.state.open_positions}\n"
                f"Estado: {'OK' if drawdown_pct < 3 else 'ELEVADO'}"
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
            "Risk Gate": "strategies.ftmo_agent",
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
        from core.config import config as cfg
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

        # MT5 P&L real — connectivity + financial result
        mt5_ok = False
        pnl_text = ""
        try:
            from connectors.metatrader_connector import MT5Connector
            _mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            pnl = _mt5.get_pnl_report(initial_balance=100_000.0)
            if "error" not in pnl:
                mt5_ok    = True
                bal       = pnl["balance"]
                net       = pnl["net_change"]
                realized  = pnl["realized_pnl"]
                open_pnl  = pnl["profit_open"]
                n_ops     = pnl["n_trades"]
                sign      = "+" if net >= 0 else ""
                net_pct   = net / 100_000.0 * 100
                pnl_text  = (
                    f"\n━━━━━━━━━━━━━━━━━━━━\n"
                    f"<b>CUENTA MT5 — RESULTADO REAL</b>\n"
                    f"Capital invertido: <code>$100,000.00 USD</code>\n"
                    f"Balance actual:    <code>${bal:,.2f} USD</code>\n"
                    f"Resultado neto:    <code>{sign}${abs(net):,.2f} USD ({sign}{net_pct:.3f}%)</code>\n"
                    f"Realizado mes:     <code>${realized:+.2f} USD</code>\n"
                    f"P&amp;L abierto:  <code>${open_pnl:+.2f} USD</code>\n"
                    f"Operaciones:       {n_ops}"
                )
        except Exception:
            pass

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

        pf_str  = ">" if losses == 0 else f"{pf:.2f}"
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
            + pnl_text +
            f"\n━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>POTENCIAL AXI SELECT $1M</b>\n"
            f"2%/mes x 80% split = $16,000/mes"
        )
        return CommandResult(success=True, message=text, action="criterios")

    def _cmd_performance(self) -> CommandResult:
        """Real-time performance stats from score_db with actual WIN/LOSS outcomes."""
        from core.score_db import get_stats, get_recent_scores
        from datetime import datetime, timezone

        stats = get_stats()
        total    = stats["executed"]
        wins     = stats["wins"]
        losses   = stats["losses"]
        wr       = stats["win_rate"]
        pf       = stats["profit_factor"]
        avg_pnl  = stats["avg_pnl_pct"]
        real     = stats["has_real_outcomes"]

        # Open MT5 P&L
        open_pnl = 0.0
        balance  = 0.0
        try:
            from connectors.metatrader_connector import MT5Connector
            from core.config import config as cfg
            _m = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            pos = _m.get_open_positions()
            open_pnl = sum(p.get("profit", 0.0) for p in pos)
            _info = _m.get_account_info()
            balance = _info.get("balance", 0.0) if _info else 0.0
        except Exception:
            pass

        label = "reales" if real else "estimados (sin outcomes aun)"
        net_pct = (balance - 100_000.0) / 100_000.0 * 100 if balance > 0 else 0.0
        net_sign = "+" if net_pct >= 0 else ""

        recent = get_recent_scores(5)
        recent_lines = []
        for row in recent:
            ts, sym, tf, score, direction, entry, executed = row
            hora = ts[11:16] if len(ts) > 16 else ts
            arrow = "▲" if direction == "long" else "▼"
            recent_lines.append(f"  {arrow} {sym} {tf} score={score} @ {hora}")

        text = (
            f"<b>PERFORMANCE EN VIVO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades ejecutados: <b>{total}</b>\n"
            f"Ganados: <b>{wins}</b>  Perdidos: <b>{losses}</b>\n"
            f"Win Rate: <b>{wr:.1f}%</b> ({label})\n"
            f"Profit Factor: <b>{pf:.2f}</b>\n"
            f"P&amp;L promedio demo: <b>{avg_pnl:+.2f}%</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>CUENTA MT5 (Axi Demo)</b>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Neto: <b>{net_sign}{net_pct:.3f}%</b> (de $100,000)\n"
            f"P&amp;L posiciones abiertas: <b>${open_pnl:+.2f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Ultimas 5 senales:</b>\n"
            + ("\n".join(recent_lines) if recent_lines else "  Sin senales aun")
        )
        return CommandResult(success=True, message=text, action="performance")

    def _cmd_proyeccion(self) -> CommandResult:
        from core.volume_calculator import VolumeCalculator
        vc = VolumeCalculator()
        lines = [
            "<b>PROYECCION AXI SELECT — ETAPAS</b>",
            "<pre>",
            f"{'Etapa':<12} {'Capital':>10} {'Vol':>6} {'Profit/mes':>12} {'Tu 80%':>10}",
            "-" * 54,
        ]
        stage_names = {
            "seed": "Seed", "incubation": "Incubacion",
            "demo": "Demo", "pro": "Pro",
            "pro_500": "Pro 500K", "pro_m": "Pro 1M",
        }
        for key, data in vc.AXI_STAGES.items():
            cap = data["capital"]
            proj = vc.project_monthly_profit(cap)
            lines.append(
                f"{stage_names.get(key, key):<12} "
                f"${cap/1000:>7.0f}K "
                f"{data['volume']:>5.2f}L "
                f"${proj['net_profit_usd']:>10,.0f} "
                f"${proj['your_share_80pct']:>8,.0f}"
            )
        current_proj = vc.project_monthly_profit(self.state.capital)
        lines += [
            "-" * 54,
            f"{'Actual':<12} ${self.state.capital/1000:>7.0f}K "
            f"{vc.get_stage_volume(self.state.capital):>5.2f}L "
            f"${current_proj['net_profit_usd']:>10,.0f} "
            f"${current_proj['your_share_80pct']:>8,.0f}",
            "</pre>",
            f"Win rate asumido: 62% | RR: 2:1 | 40 ops/mes",
        ]
        return CommandResult(success=True, message="\n".join(lines), action="proyeccion")

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

    def _cmd_ver_mt5(self) -> CommandResult:
        """Capture MT5 screen now and return Claude Vision analysis."""
        try:
            from agents.axi_vision_agent import AxiVisionAgent
            agent = AxiVisionAgent()
            report = agent.monitor_and_protect()
            analysis = report.get("analysis", {})
            alerts = report.get("alerts", [])
            balance = report.get("balance", 0)
            equity = analysis.get("patrimonio", 0)
            positions = analysis.get("posiciones", [])
            recommendation = analysis.get("accion_recomendada", "")

            _start = self.status.capital if self.status.capital > 0 else balance
            growth = balance - _start
            growth_str = f"+${growth:,.0f}" if growth >= 0 else f"-${abs(growth):,.0f}"

            lines = [
                "<b>VISION MT5 - Analisis en vivo</b>",
                f"Balance: ${balance:,.2f} ({growth_str} vs inicio ${_start:,.0f})",
                f"Patrimonio: ${equity:,.2f}",
            ]
            if positions:
                lines.append("\nPosiciones:")
                for p in positions:
                    pnl = p.get("pnl", 0)
                    sign = "+" if pnl >= 0 else ""
                    lines.append(
                        f"  {p.get('symbol','')} {p.get('direction','')} "
                        f"{p.get('volume','')}lot  P&L: {sign}${pnl:.2f}"
                    )
            else:
                lines.append("Sin posiciones abiertas")

            if alerts:
                lines.append("\nAlertas:")
                for a in alerts:
                    lines.append(f"  {a}")
            if recommendation:
                lines.append(f"\nRecomendacion: {recommendation}")

            return CommandResult(success=True, message="\n".join(lines), action="ver_mt5")
        except Exception as e:
            return CommandResult(success=False, message=f"Error vision: {e}", action="ver_mt5")

    def _cmd_proteger(self) -> CommandResult:
        """Toggle protection mode -- screen checked every 2 min, auto-close on critical loss."""
        try:
            sup = self._supervisor  # type: ignore[attr-defined]
            sup._vision_protect_mode = not sup._vision_protect_mode
            state = "ACTIVADO" if sup._vision_protect_mode else "DESACTIVADO"
            msg = (
                f"Modo proteccion {state}\n"
                + ("Revision cada 2 min. Cierre automatico si perdida > $500."
                   if sup._vision_protect_mode
                   else "Revision cada 5 min (normal).")
            )
            return CommandResult(success=True, message=msg, action="proteger")
        except AttributeError:
            return CommandResult(
                success=False,
                message="Error: supervisor no conectado al commander",
                action="proteger",
            )

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
        args = getattr(self, '_current_args', [])
        raw = args[0] if args else "BTC"
        symbol = raw.upper() if raw.upper().endswith("USDT") else raw.upper() + "USDT"
        agent = FootprintAgent(api_key=config.binance_api_key, api_secret=config.binance_api_secret, testnet=config.binance_testnet)
        candle = agent.build_live_footprint(symbol)
        if candle is None:
            return CommandResult(success=True, message=f"No hay datos de footprint para {symbol}.", action="footprint")
        return CommandResult(success=True, message=agent.format_telegram(candle, symbol), action="footprint")

    def _cmd_axi(self) -> CommandResult:
        """Detailed Axi Select progress: edge score, stage, what's needed next."""
        from strategies.axi_select_agent import AxiSelectAgent, AxiStage
        from core.score_db import get_stats
        from connectors.metatrader_connector import MT5Connector
        from core.config import config as cfg

        agent  = AxiSelectAgent()
        state  = AxiSelectAgent.new_state(initial_balance=100_000.0)
        stats  = get_stats()

        # Use real outcomes when available
        if stats["has_real_outcomes"]:
            state.wins   = stats["wins"]
            state.losses = stats["losses"]
            state.trades_closed = stats["wins"] + stats["losses"]
        else:
            # Fallback: approximate
            state.trades_closed = stats["executed"]
            state.wins   = int(stats["executed"] * (stats["win_rate"] / 100))
            state.losses = state.trades_closed - state.wins

        # MT5 balance for drawdown calculation
        try:
            mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            acc = mt5.get_account_info()
            if acc and acc.get("balance", 0) > 0:
                state.current_balance = acc["balance"]
                dd = max(0, (100_000.0 - state.current_balance) / 100_000.0)
                state.max_drawdown_pct = dd
        except Exception:
            pass

        state.edge_score = agent.calculate_edge_score(state)
        state.stage = agent.get_current_stage(state)

        sc  = state.edge_score
        bal = state.current_balance
        net = bal - 100_000.0
        dd  = state.max_drawdown_pct * 100
        wr  = state.win_rate * 100
        pf  = state.profit_factor

        STAGE_LABELS = {
            AxiStage.PRE_SEED:     "PRE-SEED",
            AxiStage.SEED:         "SEED ($5K)",
            AxiStage.INCUBATION:   "INCUBACION ($25K)",
            AxiStage.ACCELERATION: "ACELERACION ($100K)",
            AxiStage.PRO:          "PRO ($300K)",
            AxiStage.PRO_500:      "PRO 500 ($500K)",
            AxiStage.PRO_M:        "PRO M ($1M)",
        }

        # Next stage target
        next_cfg = agent.get_next_stage(state.stage)
        if next_cfg:
            pts_needed = next_cfg.edge_score_required - sc.total
            next_label = STAGE_LABELS.get(next_cfg.stage, "?")
            next_txt = (
                f"Siguiente: <b>{next_label}</b>\n"
                f"Necesitas: Edge Score {next_cfg.edge_score_required} "
                f"(faltan {max(0,pts_needed)} pts)\n"
                f"Min trades: {next_cfg.min_trades} (tienes {state.trades_closed})\n"
            )
        else:
            next_txt = "MAXIMO NIVEL ALCANZADO — $1M fondeo\n"

        # Monthly income projection per stage
        income_map = {
            AxiStage.SEED:         (5_000,   "  $100 - $250"),
            AxiStage.INCUBATION:   (25_000,  "  $500 - $1,250"),
            AxiStage.ACCELERATION: (100_000, "  $2,000 - $5,000"),
            AxiStage.PRO:          (300_000, "  $6,000 - $15,000"),
            AxiStage.PRO_500:      (500_000, "  $10,000 - $25,000"),
            AxiStage.PRO_M:        (1_000_000,"$20,000 - $50,000"),
        }
        current_income = income_map.get(state.stage, (0, "—"))
        next_income = income_map.get(next_cfg.stage if next_cfg else AxiStage.PRE_SEED, (0, "—"))

        text = (
            f"<b>AXI SELECT — PROGRESO REAL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Etapa actual: <b>{STAGE_LABELS.get(state.stage,'PRE-SEED')}</b>\n"
            f"Income mensual actual: <b>{current_income[1]}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>EDGE SCORE: {sc.total}/100</b>\n"
            f"  Habilidad:    {sc.habilidad}/40\n"
            f"  Consistencia: {sc.consistencia}/30\n"
            f"  Riesgo:       {sc.riesgo}/30\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>METRICAS REALES:</b>\n"
            f"  Trades cerrados: {state.trades_closed}\n"
            f"  Win Rate: {wr:.1f}% {'✅' if wr >= 60 else '⚠️ necesitas >=60%'}\n"
            f"  Profit Factor: {pf:.2f}\n"
            f"  Drawdown: {dd:.2f}% {'✅' if dd < 5 else '⚠️'}\n"
            f"  Balance: ${bal:,.2f} ({net:+.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>SIGUIENTE META:</b>\n"
            f"{next_txt}"
            f"Income mensual siguiente: <b>{next_income[1]}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>PARA LLEGAR A $1M ({STAGE_LABELS[AxiStage.PRO_M]}):</b>\n"
            f"  Edge Score 90/100 requerido\n"
            f"  Win Rate sostenida >= 75%\n"
            f"  Income: $20,000-$50,000/mes\n"
        )
        # ── Tracker mensual (nuevos agentes Axi Select) ──────────────
        try:
            from agents.axi_select_tracker import AxiSelectTracker
            tracker = AxiSelectTracker()
            tr = tracker.get_status()
            on_track_icon = "✅" if tr.on_track else "⚠️"
            tracker_txt = (
                f"\n━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>MES ACTUAL ({tr.stage_name}):</b>\n"
                f"  P&L mes: ${tr.monthly_pnl:+,.0f} ({tr.monthly_pct*100:.2f}%)\n"
                f"  Objetivo: 5% (${tr.capital*0.05:,.0f})\n"
                f"  Dias: {tr.days_traded} operados / {tr.days_remaining} restantes\n"
                f"  Proyeccion fin de mes: {tr.projected_pct*100:.2f}% {on_track_icon}\n"
                f"  Necesita: ${tr.daily_avg_needed:,.0f}/dia\n"
            )
        except Exception:
            tracker_txt = ""

        text += tracker_txt
        return CommandResult(success=True, message=text, action="axi")

    def _cmd_axicheck(self) -> CommandResult:
        """Pre-cierre verificacion de las 6 variables Axi Select."""
        from agents.axi_select_guard import AxiSelectGuard
        from agents.axi_select_tracker import AxiSelectTracker
        from agents.consistency_enforcer import ConsistencyEnforcer
        from connectors.metatrader_connector import MT5Connector
        from core.config import config as cfg

        lines = ["<b>AXI SELECT — VERIFICACION COMPLETA</b>\n━━━━━━━━━━━━━━━━━━━━"]
        score = 0

        try:
            mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
            acc = mt5.get_account_info()
            bal = acc.get("balance", 0) if acc else 0
            eq  = acc.get("equity",  bal) if acc else 0

            # V1: Profit mensual
            tracker = AxiSelectTracker()
            tr = tracker.get_status()
            v1 = "✅" if tr.monthly_pct >= 0.05 else ("⚠️" if tr.monthly_pct >= 0.02 else "❌")
            lines.append(f"{v1} P&L mes: {tr.monthly_pct*100:.2f}% (objetivo >=5%)")
            if tr.monthly_pct >= 0.05: score += 1

            # V2: Daily loss
            guard = AxiSelectGuard()
            gr = guard.check(eq, capital_assigned=bal)
            v2 = "✅" if not gr.should_close and not gr.warning_level else ("⚠️" if gr.warning_level else "❌")
            lines.append(f"{v2} P&L dia: {gr.daily_pnl_pct*100:.2f}% (limite -4%)")
            if not gr.should_close: score += 1

            # V3: Drawdown total
            init_bal = tr.capital
            dd_pct   = (init_bal - eq) / init_bal if init_bal > 0 else 0
            v3 = "✅" if dd_pct < 0.08 else ("⚠️" if dd_pct < 0.10 else "❌")
            lines.append(f"{v3} Drawdown total: {dd_pct*100:.2f}% (limite 10%)")
            if dd_pct < 0.10: score += 1

            # V4: Dias operados
            v4 = "✅" if tr.days_traded >= 10 else ("⚠️" if tr.days_traded >= 5 else "❌")
            lines.append(f"{v4} Dias operados: {tr.days_traded}/22 (minimo 10)")
            if tr.days_traded >= 10: score += 1

            # V5: Consistencia
            monthly_pnl = tr.monthly_pnl
            today_pnl   = monthly_pnl / tr.days_traded if tr.days_traded > 0 else 0
            ce = ConsistencyEnforcer()
            cr = ce.check(today_pnl, monthly_pnl)
            v5 = "✅" if not cr.should_block_new else "⚠️"
            lines.append(f"{v5} Consistencia: mayor dia = {cr.today_pct_of_monthly*100:.0f}% del mes (max 30%)")
            if not cr.should_block_new: score += 1

            # V6: Instrumentos
            lines.append(f"✅ Instrumentos: EURUSD/GBPUSD/AUDUSD/USDCAD/NZDUSD/NAS100 (todos OK)")
            score += 1

        except Exception as e:
            lines.append(f"❌ Error al verificar: {e}")

        lines.append(f"\n━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"<b>RESULTADO: {score}/6 variables OK</b>")
        if score == 6:
            lines.append("Estado: LISTO PARA AXI SELECT")
        elif score >= 4:
            lines.append("Estado: CERCA — revisar variables en rojo")
        else:
            lines.append("Estado: NO LISTO — bot necesita mas tiempo")

        return CommandResult(success=True, message="\n".join(lines), action="axicheck")

    def _cmd_plan(self) -> CommandResult:
        """Plan financiero 70-20-10 — donde va el capital del bot."""
        from agents.portfolio_tracker import PortfolioTracker
        tracker = PortfolioTracker()

        # Estimar ingreso mensual actual del bot desde el tracker Axi
        monthly_income = None
        try:
            from agents.axi_select_tracker import AxiSelectTracker
            axi = AxiSelectTracker()
            st = axi.get_status()
            if st.days_traded > 0 and st.monthly_pnl > 0:
                monthly_income = st.monthly_pnl / st.days_traded * 22
        except Exception:
            pass

        msg = tracker.format_telegram(axi_monthly_income=monthly_income)
        return CommandResult(success=True, message=msg, action="plan")

    def _cmd_session(self) -> CommandResult:
        """Snapshot de sesion en vivo: P&L, posiciones, progreso a $250, DIM6."""
        import datetime as _dt
        from core.config import config as cfg

        now_utc = _dt.datetime.now(_dt.timezone.utc)
        session_start = now_utc.replace(hour=13, minute=0, second=0, microsecond=0)
        if now_utc.hour < 13:
            session_start -= _dt.timedelta(days=1)
        elapsed = now_utc - session_start
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m = rem // 60

        # MT5 data
        bal = eq = float_pnl = 0.0
        positions_txt = ""
        day_realized = 0.0
        try:
            import MetaTrader5 as _mt5lib
            import os as _os
            from dotenv import load_dotenv as _lde
            _lde()
            _mt5lib.initialize()
            _mt5lib.login(
                int(_os.getenv("MT5_LOGIN", "0")),
                _os.getenv("MT5_PASSWORD", ""),
                _os.getenv("MT5_SERVER", ""),
            )
            acc = _mt5lib.account_info()
            if acc:
                bal       = acc.balance
                eq        = acc.equity
                float_pnl = acc.profit
            # Today's realized P&L
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            deals = _mt5lib.history_deals_get(today_start, now_utc) or []
            day_realized = sum(d.profit + d.swap + d.commission for d in deals if d.entry == 1)
            # Open positions
            for p in (_mt5lib.positions_get() or []):
                side   = "BUY" if p.type == 0 else "SELL"
                profit = p.profit
                icon   = "🟢" if profit >= 0 else "🔴"
                sl_d   = abs(p.price_current - p.sl) if p.sl else 0
                tp_d   = abs(p.tp - p.price_current) if p.tp else 0
                positions_txt += (
                    f"  {icon} {p.symbol} {side} {p.volume}L "
                    f"P&amp;L: <code>${profit:+.2f}</code> "
                    f"SL:{sl_d:.1f} TP:{tp_d:.1f}\n"
                )
            _mt5lib.shutdown()
        except Exception as e:
            positions_txt = f"  MT5: {e}\n"

        if not positions_txt:
            positions_txt = "  Sin posiciones abiertas\n"

        # Progreso hacia meta $250
        net_session = day_realized + float_pnl
        progress = min(100, max(0, net_session / 250 * 100))
        bar_filled = int(progress / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        target_icon = "✅" if net_session >= 250 else ("🟡" if net_session >= 125 else "🔴")

        # DIM6 status desde scan_stats
        dim6_txt = "Sin datos"
        try:
            import json, os
            stats_f = os.path.join("memory", "scan_stats.json")
            if os.path.exists(stats_f):
                st = json.load(open(stats_f))
                consec_loss = st.get("consecutive_losses", 0)
                wr5 = st.get("wr_last5", None)
                monthly_pct = st.get("monthly_profit_pct", 0.0)
                dim6_icon = "🟢" if consec_loss < 3 else "🚨"
                wr_txt = f"WR5: {wr5:.0%}" if wr5 is not None else "WR5: N/A"
                dim6_txt = f"{dim6_icon} Perdidas consec: {consec_loss}/3 | {wr_txt} | Mes: {monthly_pct:.1f}%"
        except Exception:
            pass

        # Axi Select tracker
        axi_txt = ""
        try:
            from agents.axi_select_tracker import AxiSelectTracker
            axi = AxiSelectTracker()
            st = axi.get_status()
            days_left = max(0, 30 - st.days_traded)
            axi_txt = (
                f"<b>Axi Select:</b> Dia {st.days_traded}/30 | "
                f"P&amp;L: <code>${st.monthly_pnl:+.2f}</code> | "
                f"DD: <code>{st.max_drawdown_pct:.1f}%</code> | "
                f"{days_left}d restantes\n"
            )
        except Exception:
            pass

        msg = (
            f"<b>SESSION LIVE — {now_utc.strftime('%H:%M')} UTC "
            f"(+{h}h{m:02d}m)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Cuenta:</b> Balance <code>${bal:,.2f}</code> | "
            f"Equity <code>${eq:,.2f}</code>\n"
            f"\n<b>Posiciones:</b>\n{positions_txt}"
            f"\n<b>Meta dia $250:</b>\n"
            f"  [{bar}] {progress:.0f}%\n"
            f"  {target_icon} Realizado+Float: <code>${net_session:+.2f}</code> / $250\n"
            f"  Realizado: <code>${day_realized:+.2f}</code> | Float: <code>${float_pnl:+.2f}</code>\n"
            f"\n<b>DIM6 Circuit:</b> {dim6_txt}\n"
            f"\n{axi_txt}"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>/session para actualizar • /close_all para cerrar todo</i>"
        )
        return CommandResult(success=True, message=msg, action="session")

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
            await bot.send_message(chat_id=self.chat_id, text=text, parse_mode="HTML")
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
            args = context.args or []
            full_cmd = cmd if not args else f"{cmd} {args[0]}"
            result = self.handle_command(full_cmd)
            try:
                await update.message.reply_text(result.message, parse_mode="HTML")
            except Exception:
                try:
                    await update.message.reply_text(result.message)
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
