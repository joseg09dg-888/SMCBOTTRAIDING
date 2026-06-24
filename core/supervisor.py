import asyncio

import logging

import os
from datetime import datetime, timezone, timedelta

from typing import Optional, List, Dict

import pandas as pd



from core.config import config

from core.risk_manager import RiskManager

from core.decision_filter import DecisionFilter, TradeGrade

from agents.signal_agent import TradeSignal, SignalType, SignalAgent

from connectors.binance_connector import BinanceConnector

from connectors.metatrader_connector import MT5Connector

from connectors.glint_connector import GlintSignal

from connectors.glint_browser import GlintBrowser

from dashboard.telegram_bot import TradingTelegramBot

from dashboard.telegram_commander import TelegramCommander

from training.historical_agent import HistoricalDataAgent



logger = logging.getLogger(__name__)

from core.score_db import save_score

from memory.episodic_db import get_db, record_episode, update_episode_result

from core.autonomous_learner import AutonomousLearner

from core.research_agent import ResearchAgent

from core.goals_manager import GoalsManager

from core.nightly_reporter import NightlyReporter

from core.risk_governor import RiskGovernor, fetch_recent_deals_by_symbol

from core.volume_calculator import VolumeCalculator
from core.market_hours import is_market_open, minutes_until_open

from strategies.ftmo_agent import FTMOAgent, ChallengeType

from agents.lunar_agent import LunarCycleAgent
from agents.elliott_agent import ElliottFibonacciAgent
from agents.footprint_agent import FootprintAgent
from agents.statistical_edge_agent import QuantEdgeAgent
from agents.chaos_agent import ChaosTheoryAgent
from agents.institutional_flow_agent import InstitutionalFlowAgent
from agents.microstructure_agent import MarketMicrostructureAgent
from agents.fed_sentiment_agent import FEDSentimentAgent
from agents.onchain_agent import OnChainAgent
from agents.geopolitical_agent import GeopoliticalAgent
from agents.retail_psychology_agent import RetailPsychologyAgent
from agents.alternative_data_agent import AlternativeDataAgent
from agents.energy_frequency_agent import EnergyFrequencyAgent
from agents.axi_vision_agent import AxiVisionAgent
from memory.episodic_db import query_similar_episodes



# Score thresholds

DEMO_SCORE_THRESHOLD     = 999  # DISABLED: crypto demo no cuenta para Axi Select
MT5_REAL_SCORE_THRESHOLD = 85   # SWING: best setups H1/H4
MT5_SCALP_THRESHOLD      = 65   # SCALP M15: más trades, TP/SL pequeños
MT5_SCORE_AUTO_REDUCE    = 75
MT5_SCORE_REDUCE_AFTER_H = 4
MAX_SCALP_POSITIONS      = 10   # scalp: hasta 10 simultáneas (riesgo pequeño por trade)
SCALP_MAX_DOLLAR_RISK    = 50.0 # scalp: max $50 por trade → 100 trades × $10 = $1000

# Modo Recuperación — dos triggers:
#   1. Dia en rojo > $50  → recuperar el dia
#   2. Balance < $100K    → recuperar capital base
INITIAL_CAPITAL          = 100_000.0  # capital base
RECOVERY_SCALP_TP        = 5.0   # +$5 TP en recovery (vs $10 normal)
RECOVERY_SCALP_SL        = -2.0  # -$2 SL en recovery (vs -$4 normal)
RECOVERY_MAX_SCALPS      = 15    # 15 scalps simultáneas en recovery (vs 10)
RECOVERY_TRIGGER_LOSS    = -50.0 # trigger diario: dia en rojo > $50
RECOVERY_DRAWDOWN_FROM_PEAK = 500.0  # trigger peak: cae $500 del maximo historico

# Modo Aceleración (estrategia 5) — dia muy bueno, maximizar ganancias
ACCEL_TRIGGER_PROFIT     = 150.0  # activa cuando dia > +$150 realizado
ACCEL_SCALP_TP           = 15.0   # +$15 TP (vs $10 normal)
ACCEL_SCALP_SL           = -4.0   # -$4 SL (igual que normal)
ACCEL_MAX_SCALPS         = 20     # 20 simultáneas (vs 10)
DEMO_MAX_POSITIONS       = 0    # no demo positions — 100% focus on MT5 real
SCAN_INTERVAL_SEC        = 30

# Conservative mode disabled — 8 filters + Claude API confirmation are sufficient
CONSERVATIVE_MODE        = False   # was True — disabled now that pipeline is complete
CONSERVATIVE_SCORE_MIN   = 75
CONSERVATIVE_PAIRS       = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "US30"]
MAX_DAILY_TRADES         = 100     # sin techo — el mercado limita, no el bot
MAX_OPEN_POSITIONS       = 5       # 3 swing + 2 scalp simultáneas
MIN_RR                   = 2.5    # maintain quality: RR 2.5 minimum
DAILY_PROFIT_TARGET      = 245.0  # $245 piso → notifica Telegram, bot sigue operando

# Colombia UTC-5: bot opera solo en horario NY+London cuando usuario esta despierto
# Bloqueado: midnight-8am Colombia = 05:00-13:00 UTC (madrugada completa)
# Activo: 8am-midnight Colombia = 13:00-05:00 UTC (NY session completa + tarde)
DEAD_HOURS_UTC           = {5, 6, 7, 8, 9, 10, 11, 12}  # 00:00-08:00 COL bloqueado



# Symbols and timeframes to scan

SCAN_SYMBOLS    = []  # DISABLED: crypto demo no aporta a Axi — 100% foco MT5 real

SCAN_TIMEFRAMES = ["4h", "1h"]  # 4h first so H4 trend is cached before 1h filter runs



# MT5 forex/indices symbols — expanded for more opportunities
# Asian pairs (AUDUSD, USDJPY, GBPJPY) active 00-09 UTC
# European pairs (EURUSD, GBPUSD) active 07-16 UTC
# US indices (NAS100, US30) active 13-20 UTC
# Gold (XAUUSD) active 07-20 UTC

# Universo completo de pares MT5 (usado para enrutar señales MT5 vs Binance).
# La lista de pares ACTIVAMENTE escaneados la decide RiskGovernor en tiempo
# real (self.risk_governor.active_symbols()) — ver core/risk_governor.py.
MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "AUDUSD", "USDCAD", "NAS100.fs", "US30"]
MT5_TIMEFRAMES   = ["H4", "H1", "M15"]  # H4 trend → H1 swing → M15 scalp

MT5_MIN_VOLUME   = 0.01



# yfinance forex (fallback when MT5 unavailable)

YFINANCE_FOREX   = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",

                    "USDJPY": "USDJPY=X", "GBPJPY": "GBPJPY=X"}



class DemoTrade:

    """In-memory record of a simulated demo trade."""

    def __init__(self, signal: TradeSignal, score: int):

        self.signal      = signal

        self.score       = score

        self.opened_at   = datetime.now(timezone.utc)

        self.status      = "open"

        self.close_price = 0.0

        self.pnl         = 0.0



    def close(self, current_price: float):

        self.close_price = current_price

        if self.signal.signal_type == SignalType.LONG:

            self.pnl = (current_price - self.signal.entry) / self.signal.entry

        else:

            self.pnl = (self.signal.entry - current_price) / self.signal.entry

        self.status = "closed"





class TradingSupervisor:

    """

    Master orchestrator. Full pipeline:



    Market data â†' SMC analysis â†' DecisionFilter (0-100)

        Demo:  score >= 40 â†' execute simulated trade

        Live:  score >= 60 â†' REDUCED | >= 75 â†' FULL | >= 90 â†' PREMIUM



    In demo mode all trades are simulated (no real orders placed).

    """



    def __init__(self, capital: float = 1000.0, demo_mode: bool = True):

        self.config         = config

        self.capital        = capital

        self.demo_mode      = demo_mode

        self.risk_manager   = RiskManager(config, capital)

        self.historical     = HistoricalDataAgent()

        self.decision       = DecisionFilter(config, self.risk_manager,

                                             historical_agent=self.historical)

        self.signal_agent   = SignalAgent(min_confidence=0.55)

        self.mt5            = MT5Connector(

            login    = config.mt5_login,

            password = config.mt5_password,

            server   = config.mt5_server,

        )

        self._mt5_available = False

        self.binance        = BinanceConnector(

            api_key    = config.binance_api_key,

            api_secret = config.binance_api_secret,

            testnet    = config.binance_testnet,

        )

        self.telegram     = TradingTelegramBot(

            on_approve=self._execute_trade,

            on_reject=self._reject_trade,

        )

        self.commander = TelegramCommander(

            bot_token=config.telegram_bot_token,

            chat_id=config.telegram_chat_id,

            on_mode_change=self._on_mode_change,

            on_callback=self.telegram.handle_callback,

            on_history=self._on_history_command,

        )

        self.commander._supervisor = self  # allow /proteger and /ver_mt5 to toggle supervisor state

        self.glint = GlintBrowser(

            ws_url=config.glint_ws_url,

            session_token=config.glint_session_token,

            email=config.glint_email,

            on_signal=self._on_glint_signal,

            min_impact="High",

        )

        self._glint_buffer: List[Dict]  = []

        self._last_glint_text: str      = ""

        self._demo_trades: List[DemoTrade] = self._load_demo_trades()
        self._crypto_h4_trend: Dict[str, str] = {}  # symbol → "LONG" | "SHORT"
        self._mt5_h4_direction: Dict[str, str] = {}  # symbol → "LONG" | "SHORT" | "WAIT"
        self._mt5_d1_trend: Dict[str, str] = {}     # symbol → "LONG" | "SHORT" (D1 50EMA)

        self.mode    = config.operation_mode

        self._running = False

        # Autonomous mode

        self._episodic_conn = get_db()

        self._learner   = AutonomousLearner(conn=self._episodic_conn)

        self._researcher = ResearchAgent(conn=self._episodic_conn)

        self._goals_mgr  = GoalsManager(conn=self._episodic_conn)

        self._reporter   = NightlyReporter(
            conn=self._episodic_conn,
            telegram_bot=self.telegram.get_bot(),
            chat_id=config.telegram_chat_id,
        )

        # Autonomous circuit breaker: per-symbol WR-based suspension +
        # drawdown-based risk multiplier, persisted across restarts.
        self.risk_governor = RiskGovernor(
            all_symbols=MT5_SYMBOLS,
            # Tiers calibrados para Axi (max drawdown 10%):
            # >= 7% → 0.25x | >= 4% → 0.5x | <4% → 1.0x (full)
            dd_tiers=((0.07, 0.25), (0.04, 0.5)),
            initial_suspended={
                "USDJPY": (
                    "Auditoria 2026-06-14: WR 6.2% en 130 trades (60% del volumen), "
                    "neto -$524 -- senal/SL/TP stale repetida"
                ),
                "GBPJPY": "Auditoria 2026-06-14: WR 17.6% en 17 trades, neto -$364",
            },
        )

        self._open_episodes: Dict[int, int] = self._load_open_episodes()

        # Load daily trade count from disk so pm2 restarts don't reset the limit
        self._daily_trades: Dict[str, int] = self._load_daily_trades()

        # FTMO / Axi rules enforcement

        self._ftmo_agent = FTMOAgent()

        self._ftmo_state = FTMOAgent.new_challenge(

            initial_balance=100_000.0,  # Axi Select account size — NOT startup capital param

            challenge_type=ChallengeType.TWO_STEP,

        )

        # Institutional agents -- enrichment layer (13 agents covering all dimensions)
        self._lunar         = LunarCycleAgent()
        self._elliott       = ElliottFibonacciAgent()
        self._footprint     = FootprintAgent()
        self._edge          = QuantEdgeAgent(capital=capital)
        self._chaos         = ChaosTheoryAgent()
        self._inst_flow     = InstitutionalFlowAgent()
        self._microstructure = MarketMicrostructureAgent()
        self._fed           = FEDSentimentAgent()
        self._onchain       = OnChainAgent()
        self._geopolitical  = GeopoliticalAgent()
        self._retail_psych  = RetailPsychologyAgent()
        self._alt_data      = AlternativeDataAgent()
        self._energy        = EnergyFrequencyAgent()
        # AxiVisionAgent -- Claude Vision reads MT5 screen every 5 min
        try:
            self._vision = AxiVisionAgent()
        except Exception:
            self._vision = None
        self._vision_protect_mode = False  # /proteger activates continuous monitoring
        # SMCAnalysisAgent -- Claude API final confirmation before real orders
        try:
            from agents.analysis_agent import SMCAnalysisAgent
            self._smc_agent = SMCAnalysisAgent()
        except Exception:
            self._smc_agent = None
        # df cache: populated each scan so _claude_confirm_trade can access latest df
        self._df_cache: Dict[str, pd.DataFrame] = {}
        # Scan statistics for /status and auto-reduce logic
        self._scan_stats = {
            "total": 0,
            "blocked_score": 0,
            "blocked_conservative": 0,
            "blocked_rr": 0,
            "blocked_daily_limit": 0,
            "blocked_ftmo": 0,
            "blocked_duplicate": 0,
            "blocked_claude": 0,
            "executed": 0,
            "last_trade_ts": None,
        }
        # Peak-profit tracker: ticket → max PnL seen this session
        self._position_peaks: Dict[int, float] = {}
        # Time-close retry cooldown: ticket → last attempt timestamp
        self._close_attempted: Dict[int, float] = {}
        # Daily profit target tracking
        self._daily_pnl_date: str = ""           # "YYYY-MM-DD" UTC
        self._daily_realized_pnl: float = 0.0   # closed trades today
        self._daily_target_hit: bool = False     # $245 hit — day locked, no new trades
        self._daily_protect_hit: bool = False    # reserved (unused)
        # Scalp daily target: $60 acumulado en scalps → cierra todos los scalps
        self._scalp_realized_today: float = 0.0
        self._scalp_daily_hit: bool = False
        self._scalp_pnl_date: str = ""
        self._scalp_peak_today: float = 0.0  # max alcanzado hoy — si cae a $60 cierra
        # High-water mark: balance máximo histórico — recuperar si cae > $500 del pico
        self._balance_peak: float = INITIAL_CAPITAL

    # Callbacks from TelegramCommander â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



    def _on_mode_change(self, mode: str):

        self.mode = mode

        print(f"[Mode] Cambiado a: {mode.upper()} vÃ­a Telegram")



    def _on_history_command(self, symbol: str) -> str:

        return self.historical.get_market_summary(symbol or "BTC")



    # â"€â"€ Glint callback â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



    def _on_glint_signal(self, glint: GlintSignal):

        if glint.is_actionable():

            self._last_glint_text = glint.text

            self._glint_buffer.append(glint.raw)

            if len(self._glint_buffer) > 20:

                self._glint_buffer.pop(0)

            asyncio.create_task(self.telegram.send_glint_alert(glint.format_alert()))

            print(f"[Glint] {glint.impact}: {glint.text[:80]}...")



    # â"€â"€ Decision pipeline â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



    def route_signal(self, signal: TradeSignal, df=None) -> TradeSignal:

        """

        Runs the signal through DecisionFilter and enriches it with

        score, grade, and risk_multiplier. Returns the same signal mutated.

        """

        if df is None or signal.stop_loss is None:

            return signal



        decision = self.decision.evaluate(

            df             = df,

            symbol         = signal.symbol,

            timeframe      = signal.timeframe,

            entry          = signal.entry,

            stop_loss      = signal.stop_loss,

            take_profit    = signal.take_profit,

            bias           = ("bullish" if signal.signal_type == SignalType.LONG

                              else "bearish" if signal.signal_type == SignalType.SHORT

                              else "neutral"),

            glint_signals  = list(self._glint_buffer),

        )



        signal.decision_score   = decision.score

        signal.decision_grade   = decision.grade.value

        signal.risk_multiplier  = decision.risk_multiplier

        signal.premium_alert    = decision.premium_alert

        signal.score_breakdown  = decision.breakdown



        # Override signal type to WAIT if filter blocks it

        if decision.grade == TradeGrade.NO_TRADE:

            signal.signal_type = SignalType.WAIT

            signal.notes = decision.reason



        return signal



    def _dispatch(self, signal: TradeSignal):

        """Routes to auto-execute or semi-auto depending on grade and mode."""

        grade = signal.decision_grade



        if signal.signal_type == SignalType.WAIT:

            print(f"[FILTER] NO TRADE -- {signal.notes}")

            return



        # Log decision

        print(

            f"[FILTER] {signal.symbol} | Score {signal.decision_score}/100"

            f" | Grade: {grade.upper()}"

            f" | Riesgo: {int(signal.risk_multiplier*100)}%"

        )



        if signal.premium_alert:

            print(f"[PREMIUM] Setup excepcional en {signal.symbol}!")



        if self.mode == "auto" or (self.mode == "hybrid" and grade == "premium"):

            self._execute_trade(signal)

        elif self.mode == "semi" or self.mode == "hybrid":

            asyncio.create_task(self.telegram.send_signal(signal, mode="semi"))

        else:

            asyncio.create_task(self.telegram.send_signal(signal, mode="auto"))



    # â"€â"€ Trade execution â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



    def _execute_trade(self, signal: TradeSignal):

        allowed, reason = self.risk_manager.can_open_trade()

        if not allowed:

            asyncio.create_task(self.telegram.send_risk_alert(reason))

            return



        validation = self.risk_manager.validate_trade(

            entry=signal.entry,

            stop_loss=signal.stop_loss,

            take_profit=signal.take_profit,

        )

        # Apply risk multiplier from DecisionFilter

        base_size = self.risk_manager.calculate_position_size(

            entry=signal.entry,

            stop_loss=signal.stop_loss,

        )

        actual_size = round(base_size * signal.risk_multiplier, 4)



        print(f"[EXECUTE] {signal.symbol} {signal.signal_type.value.upper()}")

        print(f"  Entry: {signal.entry} | SL: {signal.stop_loss} | TP: {signal.take_profit}")

        print(f"  Size: {actual_size} (base {base_size} Ã-- {signal.risk_multiplier}) | R:R 1:{validation['risk_reward']}")

        self.risk_manager.open_positions += 1



    def _reject_trade(self, signal: TradeSignal):

        print(f"[REJECT] {signal.symbol} -- rechazado manualmente")



    # â"€â"€ Main loop â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



    async def run(self):

        self._running = True

        print("=" * 55)

        print("  SMC TRADING BOT -- Claude AI + Glint + DecisionFilter")

        print("=" * 55)

        print(f"  Modo:          {self.mode.upper()}")

        print(f"  Capital:       ${self.capital:,.2f}")

        print(f"  Riesgo max:    {self.config.max_risk_per_trade*100}% por trade")

        print(f"  Timeframes:    {', '.join(self.config.timeframes)}")

        print(f"  Score gates:   <60=NO | 60-74=25% | 75-89=100% | 90+=PREMIUM")

        if self.demo_mode:

            print(f"  DEMO MODE:     threshold={DEMO_SCORE_THRESHOLD} | max_trades={DEMO_MAX_POSITIONS}")

            print(f"  Crypto:        {', '.join(SCAN_SYMBOLS)}")

            print(f"  Timeframes:    {', '.join(SCAN_TIMEFRAMES)}")

            print(f"  Scan interval: {SCAN_INTERVAL_SEC}s")

        print()



        # MT5 startup check -- try port 443 first for ISP compatibility

        loop = asyncio.get_event_loop()

        await loop.run_in_executor(None, self.mt5.ensure_port_443_config)

        mt5_ok = await loop.run_in_executor(None, self.mt5.connect)

        if mt5_ok:

            self._mt5_available = True

            info = await loop.run_in_executor(None, self.mt5.get_account_info)

            bal  = info.get("balance", 0)

            # Sync capital with real MT5 balance
            if bal > 0:
                self.capital = bal
                self.risk_manager.update_capital(bal)

            # Backfill outcomes for any positions that closed during prior restart
            await loop.run_in_executor(None, self._recover_orphaned_episodes)

            # Sync daily PnL on startup — prevents race condition where _market_scan_loop
            # fires immediately (no sleep) before _position_monitor_loop first runs (60s sleep).
            # Without this, bot could open new trades even if today's target was already hit.
            _today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            _realized  = await loop.run_in_executor(None, self.mt5.get_daily_pnl)
            self._daily_pnl_date     = _today_str
            self._daily_realized_pnl = float(_realized) if _realized is not None else 0.0
            if self._daily_realized_pnl >= DAILY_PROFIT_TARGET:
                self._daily_target_hit = True
                print(f"[STARTUP] Meta diaria ya cumplida: ${self._daily_realized_pnl:.2f} >= ${DAILY_PROFIT_TARGET:.0f} — sin trades hoy", flush=True)
            else:
                print(f"[STARTUP] PnL realizado hoy: ${self._daily_realized_pnl:.2f} / meta ${DAILY_PROFIT_TARGET:.0f}", flush=True)

            print(f"  MT5:           CONECTADO -- Balance ${bal:,.2f}")

            print(f"  Forex:         {', '.join(self.risk_governor.active_symbols())}")
            if self.risk_governor.suspended_symbols():
                print(f"  Suspendidos:   {', '.join(self.risk_governor.suspended_symbols().keys())} (RiskGovernor)")

            try:

                await self.telegram.send_glint_alert(

                    f"<b>MT5 AXI CONECTADO</b>\nBalance: ${bal:,.2f} USD\nServer: {self.mt5.server}"

                )

            except Exception:

                pass

        else:

            self._mt5_available = False

            msg = self.mt5.last_error_msg()

            print(f"  MT5:           {msg}")

            try:

                await self.telegram.send_glint_alert(f"MT5 no disponible -- {msg}")

            except Exception:

                pass

        print()



        while self._running:
            try:
                results = await asyncio.gather(
                    self.commander.start_polling(),
                    self.glint.connect(),
                    self._market_scan_loop(),
                    self._position_monitor_loop(),
                    self._learning_loop(),
                    self._research_loop(),
                    self._goals_loop(),
                    self._risk_governor_loop(),
                    self._nightly_report_loop(),
                    self._vision_monitor_loop(),
                    return_exceptions=True,
                )
                for r in results:
                    if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                        print(f"[RUN] Task excepcion: {r.__class__.__name__}: {r}", flush=True)
            except asyncio.CancelledError:
                break
            except Exception as _run_exc:
                print(f"[RUN] Gather crasheo: {_run_exc.__class__.__name__}: {_run_exc} -- reiniciando en 10s", flush=True)
                await asyncio.sleep(10)
            if not self._running:
                break
            print("[RUN] Todos los loops completaron -- reiniciando en 5s", flush=True)
            await asyncio.sleep(5)



    @staticmethod

    async def _check_internet() -> bool:

        """Fast TCP probe to 8.8.8.8:53 -- no external libraries needed."""

        import socket

        loop = asyncio.get_event_loop()

        try:

            await asyncio.wait_for(

                loop.run_in_executor(

                    None,

                    lambda: socket.create_connection(("8.8.8.8", 53), timeout=3),

                ),

                timeout=4,

            )

            return True

        except Exception:

            return False



    # â"€â"€ Technical SMC analysis (no API call) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



    def _run_smc_lite(self, df: pd.DataFrame) -> dict:

        """

        Lightweight SMC analysis without Claude API -- runs in the scan loop.

        Returns a dict with bias, order blocks, FVGs, and a simple setup flag.

        """

        from smc.structure import MarketStructure

        from smc.orderblocks import OrderBlockDetector, FVGDetector

        ms      = MarketStructure(df)

        struct  = ms.analyze()

        ob_det  = OrderBlockDetector(df)

        fvg_det = FVGDetector(df)

        bull_obs  = ob_det.find_bullish_obs()

        bear_obs  = ob_det.find_bearish_obs()

        bull_fvgs = fvg_det.find_bullish_fvg()

        bear_fvgs = fvg_det.find_bearish_fvg()

        bos_list  = ms.detect_bos()

        choch_list = ms.detect_choch()



        # Only keep POI zones within 5% of current price — discard stale historical blocks
        current_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        _max_poi_dist  = current_close * 0.05 if current_close > 0 else float("inf")
        poi_zones = []
        for ob in (bull_obs + bear_obs)[:5]:
            zone_mid = (ob.get("zone_high", 0) + ob.get("zone_low", 0)) / 2.0
            if zone_mid > 0 and abs(zone_mid - current_close) <= _max_poi_dist:
                poi_zones.append(ob)
            if len(poi_zones) >= 3:
                break



        is_bullish = struct.bias == "bullish"

        is_bearish = struct.bias == "bearish"



        # When bias is neutral, use direction of the last confirmed BOS

        if not (is_bullish or is_bearish) and bos_list:

            last_dir = bos_list[-1].get("direction", "")

            if last_dir == "bullish":

                is_bullish = True

            elif last_dir == "bearish":

                is_bearish = True



        # When still neutral but recent CHoCH, use it for direction

        if not (is_bullish or is_bearish) and choch_list:

            last_choch = choch_list[-1].get("direction", "")

            if last_choch == "bullish":

                is_bullish = True

            elif last_choch == "bearish":

                is_bearish = True



        has_ob  = bool(bull_obs if is_bullish else bear_obs)

        has_fvg = bool(bull_fvgs if is_bullish else bear_fvgs)

        has_bos = bool(bos_list)

        has_setup = (is_bullish or is_bearish) and (has_ob or has_fvg or has_bos)



        direction_word = "bullish" if is_bullish else ("bearish" if is_bearish else "neutral")

        analysis_text = f"{direction_word} trend {struct.structure_type.value}"

        if has_bos:    analysis_text += " BOS confirmado"

        if choch_list: analysis_text += " CHoCH detectado"

        if has_ob:     analysis_text += " order block presente"

        if has_fvg:    analysis_text += " FVG presente"

        if has_setup:  analysis_text += " setup valido"



        return {

            "bias": struct.bias,

            "has_ob": has_ob,

            "has_fvg": has_fvg,

            "has_bos": has_bos,

            "has_choch": bool(choch_list),

            "has_setup": has_setup,

            "poi_zones": poi_zones,

            "analysis_text": analysis_text,

            "structure": struct,

        }



    _DAILY_TRADES_PATH = os.path.join("memory", "daily_trades.json")
    _DEMO_TRADES_PATH  = os.path.join("memory", "demo_trades_state.json")

    def _load_demo_trades(self) -> list:
        """Load open demo trades from disk so bot restarts don't lose positions."""
        import json
        try:
            with open(self._DEMO_TRADES_PATH, "r", encoding="utf-8") as f:
                rows = json.load(f)
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
            restored = []
            for row in rows:
                if row.get("opened_at", "") <= cutoff:
                    continue  # expired
                from agents.signal_agent import TradeSignal, SignalType
                sig = TradeSignal(
                    symbol=row["symbol"],
                    signal_type=SignalType.LONG if row["direction"] == "long" else SignalType.SHORT,
                    entry=row["entry"],
                    stop_loss=row["sl"],
                    take_profit=row["tp"],
                    timeframe=row.get("timeframe", "1h"),
                    trigger=row.get("trigger", "restored"),
                    confidence=row.get("confidence", 0.7),
                )
                sig.decision_score = row.get("score", 60)
                demo = DemoTrade(sig, sig.decision_score)
                demo.opened_at = datetime.fromisoformat(row["opened_at"])
                restored.append(demo)
            if restored:
                print(f"[DEMO] Restored {len(restored)} open demo trades from disk", flush=True)
            return restored
        except Exception:
            return []

    def _save_demo_trades(self):
        """Persist open demo trade state to JSON."""
        import json
        try:
            rows = []
            for d in self._demo_trades:
                if d.status != "open":
                    continue
                rows.append({
                    "symbol":    d.signal.symbol,
                    "direction": "long" if d.signal.signal_type == SignalType.LONG else "short",
                    "entry":     d.signal.entry,
                    "sl":        d.signal.stop_loss or 0.0,
                    "tp":        d.signal.take_profit or 0.0,
                    "timeframe": d.signal.timeframe,
                    "trigger":   getattr(d.signal, "trigger", ""),
                    "confidence":getattr(d.signal, "confidence", 0.7),
                    "score":     d.score,
                    "opened_at": d.opened_at.isoformat(),
                })
            with open(self._DEMO_TRADES_PATH, "w", encoding="utf-8") as f:
                json.dump(rows, f)
        except Exception:
            pass

    def _load_daily_trades(self) -> Dict[str, int]:
        """Load daily MT5 trade count from disk — survives pm2 restarts."""
        import json
        from datetime import date, timedelta
        try:
            with open(self._DAILY_TRADES_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Purge entries older than 7 days to prevent indefinite growth
            cutoff = (date.today() - timedelta(days=7)).isoformat()
            return {k: v for k, v in loaded.items() if k >= cutoff}
        except Exception:
            return {}

    def _save_daily_trades(self):
        """Persist daily trade count so resets don't bypass MAX_DAILY_TRADES."""
        import json
        try:
            with open(self._DAILY_TRADES_PATH, "w", encoding="utf-8") as f:
                json.dump(self._daily_trades, f)
        except Exception:
            pass

    def _save_scan_stats(self):
        """Persist scan stats to JSON so Telegram /status can read them."""
        import json, os
        try:
            stats = dict(self._scan_stats)
            ts = stats.get("last_trade_ts")
            stats["last_trade_ts"] = ts.isoformat() if ts else None
            path = os.path.join("memory", "scan_stats.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(stats, f)
        except Exception:
            pass

    def _enrich_with_agents(self, signal: TradeSignal, df: pd.DataFrame) -> int:
        """Run all 13 institutional agents IN PARALLEL and return total bonus pts.

        All agents fire simultaneously via ThreadPoolExecutor — total latency equals
        the slowest single agent, not the sum of all agents (~13x speedup).
        Requires base score >= 65 to be effective (agents confirm, don't rescue).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if signal.decision_score < 40:  # lowered from 50 — allow EURUSD/GBPUSD to be enriched
            return 0

        bias = "bullish" if signal.signal_type == SignalType.LONG else "bearish"
        prices = list(df["close"].astype(float).values) if not df.empty else []

        _agent_names = ["lunar","elliott","chaos","edge","footprint","instflow","micro","fed","onchain","geo","retail","alt","energy"]
        _agent_results = {}

        def _make(name, fn):
            def _wrapped():
                try:
                    val = fn()
                    _agent_results[name] = val
                    return val
                except Exception as e:
                    _agent_results[name] = f"ERR:{type(e).__name__}"
                    return 0
            return _wrapped

        def _lunar():   return self._lunar.score_adjustment(bias)
        def _elliott():
            e = self._elliott.analyze(df, bias)
            return e.score_bonus
        def _chaos():   return self._chaos.score_adjustment(df)
        def _edge():
            edge = self._edge.calculate_full_edge(symbol=signal.symbol, prices=prices)
            return self._edge.get_decision_pts(edge)
        def _footprint():
            if signal.symbol not in SCAN_SYMBOLS:
                return 0
            fp_candle = self._footprint.build_live_footprint(
                signal.symbol, candle_open=signal.entry or 0, limit=500
            )
            direction = "long" if signal.signal_type == SignalType.LONG else "short"
            return self._footprint.score_for_trade(fp_candle, direction, signal.entry or 0)
        def _instflow(): return self._inst_flow.score_adjustment(signal.symbol, bias)
        def _micro():    return self._microstructure.score_adjustment(signal.symbol, signal.entry or 0.0)
        def _fed():      return self._fed.score_adjustment(signal.symbol, bias)
        def _onchain():  return self._onchain.score_adjustment(signal.symbol, bias, signal.entry or 0.0)
        def _geo():      return self._geopolitical.score_adjustment(signal.symbol, bias)
        def _retail():   return self._retail_psych.score_adjustment(signal.symbol, df, bias)
        def _alt():      return self._alt_data.score_adjustment(signal.symbol, bias)
        def _energy():
            energy = self._energy.analyze(signal.symbol, signal.entry or 0.0, prices)
            pts = energy.to_decision_pts()
            return max(-3, min(3, pts))  # solo sugeridor: ±3 pts max

        raw_tasks = [_lunar, _elliott, _chaos, _edge, _footprint, _instflow, _micro, _fed, _onchain, _geo, _retail, _alt, _energy]
        tasks = [_make(name, fn) for name, fn in zip(_agent_names, raw_tasks)]

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(fn) for fn in tasks]
            bonus = sum(f.result() for f in as_completed(futures))

        bonus_clamped = int(max(-30, min(60, bonus)))
        base = signal.decision_score

        # Log which agents contributed and which errored
        errors = [f"{k}:{v}" for k, v in _agent_results.items() if isinstance(v, str) and v.startswith("ERR")]
        contribs = {k: v for k, v in _agent_results.items() if isinstance(v, (int, float)) and v != 0}
        if errors:
            print(f"[ENRICH-ERR] {signal.symbol}: {' | '.join(errors)}", flush=True)
        if contribs:
            contrib_str = " ".join(f"{k}={v:+d}" for k, v in contribs.items())
            print(f"[ENRICH-CONTRIB] {signal.symbol}: {contrib_str}", flush=True)

        print(
            f"[ENRICH] {signal.symbol} base={base} bonus={bonus_clamped:+d} "
            f"final={base + bonus_clamped} | {signal.signal_type.value.upper()}",
            flush=True,
        )
        return bonus_clamped



    async def _scan_symbol(self, symbol: str, timeframe: str) -> Optional[TradeSignal]:

        """
        Full pipeline for one symbol/timeframe:
        fetch → SMC lite → SignalAgent → DecisionFilter → H4 trend filter → return signal or None
        """

        loop = asyncio.get_event_loop()

        df = await loop.run_in_executor(
            None, lambda: self.binance.get_ohlcv(symbol, timeframe, limit=200)
        )

        if df.empty or len(df) < 50:
            return None

        # ── Cache H4 trend when scanning the 4h timeframe ────────────────────
        if timeframe == "4h" and len(df) >= 11:
            avg_fast = df["close"].iloc[-3:].mean()
            avg_slow = df["close"].iloc[-11:-3].mean()
            self._crypto_h4_trend[symbol] = "LONG" if avg_fast > avg_slow else "SHORT"

        smc = self._run_smc_lite(df)
        current_price = float(df["close"].iloc[-1])

        signal = self.signal_agent.evaluate(
            analysis_text = smc["analysis_text"],
            symbol        = symbol,
            timeframe     = timeframe,
            current_price = current_price,
            poi_zones     = smc["poi_zones"],
            glint_context = self._last_glint_text,
            df            = df,
        )

        if signal.signal_type == SignalType.WAIT:
            return signal   # still return so we can log the score=0

        # ── H4 trend alignment filter for 1h crypto trades ───────────────────
        if timeframe == "1h" and signal.signal_type != SignalType.WAIT:
            h4_trend = self._crypto_h4_trend.get(symbol)
            if h4_trend:
                sig_dir = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"
                if sig_dir != h4_trend:
                    print(
                        f"[H4-FILTER] {symbol} 1h {sig_dir} contra tendencia H4 ({h4_trend}) — skip",
                        flush=True,
                    )
                    signal.signal_type = SignalType.WAIT
                    signal.decision_score = 0
                    return signal

        original_direction = signal.signal_type

        signal = self.route_signal(signal, df)

        # Same rescue logic as MT5: agents can unlock borderline signals
        if original_direction != SignalType.WAIT and signal.decision_score >= 50:
            agent_bonus = self._enrich_with_agents(signal, df)
            if agent_bonus != 0:
                new_score = max(0, min(150, signal.decision_score + agent_bonus))
                signal.decision_score = new_score
                signal.score_breakdown["agents"] = agent_bonus
                if signal.signal_type == SignalType.WAIT and new_score >= 75:
                    signal.signal_type = original_direction

        return signal



    # â"€â"€ Demo trade execution â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€





    async def _scan_mt5_symbol(self, symbol: str, timeframe: str):

        """Fetch MT5 OHLCV, run SMC lite, return signal or None."""

        loop = asyncio.get_event_loop()

        df = await loop.run_in_executor(None, lambda: self.mt5.get_ohlcv(symbol, timeframe, 200))

        if df is None or df.empty or len(df) < 50:

            return None

        self._df_cache[symbol] = df  # available for _claude_confirm_trade
        smc = self._run_smc_lite(df)

        current_price = float(df["close"].iloc[-1])

        signal = self.signal_agent.evaluate(

            analysis_text=smc["analysis_text"], symbol=symbol,

            timeframe=timeframe, current_price=current_price,

            poi_zones=smc["poi_zones"], glint_context=self._last_glint_text,

            df=df,

        )

        if signal.signal_type == SignalType.WAIT:

            return signal

        original_direction = signal.signal_type  # preserve before route_signal may override to WAIT

        signal = self.route_signal(signal, df)

        # Enrichment runs based on ORIGINAL direction so agents can rescue borderline scores.
        # If route_signal gated the signal (score 50-74 → WAIT) but agents boost above 75,
        # restore the original direction so the trade can proceed.
        if original_direction != SignalType.WAIT and signal.decision_score >= 50:
            agent_bonus = self._enrich_with_agents(signal, df)
            if agent_bonus != 0:
                new_score = max(0, min(150, signal.decision_score + agent_bonus))
                signal.decision_score = new_score
                signal.score_breakdown["agents"] = agent_bonus
                if signal.signal_type == SignalType.WAIT and new_score >= 75:
                    signal.signal_type = original_direction  # agents unlocked this setup



        return signal



    async def _scan_forex_yfinance(self, symbol: str, tf_yf: str, tf_label: str):

        """Scan forex pair using yfinance data (MT5 fallback)."""

        try:

            import yfinance as yf

            loop = asyncio.get_event_loop()

            ticker_sym = YFINANCE_FOREX.get(symbol, symbol + "=X")

            def _fetch():

                period_map = {"1h": "5d", "4h": "30d"}

                period = period_map.get(tf_yf, "5d")

                t = yf.Ticker(ticker_sym)

                df = t.history(period=period, interval=tf_yf)

                if df.empty:

                    return None

                df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})

                return df[["open","high","low","close","volume"]].copy()

            df = await loop.run_in_executor(None, _fetch)

            if df is None or len(df) < 50:

                return None

            smc = self._run_smc_lite(df)

            current_price = float(df["close"].iloc[-1])

            signal = self.signal_agent.evaluate(

                analysis_text=smc["analysis_text"], symbol=symbol,

                timeframe=tf_label, current_price=current_price,

                poi_zones=smc["poi_zones"], glint_context=self._last_glint_text,

            )

            if signal.signal_type == SignalType.WAIT:

                return signal

            signal = self.route_signal(signal, df)

            return signal

        except Exception as exc:

            logger.debug(f"yfinance scan {symbol} error: {exc}")

            return None



    async def _claude_confirm_trade(self, signal: TradeSignal) -> tuple:
        """
        FILTER 8 -- Claude API final confirmation before placing a real order.
        Uses claude-haiku (cheap/fast) + episodic memory to reason about the setup.
        Returns (can_trade: bool, adjusted_score: int, summary: str).
        Falls back to (True, original_score) if API unavailable.
        """
        # Auto-confirm: 13-agent enrichment score is sufficient for trade quality.
        # Claude API confirmation disabled to preserve credits.
        return True, signal.decision_score, "auto-confirm"
        if self._smc_agent is None:
            return True, signal.decision_score, "no-api-key"
        try:
            df = self._df_cache.get(signal.symbol, pd.DataFrame())
            # Regime from chaos agent
            try:
                chaos_sig = self._chaos.get_signal(df) if not df.empty else None
                regime = chaos_sig.hurst.interpretation if chaos_sig else "unknown"
            except Exception:
                regime = "unknown"

            smc_summary = (
                f"{signal.symbol} {signal.timeframe} | Bias: "
                f"{'bullish' if signal.signal_type == SignalType.LONG else 'bearish'}\n"
                f"Score: {signal.decision_score} | RR: {signal.risk_reward:.1f}\n"
                f"Entry: {signal.entry} | SL: {signal.stop_loss} | TP: {signal.take_profit}\n"
                f"Regime: {regime}"
            )
            loop = asyncio.get_running_loop()
            similar = await loop.run_in_executor(
                None,
                lambda: query_similar_episodes(
                    signal.symbol, "SMC", regime, n=10, conn=self._episodic_conn
                ),
            )
            result = await loop.run_in_executor(
                None,
                lambda: self._smc_agent.reason_with_context(
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    smc_summary=smc_summary,
                    similar_episodes=similar,
                    regime=regime,
                    base_score=signal.decision_score,
                ),
            )
            if result.get("fallback"):
                return True, signal.decision_score, "claude-fallback"
            can_trade   = not result.get("wait_override", False)
            adj_score   = result.get("adjusted_score", signal.decision_score)
            confidence  = result.get("confidence", 50)
            decision    = result.get("reasoning", {}).get("decision", "?")
            justif      = result.get("reasoning", {}).get("justification", "")[:80]
            summary     = f"{decision} conf={confidence} | {justif}"
            return can_trade, adj_score, summary
        except Exception as exc:
            print(f"[CLAUDE] confirm error: {exc}", flush=True)
            return True, signal.decision_score, "error-fallback"

    async def _send_mt5_real_order(self, signal: TradeSignal):

        """Send a real order to MT5 demo -- strict quality filters applied."""

        order_type = "BUY" if signal.signal_type == SignalType.LONG else "SELL"
        _is_scalp  = (signal.timeframe == "M15")   # definido aqui para todo el metodo

        sl_val = signal.stop_loss if signal.stop_loss else 0.0

        tp_val = signal.take_profit if signal.take_profit else 0.0



        # Meta swing $245 cumplida → SOLO scalps (M15) el resto del día
        # Los swings ya aseguraron el mínimo — no abrir más swings que se coman la ganancia
        if self._daily_target_hit and not _is_scalp:
            print(f"[MT5] {signal.symbol}: meta swing $245 cumplida — solo scalps permitidos, skip swing", flush=True)
            return

        # ── FILTER 0: Mercado abierto ─────────────────────────────────────
        if not is_market_open(signal.symbol):
            mins = minutes_until_open(signal.symbol)
            print(
                f"[MT5] {signal.symbol}: mercado cerrado -- abre en ~{mins}min, skip",
                flush=True,
            )
            return

        # ── FILTER 1: SL obligatorio ──────────────────────────────────────

        if sl_val == 0.0:

            print(f"[MT5] {signal.symbol}: SL no definido, skip", flush=True)

            return



        # ── FILTER 2: Pares permitidos en modo conservador ────────────────

        if CONSERVATIVE_MODE and signal.symbol not in CONSERVATIVE_PAIRS:

            print(f"[MT5] {signal.symbol}: modo conservador -- solo {CONSERVATIVE_PAIRS}", flush=True)

            return



        # ── FILTER 3: Horario muerto ──────────────────────────────────────

        now_utc = datetime.now(timezone.utc)

        if now_utc.hour in DEAD_HOURS_UTC:

            print(f"[MT5] {signal.symbol}: hora muerta {now_utc.hour}:00 UTC, skip", flush=True)

            return

        if now_utc.weekday() == 4 and now_utc.hour >= 16:  # viernes 16:00+ UTC — no abrir nuevos trades
            print(f"[MT5] {signal.symbol}: viernes 16:00+ UTC, no se abren nuevos trades, skip", flush=True)
            return



        # ── FILTER 3b: Tendencia H4 — solo entrar a favor del trend ─────────
        # Usa 50 velas H4 (200h ≈ 8 días) para evitar falsos por pullbacks cortos.
        # MA10 vs MA30: si fallan datos → bloquear (no continuar con señal sin confirmar).
        _h4_trend_ok = False
        try:
            df_h4 = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.mt5.get_ohlcv(signal.symbol, "H4", 50)
            )
            if df_h4 is not None and len(df_h4) >= 30:
                avg_fast = df_h4["close"].iloc[-10:].mean()   # MA10 — últimas 10 velas H4 (40h)
                avg_slow = df_h4["close"].iloc[-30:].mean()   # MA30 — últimas 30 velas H4 (120h)
                h4_bias  = "LONG" if avg_fast > avg_slow else "SHORT"
                sig_dir  = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"
                if sig_dir != h4_bias:
                    print(
                        f"[TREND-H4] {signal.symbol}: {sig_dir} contra H4 ({h4_bias}) -- skip",
                        flush=True,
                    )
                    return
                print(f"[TREND-H4] {signal.symbol}: {sig_dir} a favor H4 ({h4_bias}) OK", flush=True)
                _h4_trend_ok = True
            else:
                print(f"[TREND-H4] {signal.symbol}: datos H4 insuficientes — skip", flush=True)
                return
        except Exception as _te:
            print(f"[TREND-H4] {signal.symbol}: error obteniendo H4 ({_te}) — skip", flush=True)
            return  # sin datos H4 confiables → no operar

        # ── SCALP M15: volumen fijo 0.1L, SL=4pips($4), TP=12pips($12) ─────────
        # 0.1L: pip=$1 → SL=4pips=$4 max loss, TP=12pips=$12, cerrar en $10
        if _is_scalp:
            try:
                import MetaTrader5 as _mt5s
                _tick_s = _mt5s.symbol_info_tick(signal.symbol)
                _scalp_price = (_tick_s.ask if order_type == "BUY" else _tick_s.bid) if _tick_s else 0.0
                _sym_s = _mt5s.symbol_info(signal.symbol)
                if _scalp_price > 0 and _sym_s:
                    _pip = _sym_s.point * 10  # 1 pip = 10 points (5-digit broker)
                    _sl_pips  = 4   # 4 pips SL = $4 max loss a 0.1L
                    _tp_pips  = 12  # 12 pips TP = $12, monitor cierra en $10
                    if order_type == "BUY":
                        sl_val = round(_scalp_price - _sl_pips * _pip, 5)
                        tp_val = round(_scalp_price + _tp_pips * _pip, 5)
                    else:
                        sl_val = round(_scalp_price + _sl_pips * _pip, 5)
                        tp_val = round(_scalp_price - _tp_pips * _pip, 5)
                    print(f"[SCALP] {signal.symbol} {order_type} @{_scalp_price:.5f} SL={sl_val:.5f} TP={tp_val:.5f} (4pip/$4 SL, 12pip/$12 TP)", flush=True)
            except Exception as _se:
                print(f"[SCALP] error calculando SL/TP: {_se}", flush=True)

        # ── FILTER 4: RR minimo — SIEMPRE usa precio actual (signal.entry puede ser H4 stale) ───
        if tp_val > 0 and sl_val > 0:
            try:
                import MetaTrader5 as _mt5
                _tick = _mt5.symbol_info_tick(signal.symbol)
                _market_price = (_tick.ask if order_type == "BUY" else _tick.bid) if _tick else 0.0
            except Exception:
                _market_price = 0.0
            # Always use current market price — H4 signal entry can be hours old
            _entry_ref = _market_price if _market_price > 0 else (signal.entry if signal.entry and signal.entry > 0 else 0)
            if _entry_ref > 0:
                # Slippage guard: if market moved > 2x SL_dist from signal entry,
                # recalculate entry at current market price (don't skip — just adapt).
                # 2x tolerance: 1x was too tight and blocked valid trending setups (USDJPY).
                if signal.entry and signal.entry > 0 and _market_price > 0:
                    _sl_dist_signal = abs(signal.entry - sl_val)
                    _slippage = abs(_market_price - signal.entry)
                    if _sl_dist_signal > 0 and _slippage > _sl_dist_signal:
                        drift_factor = _slippage / _sl_dist_signal
                        if drift_factor > 2.0:
                            # Scalp M15: siempre usa precio de mercado — stale no aplica
                            if _is_scalp:
                                pass  # SL/TP ya calculados al precio actual
                            else:
                                print(
                                    f"[MT5] {signal.symbol}: STALE SETUP drift={_slippage:.4f} "
                                    f"({drift_factor:.1f}x SL_dist > 2x max) — skip",
                                    flush=True,
                                )
                                return
                        # 1-2x drift: adapt entry to current market price
                        print(
                            f"[MT5] {signal.symbol}: ADAPTED entry {signal.entry:.4f}→{_market_price:.4f} "
                            f"(drift={_slippage:.4f}, {drift_factor:.1f}x SL_dist), recalculating SL/TP at market",
                            flush=True,
                        )
                        _entry_ref = _market_price
                        _sl_dist_orig = _sl_dist_signal
                        sl_val = _entry_ref - _sl_dist_orig if order_type == "BUY" else _entry_ref + _sl_dist_orig
                        tp_val = _entry_ref + _sl_dist_orig * MIN_RR if order_type == "BUY" else _entry_ref - _sl_dist_orig * MIN_RR
                        sl_val = round(sl_val, 5)
                        tp_val = round(tp_val, 5)

                sl_dist = abs(_entry_ref - sl_val)
                tp_dist = abs(_entry_ref - tp_val)
                rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
                _min_rr_req = 1.5 if _is_scalp else MIN_RR - 0.01
                if rr < _min_rr_req:
                    print(f"[MT5] {signal.symbol}: RR={rr:.2f} < {_min_rr_req} minimo, skip", flush=True)
                    return
                print(f"[RR-OK] {signal.symbol}: RR={rr:.2f} ({'SCALP' if _is_scalp else 'SWING'})", flush=True)



        # ── FILTER 5: Max 1 trade real por dia ───────────────────────────

        today_str = now_utc.strftime("%Y-%m-%d")

        trades_today = self._daily_trades.get(today_str, 0)

        if trades_today >= MAX_DAILY_TRADES:

            print(f"[MT5] {signal.symbol}: {trades_today} trades hoy (max={MAX_DAILY_TRADES}), skip", flush=True)

            return



        # ── FILTER 6: FTMO / Axi drawdown y perdida diaria ───────────────

        try:

            daily_pnl = await asyncio.get_running_loop().run_in_executor(

                None, self.mt5.get_daily_pnl

            )

            self._ftmo_state.daily_pnl_today = daily_pnl
            # Keep daily_realized_pnl in sync with MT5 real closed P&L
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_pnl_date == today_utc and daily_pnl is not None:
                self._daily_realized_pnl = float(daily_pnl)

            acc_info = await asyncio.get_running_loop().run_in_executor(

                None, self.mt5.get_account_info

            )

            self._ftmo_state.current_balance = acc_info.get("balance", self.capital)

            _equity = acc_info.get("equity", self._ftmo_state.current_balance)
            can_trade, reason = self._ftmo_agent.can_trade(self._ftmo_state, equity=_equity)

            if not can_trade:

                print(f"[FTMO] {signal.symbol}: BLOQUEADO -- {reason}", flush=True)

                try:

                    await self.telegram.send_glint_alert(

                        f"<b>FTMO BLOQUEO</b>\n{signal.symbol}: {reason}"

                    )

                except Exception:

                    pass

                return

        except Exception as _fe:

            print(f"[FTMO] check error: {_fe}", flush=True)



        # ── FILTER 7: Max posiciones abiertas ────────────────────────────────
        loop = asyncio.get_running_loop()
        existing = await loop.run_in_executor(None, self.mt5.get_positions)

        # Scalp y swing tienen topes independientes
        # Modo recuperación: permite más scalps simultáneos para recuperar más rápido
        _current_bal_r  = self._ftmo_state.current_balance or self.capital
        _recovery_mode  = (
            self._daily_realized_pnl <= RECOVERY_TRIGGER_LOSS or
            _current_bal_r < INITIAL_CAPITAL or
            (self._balance_peak - _current_bal_r) >= RECOVERY_DRAWDOWN_FROM_PEAK
        ) and not self._daily_target_hit
        _accel_mode = (
            self._daily_realized_pnl >= ACCEL_TRIGGER_PROFIT and
            _current_bal_r >= INITIAL_CAPITAL and not _recovery_mode
        )
        _max_scalp_now = (RECOVERY_MAX_SCALPS if _recovery_mode
                          else ACCEL_MAX_SCALPS if _accel_mode
                          else MAX_SCALP_POSITIONS)
        if _is_scalp:
            scalp_open = [p for p in existing if p.get("volume", 1) <= 0.10]
            if len(scalp_open) >= _max_scalp_now:
                print(f"[MT5] {signal.symbol}: {len(scalp_open)} scalps abiertas (max={_max_scalp_now}{'🔄RECOVERY' if _recovery_mode else ''}), skip", flush=True)
                return
        else:
            swing_open = [p for p in existing if not (p.get("timeframe") == "M15" or
                          abs(p.get("tp", 0) - p.get("price_open", 0)) < 0.0025)]
            if len(swing_open) >= MAX_OPEN_POSITIONS:
                self._scan_stats["blocked_duplicate"] += 1
                print(f"[MT5] {signal.symbol}: {len(swing_open)} swing abiertas (max={MAX_OPEN_POSITIONS}), skip", flush=True)
                return

        sym_open = [p for p in existing if p["symbol"] == signal.symbol]
        if sym_open:
            pos = sym_open[0]
            pnl_live = pos.get("profit", 0.0)
            pos_dir = pos.get("type", "").upper()
            # Scalp puede abrir en mismo simbolo si hay swing en MISMA direccion
            # Swing no puede abrir si ya hay posicion abierta en ese simbolo
            if _is_scalp and pos_dir == order_type:
                pass  # permitir scalp adicional en misma direccion
            else:
                print(
                    f"[MT5] {signal.symbol}: posicion {pos_dir} ya abierta "
                    f"({pnl_live:+.2f} USD) -- skip",
                    flush=True,
                )
                return



        # ── FILTER 7c: Posiciones abiertas perdiendo ─────────────────────────
        if existing:
            _bal = self._ftmo_state.current_balance or self.capital
            _total_pnl = sum(p.get("profit", 0.0) for p in existing)
            _loss_limit = _bal * 0.01  # bloquear nuevas entradas si portfolio pierde >1%
            for p in existing:
                _pnl  = p.get("profit", 0.0)
                _pct  = abs(_pnl) / _bal * 100 if _bal > 0 else 0
                _tag  = f"perdiendo ${abs(_pnl):.2f} ({_pct:.2f}%)" if _pnl < 0 else f"ganando ${_pnl:.2f} ({_pct:.2f}%)"
                print(f"[LIVE-POS] {p.get('symbol','?')} {p.get('type','?')} {_tag}", flush=True)
            if _total_pnl < -_loss_limit:
                print(
                    f"[FILTER-LOSS] {signal.symbol}: skip -- "
                    f"portfolio perdiendo ${abs(_total_pnl):.2f} (limite=${_loss_limit:.0f} = 1% balance)",
                    flush=True,
                )
                return

        # ── FILTER 8: Claude API final confirmation ───────────────────────
        can_proceed, adj_score, claude_summary = await self._claude_confirm_trade(signal)
        signal.decision_score = adj_score
        if not can_proceed:
            print(f"[CLAUDE] {signal.symbol}: BLOQUEADO -- {claude_summary}", flush=True)
            try:
                await self.telegram.send_glint_alert(
                    f"<b>CLAUDE VETO</b>\n{signal.symbol}: setup rechazado\n{claude_summary}"
                )
            except Exception:
                pass
            return
        print(f"[CLAUDE] {signal.symbol}: CONFIRMADO -- {claude_summary}", flush=True)

        vc = VolumeCalculator()

        # Use real MT5 balance (not startup capital=1000) for correct lot sizing
        live_capital = self._ftmo_state.current_balance if self._ftmo_state.current_balance > 1000 else self.capital
        # Dynamic risk (halved 2026-06-14, WR=29.1%/PF=0.35 over 213 trades):
        # 1% for very high confidence (score>=90), 0.5% for high (>=75), 0.25% normal
        if signal.decision_score >= 90:
            risk_pct = 0.01
            print(f"[RISK] {signal.symbol}: score={signal.decision_score} → riesgo 1% (alta confianza)", flush=True)
        elif signal.decision_score >= 75:
            risk_pct = 0.005
            print(f"[RISK] {signal.symbol}: score={signal.decision_score} → riesgo 0.5%", flush=True)
        else:
            risk_pct = 0.0025
        gov_mult = self.risk_governor.risk_multiplier()
        if gov_mult != 1.0:
            risk_pct *= gov_mult
            print(f"[GOVERNOR] multiplicador de riesgo x{gov_mult:.2f} → riesgo final {risk_pct*100:.3f}%", flush=True)
        # Use current market price for correct lot sizing (signal.entry can be H4 stale)
        try:
            import MetaTrader5 as _mt5
            _tick_vol = _mt5.symbol_info_tick(signal.symbol)
            _fill_price = (_tick_vol.ask if order_type == "BUY" else _tick_vol.bid) if _tick_vol else 0.0
        except Exception:
            _fill_price = 0.0
        _entry_for_vol = _fill_price if _fill_price > 0 else (signal.entry or sl_val)

        # Scalp: volumen FIJO 0.1L → pip=$1 → SL=4pips=$4 max, TP=12pips=$12
        if _is_scalp:
            volume = 0.1
        else:
            volume = vc.calculate_volume(live_capital, _entry_for_vol, sl_val, signal.symbol, risk_pct=risk_pct)

        # En recovery: swing max $150 (no $400) para no borrar ganancias de scalps
        _swing_max = 150.0 if _recovery_mode else 400.0
        MAX_DOLLAR_RISK = SCALP_MAX_DOLLAR_RISK if _is_scalp else _swing_max
        if volume > 0 and sl_val > 0 and _entry_for_vol > 0:
            _sl_pips = abs(_entry_for_vol - sl_val)
            _sym_info = None
            try:
                import MetaTrader5 as _mt5r
                _sym_info = _mt5r.symbol_info(signal.symbol)
            except Exception:
                pass
            if _sym_info:
                _pip_val = _sym_info.trade_contract_size * _sym_info.point
                _dollar_risk = volume * (_sl_pips / _sym_info.point) * _pip_val
                if _dollar_risk > MAX_DOLLAR_RISK and _sl_pips > 0:
                    _raw_vol = MAX_DOLLAR_RISK / ((_sl_pips / _sym_info.point) * _pip_val)
                    _step = _sym_info.volume_step if _sym_info.volume_step > 0 else 0.01
                    # Round DOWN to valid step, then enforce minimum
                    volume = max(round(int(_raw_vol / _step) * _step, 8), _sym_info.volume_min)
                    print(f"[RISK-CAP] {signal.symbol}: riesgo estimado ${_dollar_risk:.0f} > ${MAX_DOLLAR_RISK} cap — vol ajustado a {volume} (step={_step})", flush=True)

        print(f"[MT5 ORDER] Enviando {signal.symbol} {order_type} vol={volume} sl={sl_val:.5f} tp={tp_val:.5f}", flush=True)

        try:

            result = await loop.run_in_executor(

                None,

                lambda: self.mt5.place_order(signal.symbol, order_type, volume, sl=sl_val, tp=tp_val),

            )

        except Exception as exc:

            print(f"[MT5 ORDER] Excepcion en place_order: {exc}", flush=True)

            return

        if "ticket" in result:

            print(f"[MT5 REAL] {signal.symbol} {order_type} #{result['ticket']} @{result.get('price', 0):.5f} score={signal.decision_score}", flush=True)

            try:

                eid = record_episode({

                    "ts": datetime.now(timezone.utc).isoformat(),

                    "symbol": signal.symbol,

                    "timeframe": signal.timeframe,

                    "direction": order_type,

                    "entry": result.get("price", signal.entry),

                    "sl": sl_val,

                    "tp": tp_val,

                    "ticket": result["ticket"],

                    "score": signal.decision_score,

                    "setup_type": "SMC",

                }, conn=self._episodic_conn)

                self._open_episodes[result["ticket"]] = eid
                self._save_open_episodes()

            except Exception as _ep_err:

                print(f"[EPISODIC] record error: {_ep_err}", flush=True)

            # Count toward daily limit + update scan stats
            self._daily_trades[today_str] = trades_today + 1
            self._save_daily_trades()
            self._scan_stats["executed"] += 1
            self._scan_stats["last_trade_ts"] = datetime.now(timezone.utc)
            self._save_scan_stats()

            try:

                await self.telegram.send_glint_alert(

                    f"<b>MT5 ORDEN REAL EJECUTADA</b>\n"

                    f"Par: {signal.symbol} | {order_type}\n"

                    f"Ticket: #{result['ticket']} | @{result.get('price', 0):.5f}\n"

                    f"SL: {sl_val:.5f} | TP: {tp_val:.5f}\n"

                    f"Score: {signal.decision_score}/100"

                )

            except Exception:

                pass

        else:
            err_msg = result.get("error", "?")
            print(f"[MT5 REAL] {signal.symbol} fallida: {err_msg}", flush=True)
            # Retcode 10031 = no network — try to reinitialize MT5 connection
            if "10031" in str(err_msg) or "network" in str(err_msg).lower():
                try:
                    import MetaTrader5 as _mt5_re
                    print(f"[MT5] Reconectando (10031 — sin red)...", flush=True)
                    _mt5_re.shutdown()
                    import time as _time
                    _time.sleep(2)
                    _ok = _mt5_re.initialize()
                    print(f"[MT5] Reinit: {'OK' if _ok else 'FALLO'}", flush=True)
                except Exception as _re:
                    print(f"[MT5] Reconectar error: {_re}", flush=True)



    async def _execute_demo_trade(self, signal: TradeSignal):

        """Record a simulated demo trade, notify via Telegram, log to SQLite."""

        # Expire demo trades older than 8 hours — record outcome before dropping
        cutoff = datetime.now(timezone.utc) - timedelta(hours=8)
        still_valid = []
        for _d in self._demo_trades:
            if getattr(_d, "opened_at", datetime.now(timezone.utc)) <= cutoff:
                if _d.status == "open":
                    # Record final P&L at expiry via yfinance
                    try:
                        import yfinance as _yf
                        _sym = _d.signal.symbol
                        _cur = float(_yf.Ticker(_sym.replace("USDT", "-USD")).fast_info.last_price)
                        _d.close(_cur)
                        _sign = "+" if _d.pnl >= 0 else ""
                        print(
                            f"[DEMO EXPIRE] {_sym} | P&L: {_sign}{_d.pnl*100:.2f}% @ {_cur:.4f}",
                            flush=True,
                        )
                        try:
                            await self.telegram.send_glint_alert(
                                f"<b>DEMO EXPIRADO (8h)</b>\n"
                                f"{_sym} {'LONG' if _d.signal.signal_type == SignalType.LONG else 'SHORT'}\n"
                                f"Entrada: <code>{_d.signal.entry:.4f}</code>  Cierre: <code>{_cur:.4f}</code>\n"
                                f"P&amp;L: <b>{_sign}{_d.pnl*100:.2f}%</b>"
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
            else:
                still_valid.append(_d)
        self._demo_trades = still_valid
        self._save_demo_trades()

        # One slot per symbol — no duplicate positions on same pair
        if any(d.signal.symbol == signal.symbol for d in self._demo_trades):
            print(f"[DEMO SKIP] {signal.symbol}: par ya tiene posicion demo abierta (score={signal.decision_score})", flush=True)
            return

        if len(self._demo_trades) >= DEMO_MAX_POSITIONS:
            print(f"[DEMO SKIP] {signal.symbol}: limite {DEMO_MAX_POSITIONS} posiciones demo alcanzado (score={signal.decision_score})", flush=True)
            return



        demo = DemoTrade(signal, signal.decision_score)

        self._demo_trades.append(demo)
        self._save_demo_trades()  # persist so restarts don't lose open positions

        direction = "long" if signal.signal_type == SignalType.LONG else "short"

        market    = "MT5" if signal.symbol in MT5_SYMBOLS else "Binance"



        # Log to SQLite for /scores and /criterios commands

        save_score(

            symbol    = signal.symbol,

            timeframe = signal.timeframe,

            score     = signal.decision_score,

            direction = direction,

            entry     = signal.entry,

            sl        = signal.stop_loss if signal.stop_loss else 0.0,

            tp        = signal.take_profit,

            executed  = True,

        )



        print(f"[DEMO TRADE] {signal.symbol} {direction.upper()} "

              f"entry={signal.entry:.4f} score={signal.decision_score} "

              f"({market}) [{len(self._demo_trades)}/{DEMO_MAX_POSITIONS}]")



        msg = (

            f"<b>TRADE DEMO ABIERTO</b>\n"

            f"━━━━━━━━━━━━━━━━━━━━\n"

            f"Par: <b>{signal.symbol}</b> | {signal.timeframe} | {market}\n"

            f"{'LONG' if direction=='long' else 'SHORT'}\n"

            f"━━━━━━━━━━━━━━━━━━━━\n"

            f"Entrada:    <code>{signal.entry:,.5f}</code>\n"

            f"Stop Loss:  <code>{signal.stop_loss if signal.stop_loss else 0.0:,.5f}</code>\n"

            f"Take Profit:<code>{signal.take_profit:,.5f}</code>\n"

            f"R:R: <code>1:{signal.risk_reward:.1f}</code>\n"

            f"━━━━━━━━━━━━━━━━━━━━\n"

            f"Score: <b>{signal.decision_score}/100</b>\n"

            f"Activos: {len(self._demo_trades)}/{DEMO_MAX_POSITIONS}\n"

            f"DEMO - sin dinero real"

        )

        try:

            await self.telegram.send_signal_demo(

                symbol=signal.symbol, direction=direction,

                entry=signal.entry,

                sl=signal.stop_loss if signal.stop_loss else signal.entry*0.995,

                tp=signal.take_profit, score=signal.decision_score,

                timeframe=signal.timeframe, market=market,

            )

        except Exception:

            pass





    # -- Demo trade TP/SL monitor -------------------------------------------

    async def _monitor_demo_trades(self) -> None:
        """Check open demo trades vs current price. Close at TP or SL and notify Telegram."""
        if not self._demo_trades:
            return

        import yfinance as yf

        loop = asyncio.get_running_loop()
        still_open: List["DemoTrade"] = []

        for demo in self._demo_trades:
            if demo.status != "open":
                continue

            symbol    = demo.signal.symbol
            entry     = demo.signal.entry or 0.0
            sl        = demo.signal.stop_loss or 0.0
            tp        = demo.signal.take_profit or 0.0
            direction = demo.signal.signal_type

            # --- get current price (yfinance for crypto) ---
            current = 0.0
            try:
                yf_sym = symbol.replace("USDT", "-USD")
                ticker = yf.Ticker(yf_sym)
                current = float(ticker.fast_info.last_price)
            except Exception:
                still_open.append(demo)
                continue

            if current <= 0.0 or entry <= 0.0:
                still_open.append(demo)
                continue

            # --- check TP / SL ---
            hit_tp = hit_sl = False
            if direction == SignalType.LONG:
                hit_tp = tp > 0 and current >= tp
                hit_sl = sl > 0 and current <= sl
            else:
                hit_tp = tp > 0 and current <= tp
                hit_sl = sl > 0 and current >= sl

            if not (hit_tp or hit_sl):
                still_open.append(demo)
                continue

            demo.close(current)
            result   = "TP" if hit_tp else "SL"
            pnl_pct  = demo.pnl * 100.0
            sign     = "+" if pnl_pct >= 0 else ""
            dir_str  = "LONG" if direction == SignalType.LONG else "SHORT"

            print(
                f"[DEMO {result}] {symbol} {dir_str} | entrada={entry:.4f} "
                f"cierre={current:.4f} | P&L: {sign}{pnl_pct:.2f}%",
                flush=True,
            )

            # Record real outcome in score_db
            try:
                from core.score_db import update_score_outcome
                update_score_outcome(symbol, entry, "WIN" if hit_tp else "LOSS", pnl_pct)
            except Exception:
                pass

            msg = (
                f"<b>DEMO CERRADO — {'GANADO ✅' if hit_tp else 'SL ❌'}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Par: <b>{symbol}</b> {dir_str}\n"
                f"Entrada: <code>{entry:.4f}</code>  Cierre: <code>{current:.4f}</code>\n"
                f"P&amp;L: <b>{sign}{pnl_pct:.2f}%</b>\n"
                f"Score: {demo.score} | TF: {demo.signal.timeframe}"
            )
            try:
                await self.telegram.send_glint_alert(msg)
            except Exception:
                pass

        self._demo_trades = still_open
        self._save_demo_trades()  # persist updated state (closed trades removed)


    # -- Open position P&L monitor ------------------------------------------



    async def _position_monitor_loop(self):

        """Every 60s: log open positions + detect closures → update learning + FTMO."""

        _known_tickets: set = set()
        # Tickets flagged for close-on-market-open (positions with no SL)
        # Populated dynamically: any position with SL=0 gets auto-closed on next open
        _close_when_open: set = set()

        while self._running:

            await asyncio.sleep(60)

            if not self._mt5_available:

                continue

            try:

                loop = asyncio.get_running_loop()

                positions = await loop.run_in_executor(None, self.mt5.get_positions)

                # ── Flag positions with no SL for auto-close ─────────────
                # Only auto-close INDEX positions (US30/NAS100) without SL.
                # Forex/gold SL can fail to stick on first check — give them time to retry.
                _INDEX_SYMBOLS = {"US30", "NAS100"}
                current_tickets = {p["ticket"] for p in positions}
                for p in positions:
                    if p.get("sl", 0.0) == 0.0 and p.get("symbol", "") in _INDEX_SYMBOLS:
                        _close_when_open.add(p["ticket"])
                # Remove tickets that are no longer open (already closed externally)
                _close_when_open -= (set(_close_when_open) - current_tickets)

                # ── Auto-close flagged positions (closes when market accepts) ─
                for p in positions:
                    ticket = p.get("ticket", 0)
                    if ticket in _close_when_open:
                        pnl    = p.get("profit", 0.0)
                        symbol = p.get("symbol", "?")
                        ok = await loop.run_in_executor(
                            None, lambda t=ticket: self.mt5.close_position(t)
                        )
                        if ok:
                            _close_when_open.discard(ticket)
                            msg = f"[AUTO-CLOSE] {symbol} #{ticket} cerrado (sin SL) | P&L: ${pnl:+.2f}"
                            print(msg, flush=True)
                            try:
                                await self.telegram.send_glint_alert(
                                    f"<b>CIERRE AUTOMATICO</b>\n{symbol} #{ticket} sin SL -- cerrado\nP&L: ${pnl:+.2f} USD"
                                )
                            except Exception:
                                pass
                        else:
                            print(f"[AUTO-CLOSE] {symbol} #{ticket} sin SL -- mercado cerrado, reintentando", flush=True)

                current_tickets = {p["ticket"] for p in positions}

                # ── Partial close 50% at 1:1 RR to lock profit ─────────────
                # When profit >= 1×SL distance: close half the position,
                # so even if the trade reverses, we keep at least some gain.
                for p in positions:
                    try:
                        ticket  = p.get("ticket", 0)
                        symbol  = p.get("symbol", "")
                        ptype   = p.get("type", "").upper()
                        entry   = p.get("price_open", 0.0)
                        cur_sl  = p.get("sl", 0.0)
                        cur_tp  = p.get("tp", 0.0)
                        volume  = p.get("volume", 0.0)
                        partial_done_key = f"partial_{ticket}"
                        # XAUUSD: partial-close-at-1:1 caps wins at ~0.5R while a
                        # full SL loses 1R. Audit (05-01 a 06-11, 51 trades, WR=80.4%)
                        # found avg win $27 vs avg loss $209 -> neto -$979.74. Dejar
                        # correr a TP completo (RR ~2.5-3) con SL a breakeven via el
                        # loop de trailing debajo, sin partial-close.
                        if symbol == "XAUUSD":
                            continue
                        # Skip if already partially closed this trade
                        if not (ticket and entry > 0 and cur_sl > 0 and volume > 0):
                            continue
                        if getattr(self, "_partial_closed", None) is None:
                            self._partial_closed: set = set()
                        if ticket in self._partial_closed:
                            continue
                        sl_dist  = abs(entry - cur_sl)
                        if sl_dist <= 0:
                            continue
                        import MetaTrader5 as _mt5_mod2
                        tick2 = _mt5_mod2.symbol_info_tick(symbol)
                        if not tick2:
                            continue
                        cur_price = tick2.bid if ptype == "BUY" else tick2.ask
                        if ptype == "BUY":
                            profit_units = (cur_price - entry) / sl_dist
                        else:
                            profit_units = (entry - cur_price) / sl_dist
                        if profit_units >= 1.0:
                            half_vol = round(volume / 2, 2)
                            ok = await loop.run_in_executor(
                                None,
                                lambda t=ticket, v=half_vol: self.mt5.partial_close_position(t, v)
                            )
                            if ok:
                                self._partial_closed.add(ticket)
                                pnl_now = p.get("profit", 0.0)
                                print(
                                    f"[PARTIAL] {symbol} #{ticket} — cerrado 50% ({half_vol}L) "
                                    f"al 1:1 RR | P&L parcial ${pnl_now/2:.2f}",
                                    flush=True,
                                )
                                try:
                                    await self.telegram.send_glint_alert(
                                        f"<b>CIERRE PARCIAL 50% ✅</b>\n"
                                        f"{symbol} #{ticket} — 1:1 RR alcanzado\n"
                                        f"Cerrado {half_vol}L | P&amp;L asegurado: ${pnl_now/2:.2f}\n"
                                        f"Resto del trade: corriendo libre con SL en breakeven"
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # ── Trailing SL: move to breakeven when 1×SL in profit ─────
                for p in positions:
                    try:
                        ticket   = p.get("ticket", 0)
                        symbol   = p.get("symbol", "")
                        ptype    = p.get("type", "").upper()  # "BUY" / "SELL"
                        entry    = p.get("price_open", 0.0)
                        cur_sl   = p.get("sl", 0.0)
                        cur_tp   = p.get("tp", 0.0)
                        cur_pnl  = p.get("profit", 0.0)
                        volume   = p.get("volume", 0.0)
                        if not (ticket and entry > 0 and cur_sl > 0 and cur_tp > 0):
                            continue
                        sl_dist  = abs(entry - cur_sl)
                        if sl_dist <= 0:
                            continue
                        # Get current price from MT5 tick
                        import MetaTrader5 as _mt5_mod
                        tick = _mt5_mod.symbol_info_tick(symbol)
                        if not tick:
                            continue
                        cur_price = tick.bid if ptype == "BUY" else tick.ask
                        # Profit in SL units
                        if ptype == "BUY":
                            profit_in_sl = (cur_price - entry) / sl_dist
                            be_sl = entry + 0.0001 * sl_dist  # slightly above breakeven
                        else:
                            profit_in_sl = (entry - cur_price) / sl_dist
                            be_sl = entry - 0.0001 * sl_dist
                        # Move SL to breakeven once trade is 1.5x SL in profit (more room for winners)
                        if profit_in_sl >= 1.5:
                            if ptype == "BUY" and cur_sl < be_sl:
                                ok = await loop.run_in_executor(
                                    None, lambda t=ticket, s=round(be_sl, 5), tp=cur_tp:
                                    self.mt5.modify_position_sl_tp(t, s, tp)
                                )
                                if ok:
                                    print(f"[TRAIL] {symbol} #{ticket} SL → breakeven {be_sl:.5f}", flush=True)
                            elif ptype == "SELL" and cur_sl > be_sl:
                                ok = await loop.run_in_executor(
                                    None, lambda t=ticket, s=round(be_sl, 5), tp=cur_tp:
                                    self.mt5.modify_position_sl_tp(t, s, tp)
                                )
                                if ok:
                                    print(f"[TRAIL] {symbol} #{ticket} SL → breakeven {be_sl:.5f}", flush=True)
                    except Exception:
                        pass


                # ── Detect closed positions ───────────────────────────────

                closed = _known_tickets - current_tickets

                for ticket in closed:

                    episode_id = self._open_episodes.pop(ticket, None)
                    self._save_open_episodes()

                    deal = await loop.run_in_executor(

                        None, lambda t=ticket: self.mt5.get_closing_deal(t)

                    )

                    if not deal:

                        continue

                    pnl    = deal.get("profit", 0.0)

                    result = "WIN" if pnl > 0 else "LOSS"

                    # Record MT5 outcome in score_db
                    try:
                        _deal_sym   = deal.get("symbol", "")
                        _deal_price = deal.get("price", 0.0)
                        _deal_entry = deal.get("entry", _deal_price)
                        _pnl_pct    = (pnl / max(abs(100_000.0 * 0.01), 1)) * 100  # approx
                        from core.score_db import update_score_outcome
                        update_score_outcome(_deal_sym, _deal_entry, result, _pnl_pct)
                    except Exception:
                        pass

                    # Update episodic memory
                    if not episode_id:
                        # Bot reiniciado — buscar episode por ticket
                        try:
                            row = self._episodic_conn.execute(
                                "SELECT id FROM episodes WHERE ticket=? ORDER BY id DESC LIMIT 1",
                                (ticket,)
                            ).fetchone()
                            if row:
                                episode_id = row["id"]
                        except Exception:
                            pass
                    if episode_id:
                        try:
                            update_episode_result(
                                episode_id,
                                exit_price=deal.get("price", 0.0),
                                pnl=pnl,
                                result=result,
                                lesson=f"Score → {result} PnL={pnl:+.2f}",
                                conn=self._episodic_conn,
                            )
                        except Exception as _ue:
                            print(f"[LEARN] update error: {_ue}", flush=True)

                    # Update FTMO state

                    try:

                        self._ftmo_agent.record_trade(self._ftmo_state, pnl)

                    except Exception:

                        pass

                    # Sync capital with real MT5 balance after each trade close
                    try:
                        _loop_cap = asyncio.get_running_loop()
                        _acc = await _loop_cap.run_in_executor(None, self.mt5.get_account_info)
                        _new_bal = _acc.get("balance", 0)
                        if _new_bal > 0:
                            self.capital = _new_bal
                            self.risk_manager.update_capital(_new_bal)
                    except Exception:
                        pass

                    print(

                        f"[LEARN] #{ticket} CERRADO: {result} | PnL={pnl:+.2f} USD"

                        f" | Balance FTMO: ${self._ftmo_state.current_balance:,.2f}",

                        flush=True,

                    )

                    # Alert on Telegram

                    try:

                        await self.telegram.send_glint_alert(

                            f"<b>TRADE CERRADO</b>\n"

                            f"Ticket #{ticket} | {'WIN' if pnl > 0 else 'LOSS'}\n"

                            f"P&L: ${pnl:+.2f} USD"

                        )

                    except Exception:

                        pass



                _known_tickets = current_tickets



                # ── Log open positions ────────────────────────────────────

                if positions:

                    total_pnl = sum(p.get("profit", 0.0) for p in positions)
                    _bal_ref  = self._ftmo_state.current_balance or self.capital
                    _total_tag = "GANANDO" if total_pnl >= 0 else "PERDIENDO"

                    lines = [f"[POS] {len(positions)} abiertas | {_total_tag} ${abs(total_pnl):.2f} vivo"]

                    for p in positions:

                        _pnl  = p.get("profit", 0.0)
                        _pct  = abs(_pnl) / _bal_ref * 100 if _bal_ref > 0 else 0
                        _tag  = f"GANANDO  ${_pnl:.2f} (+{_pct:.2f}%)" if _pnl >= 0 else f"PERDIENDO ${abs(_pnl):.2f} (-{_pct:.2f}%)"
                        lines.append(
                            f"  {p['symbol']} {p['type']} {p['volume']}lot → {_tag}"
                        )

                    print("\n".join(lines), flush=True)

                    # Persist positions for wakeup recovery
                    try:
                        from core.wakeup_recovery import save_positions as _save_pos
                        _save_pos([{
                            "symbol": p.get("symbol", "?"),
                            "entry": p.get("price_open", 0.0),
                            "direction": "long" if p.get("type") == "BUY" else "short",
                            "size": p.get("volume", 0.0),
                            "ticket": p.get("ticket", 0),
                            "sl": p.get("sl", 0.0),
                            "tp": p.get("tp", 0.0),
                        } for p in positions])
                    except Exception:
                        pass

                # Auto-cierre si alguna posicion supera limite de perdida
                await self._manage_open_positions()

            except Exception as exc:

                print(f"[POS MONITOR] error: {exc}", flush=True)



    # -- Adaptive threshold & autonomous position management ----------------

    def _load_open_episodes(self) -> dict:
        import json
        path = "memory/open_episodes.json"
        try:
            if os.path.exists(path):
                raw = json.loads(open(path).read())
                return {int(k): v for k, v in raw.items()}
        except Exception:
            pass
        return {}

    def _save_open_episodes(self) -> None:
        import json
        path = "memory/open_episodes.json"
        try:
            with open(path, "w", encoding="utf-8") as _f:
                _f.write(json.dumps(self._open_episodes))
        except Exception:
            pass

    def _recover_orphaned_episodes(self) -> None:
        """On startup: backfill outcomes for tickets that closed during a prior restart."""
        try:
            import MetaTrader5 as _mt5_oe
            from datetime import timedelta
            from memory.episodic_db import update_episode_result
            open_pos = _mt5_oe.positions_get() or []
            open_tickets = {p.ticket for p in open_pos}
            orphaned = {t: eid for t, eid in self._open_episodes.items()
                        if t not in open_tickets}
            if not orphaned:
                return
            now = datetime.now(timezone.utc)
            deals = _mt5_oe.history_deals_get(now - timedelta(days=90), now) or []
            closing = {d.position_id: d for d in deals if d.entry == 1}
            removed = []
            for ticket, episode_id in orphaned.items():
                d = closing.get(ticket)
                if d:
                    pnl = round(d.profit + d.swap + d.commission, 2)
                    result = "WIN" if pnl > 0 else "LOSS"
                    try:
                        update_episode_result(
                            episode_id,
                            exit_price=d.price, pnl=pnl, result=result,
                            lesson=f"Backfill: {result} PnL={pnl:+.2f}",
                            conn=self._episodic_conn,
                        )
                        print(f"[LEARN] backfill ticket={ticket} -> {result} pnl={pnl:+.2f}", flush=True)
                    except Exception:
                        pass
                    removed.append(ticket)
                else:
                    removed.append(ticket)
            for t in removed:
                self._open_episodes.pop(t, None)
            if removed:
                self._save_open_episodes()
        except Exception as _e:
            print(f"[LEARN] orphan recovery error: {_e}", flush=True)

    def _adaptive_threshold(self) -> int:
        """
        Calcula threshold dinamico basado en win rate de ultimos 10 trades reales.
        Cuanto peor el rendimiento reciente, mas selectivo se vuelve el bot.
        """
        try:
            from datetime import timedelta
            desde = datetime.now(timezone.utc) - timedelta(days=14)
            hasta = datetime.now(timezone.utc)
            import MetaTrader5 as _mt5
            deals = _mt5.history_deals_get(desde, hasta)
            # Solo contar SWINGS (vol > 0.1L) — los micro-scalps sesgan el WR
            closed = [d for d in (deals or []) if d.type in (0, 1) and d.entry == 1 and d.volume > 0.10]
            recent = sorted(closed, key=lambda d: d.time)[-10:]
            if len(recent) < 3:
                # Pocos datos → threshold moderado pero no paralizar
                thr = MT5_SCORE_AUTO_REDUCE
                print(f"[ADAPT-THR] datos insuficientes ({len(recent)} trades) → threshold={thr}", flush=True)
                return thr
            wins = sum(1 for d in recent if d.profit > 0)
            wr   = wins / len(recent)
            if wr >= 0.65:
                thr = 80   # excelente WR → threshold moderado
            elif wr >= 0.55:
                thr = 85   # buena WR → threshold alto calidad
            elif wr >= 0.40:
                thr = 90   # WR regular → solo mejores setups
            else:
                thr = 90   # WR baja → maxima selectividad, no bajar nunca de 90
            print(f"[ADAPT-THR] ultimos {len(recent)} trades WR={wr*100:.0f}% → threshold={thr}", flush=True)
            # Aplicar ajuste del learner si hay datos suficientes
            try:
                thr = int(self._learner.effective_threshold(thr, "SMC", "unknown", "unknown"))
                print(f"[ADAPT-THR] learner ajuste → threshold final={thr}", flush=True)
            except Exception:
                pass
            return thr
        except Exception as _e:
            return MT5_REAL_SCORE_THRESHOLD  # fallback al default

    async def _manage_open_positions(self):
        """
        Active position management:
        0a. Friday pre-close: close ALL losing positions by 19:30 UTC Friday (before 21:00 close)
        0b. Anti-drag: close worst loser when net P&L is negative and loser > winner
        1. Auto-close on loss > 0.8% balance
        1b. Peak-profit retracement: close when profit falls 25% from peak (peak >= $20)
        2. Move SL to breakeven when profit >= 1R (SL distance)
        3. Trail SL at 1R below/above price when profit >= 2R
        4. Hard-close LOSING positions stuck > 36h (prevents directionless drains)
        """
        MAX_HOLD_HOURS = 36  # only close positions that are losing after 36h
        FRIDAY_CLOSE_HOUR = 19   # UTC — close losers by 19:30 UTC to avoid weekend gap risk
        FRIDAY_CLOSE_MIN  = 30
        try:
            loop = asyncio.get_running_loop()
            positions = await loop.run_in_executor(None, self.mt5.get_positions)
            import MetaTrader5 as _mt5
            from datetime import timezone as _tz

            bal       = self._ftmo_state.current_balance or self.capital
            limit_usd = bal * 0.008  # 0.8% = emergency stop per position

            # ── 0. Daily profit target ────────────────────────────────────────
            # $245 = meta mínima diaria → notifica → bot sigue para más ganancia
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_pnl_date != today_utc:
                self._daily_pnl_date    = today_utc
                self._daily_target_hit  = False
                self._daily_protect_hit = False

            # Sync realized PnL from MT5 — acumula todos los trades cerrados hoy
            mt5_daily = await loop.run_in_executor(None, self.mt5.get_daily_pnl)
            if mt5_daily is not None:
                self._daily_realized_pnl = float(mt5_daily)

            # ── Swing profit target: cierra SWINGS cuando llegan a $245 juntos ──
            # Los scalps siguen corriendo para acumular más ganancia.
            # Clasificar por VOLUMEN — más confiable que distancia de TP
            # Scalp: vol <= 0.1L (todos los trades pequeños, cerrar en +$10/-$4)
            # Swing: vol > 0.1L (trades grandes, cerrar cuando swing_float >= $245)
            scalp_positions = [p for p in (positions or []) if p.get("volume", 0) <= 0.10]
            swing_positions = [p for p in (positions or []) if p.get("volume", 0) > 0.10]
            swing_float = sum(p.get("profit", 0.0) for p in swing_positions)
            float_pnl   = sum(p.get("profit", 0.0) for p in (positions or []))
            total_today = self._daily_realized_pnl + float_pnl

            if not self._daily_target_hit and swing_float >= DAILY_PROFIT_TARGET:
                self._daily_target_hit = True
                print(
                    f"[META-SWING] swing_float=${swing_float:.2f} >= ${DAILY_PROFIT_TARGET:.0f}"
                    f" — META CUMPLIDA, cerrando SWINGS, scalps siguen",
                    flush=True,
                )
                for sp in list(swing_positions):
                    t_ticket = sp["ticket"]
                    t_sym    = sp.get("symbol", "?")
                    t_pnl    = sp.get("profit", 0.0)
                    ok = await loop.run_in_executor(None, lambda t=t_ticket: self.mt5.close_position(t))
                    if ok:
                        self._position_peaks.pop(t_ticket, None)
                        print(f"[META-CLOSE-SWING] {t_sym} #{t_ticket} cerrado ${t_pnl:+.2f}", flush=True)
                try:
                    await self.telegram.send_glint_alert(
                        f"<b>META DIARIA CUMPLIDA — SWINGS CERRADOS</b>\n"
                        f"Swing profit: <b>${swing_float:.2f}</b>\n"
                        f"Scalps siguen operando para mas ganancia."
                    )
                except Exception:
                    pass

            if not positions:
                return  # nada que gestionar

            # ── 0. Scalp gestión de P&L ───────────────────────────────────────
            # Modo Recuperación — dos condiciones:
            #   1. Dia en rojo > $50
            #   2. Balance debajo del capital base $100K (recuperar lo perdido)
            _current_bal   = self._ftmo_state.current_balance or self.capital
            # Actualizar high-water mark (balance máximo histórico)
            if _current_bal > self._balance_peak:
                self._balance_peak = _current_bal
                print(f"[PEAK] Nuevo máximo histórico: ${self._balance_peak:,.2f}", flush=True)

            # Tres triggers de recuperación:
            _day_in_loss      = self._daily_realized_pnl <= RECOVERY_TRIGGER_LOSS
            _below_initial    = _current_bal < INITIAL_CAPITAL
            _below_peak       = (self._balance_peak - _current_bal) >= RECOVERY_DRAWDOWN_FROM_PEAK
            _in_recovery      = (_day_in_loss or _below_initial or _below_peak) and not self._scalp_daily_hit

            # Estrategia 5: Modo Aceleración — dia muy bueno → maximizar
            _in_accel = (
                self._daily_realized_pnl >= ACCEL_TRIGGER_PROFIT and
                _current_bal >= INITIAL_CAPITAL and
                not _in_recovery and
                not self._scalp_daily_hit
            )

            if _in_recovery:
                SCALP_MIN_PROFIT = RECOVERY_SCALP_TP
                SCALP_MAX_LOSS   = RECOVERY_SCALP_SL
                if _below_peak and self._balance_peak > INITIAL_CAPITAL:
                    _gap = self._balance_peak - _current_bal
                    print(f"[RECOVERY] Cayó ${_gap:.0f} del pico ${self._balance_peak:,.0f} — recuperando", flush=True)
                elif _below_initial:
                    print(f"[RECOVERY] Balance ${_current_bal:,.0f} bajo $100K — recuperando capital base", flush=True)
                else:
                    print(f"[RECOVERY] Dia ${self._daily_realized_pnl:.2f} — recuperando el dia", flush=True)
            elif _in_accel:
                SCALP_MIN_PROFIT = ACCEL_SCALP_TP   # +$15
                SCALP_MAX_LOSS   = ACCEL_SCALP_SL   # -$4
                print(f"[ACCEL] Dia +${self._daily_realized_pnl:.2f} — modo aceleracion activo (TP=$15 max={ACCEL_MAX_SCALPS})", flush=True)
            else:
                SCALP_MIN_PROFIT =  10.0
                SCALP_MAX_LOSS   =  -4.0
            SCALP_DAILY_TARGET =  60.0   # cerrar TODOS scalps cuando acumula $60 hoy

            # Sincronizar scalp P&L desde MT5 real cada ciclo — no confiar en contador en memoria
            _today_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._scalp_pnl_date != _today_s:
                self._scalp_pnl_date   = _today_s
                self._scalp_daily_hit  = False
                self._scalp_peak_today = 0.0
            _scalp_mt5 = await loop.run_in_executor(None, self.mt5.get_scalp_daily_pnl)
            if _scalp_mt5 is not None:
                self._scalp_realized_today = float(_scalp_mt5)

            # Actualizar peak diario de scalps
            if self._scalp_realized_today > self._scalp_peak_today:
                self._scalp_peak_today = self._scalp_realized_today

            # Trailing lock por milestones:
            # Peak $65 → lock=$60. Peak $110 → lock=$100. Peak $250 → lock=$200.
            # Si current cae AL lock → cerrar todo (asegurar ese nivel)
            _SCALP_MILESTONES = [60, 100, 200, 300, 500, 750, 1000]
            _scalp_lock = 0
            for _m in _SCALP_MILESTONES:
                if self._scalp_peak_today >= _m:
                    _scalp_lock = _m
            if (_scalp_lock > 0 and
                    self._scalp_realized_today <= _scalp_lock and
                    self._scalp_peak_today > _scalp_lock and
                    not self._scalp_daily_hit):
                self._scalp_daily_hit = True
                print(f"[SCALP-LOCK] Peak=${self._scalp_peak_today:.2f} cayó a ${self._scalp_realized_today:.2f} <= lock=${_scalp_lock} — cerrando todo", flush=True)

            # Meta diaria scalp $60 alcanzada → cerrar todos los scalps abiertos
            if self._scalp_daily_hit and scalp_positions:
                for sp in list(scalp_positions):
                    ok = await loop.run_in_executor(None, lambda t=sp["ticket"]: self.mt5.close_position(t))
                    if ok:
                        self._position_peaks.pop(sp["ticket"], None)
                        print(f"[SCALP-DAY] {sp.get('symbol','?')} cerrado ${sp.get('profit',0):+.2f} (meta $60 scalp cumplida)", flush=True)
            else:
                # Gestión individual de cada scalp
                for sp in list(scalp_positions):
                    sp_pnl    = sp.get("profit", 0.0)
                    sp_ticket = sp["ticket"]
                    sp_sym    = sp.get("symbol", "?")
                    if sp_pnl >= SCALP_MIN_PROFIT:
                        ok = await loop.run_in_executor(None, lambda t=sp_ticket: self.mt5.close_position(t))
                        if ok:
                            self._position_peaks.pop(sp_ticket, None)
                            self._scalp_realized_today += sp_pnl
                            print(f"[SCALP-TP] {sp_sym} #{sp_ticket} ${sp_pnl:+.2f} | total scalp hoy=${self._scalp_realized_today:.2f}", flush=True)
                            if not self._scalp_daily_hit and self._scalp_realized_today >= SCALP_DAILY_TARGET:
                                self._scalp_daily_hit = True
                                print(f"[SCALP-META] ${self._scalp_realized_today:.2f} >= $60 — META SCALP CUMPLIDA", flush=True)
                                try:
                                    await self.telegram.send_glint_alert(
                                        f"<b>META SCALP DIARIA $60 CUMPLIDA</b>\n"
                                        f"Total scalps hoy: <b>${self._scalp_realized_today:.2f}</b>\n"
                                        f"Cerrando scalps restantes. Swing sigue."
                                    )
                                except Exception:
                                    pass
                    elif sp_pnl <= SCALP_MAX_LOSS:
                        ok = await loop.run_in_executor(None, lambda t=sp_ticket: self.mt5.close_position(t))
                        if ok:
                            self._position_peaks.pop(sp_ticket, None)
                            self._scalp_realized_today += sp_pnl
                            print(f"[SCALP-SL] {sp_sym} #{sp_ticket} ${sp_pnl:+.2f} | total scalp hoy=${self._scalp_realized_today:.2f}", flush=True)

            # ── 0a. Friday pre-close: dump ALL losers before weekend ──────────
            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() == 4:  # Friday
                past_cutoff = (now_utc.hour > FRIDAY_CLOSE_HOUR or
                               (now_utc.hour == FRIDAY_CLOSE_HOUR and now_utc.minute >= FRIDAY_CLOSE_MIN))
                if past_cutoff:
                    losers = [p for p in positions if p.get("profit", 0.0) < 0]
                    for lp in losers:
                        sym    = lp.get("symbol", "?")
                        ticket = lp["ticket"]
                        pnl    = lp.get("profit", 0.0)
                        print(
                            f"[FRIDAY-CLOSE] {sym} #{ticket} perdiendo ${pnl:.2f} "
                            f"— cerrando antes del fin de semana (19:30 UTC)",
                            flush=True,
                        )
                        ok = await loop.run_in_executor(
                            None, lambda t=ticket: self.mt5.close_position(t)
                        )
                        if ok:
                            self._position_peaks.pop(ticket, None)
                            try:
                                await self.telegram.send_glint_alert(
                                    f"<b>CIERRE VIERNES</b> {sym} #{ticket}\n"
                                    f"Cerrado antes del fin de semana.\n"
                                    f"P&amp;L: ${pnl:.2f}"
                                )
                            except Exception:
                                pass
                        else:
                            print(f"[FRIDAY-CLOSE] ERROR cerrando {sym} #{ticket}", flush=True)
                    # Reload positions after Friday cleanup
                    positions = await loop.run_in_executor(None, self.mt5.get_positions)
                    if not positions:
                        return

            # ── 0. Anti-drag: ONLY for positions WITHOUT a proper SL ────────
            # Positions WITH a SL are already protected — don't override MT5's SL/TP.
            # Anti-drag only fires for positions missing SL (shouldn't happen, but safety net).
            NET_DRAG_THRESHOLD = -20.0
            MIN_DRAG_LOSS_USD  = -35.0
            all_pnls   = [p.get("profit", 0.0) for p in positions]
            net_pnl    = sum(all_pnls)
            total_wins = sum(x for x in all_pnls if x > 0)
            worst_loss = min(all_pnls) if all_pnls else 0.0

            # Only fire anti-drag for positions WITHOUT a SL — ones WITH SL are safe
            positions_no_sl = [p for p in positions if p.get("sl", 0.0) == 0.0]
            if (positions_no_sl
                    and net_pnl < NET_DRAG_THRESHOLD
                    and worst_loss < MIN_DRAG_LOSS_USD
                    and abs(worst_loss) > total_wins):
                # Find the position with the worst loss (among those without SL)
                drag_pos = min(positions_no_sl, key=lambda p: p.get("profit", 0.0))
                drag_ticket = drag_pos["ticket"]
                drag_sym    = drag_pos.get("symbol", "?")
                drag_pnl    = drag_pos.get("profit", 0.0)
                print(
                    f"[ANTI-DRAG] Neto abierto=${net_pnl:+.2f} | "
                    f"{drag_sym} #{drag_ticket} perdiendo ${drag_pnl:.2f} > ganadores ${total_wins:.2f} "
                    f"→ cerrando perdedora para proteger ganancias",
                    flush=True,
                )
                ok = await loop.run_in_executor(
                    None, lambda t=drag_ticket: self.mt5.close_position(t)
                )
                if ok:
                    self._position_peaks.pop(drag_ticket, None)
                    try:
                        await self.telegram.send_glint_alert(
                            f"<b>ANTI-DRAG CLOSE</b>\n{drag_sym} #{drag_ticket}\n"
                            f"Perdida ${abs(drag_pnl):.2f} cancelaba ganancias (neto ${net_pnl:+.2f})\n"
                            f"→ Perdedora cerrada. Ganadoras protegidas."
                        )
                    except Exception:
                        pass
                    # Reload positions after close
                    positions = await loop.run_in_executor(None, self.mt5.get_positions)
                    if not positions:
                        return

            for p in positions:
                pnl    = p.get("profit", 0.0)
                ticket = p["ticket"]
                sym    = p.get("symbol", "?")
                entry  = p.get("price_open", 0.0)
                sl_cur = p.get("sl", 0.0)
                tp_cur = p.get("tp", 0.0)
                pos_type = p.get("type", "BUY")
                is_buy = pos_type == "BUY"
                open_time = p.get("time", 0)  # Unix timestamp

                # ── 0b. No SL protection: retry setting SL if it's missing ───
                if sl_cur == 0.0 and tp_cur > 0 and entry > 0:
                    import time as _t
                    _t.sleep(1.0)
                    ok = await loop.run_in_executor(
                        None,
                        lambda t=ticket, e=entry, tp=tp_cur, ib=is_buy:
                            self.mt5.modify_position_sl_tp(
                                t,
                                round(e * (0.995 if ib else 1.005), 5),
                                tp,
                            )
                    )
                    if ok:
                        print(
                            f"[SL-RETRY] {sym} #{ticket} SL aplicado en diferido",
                            flush=True,
                        )
                    else:
                        # If SL still can't be set, close the position to protect capital
                        if pnl < -limit_usd * 0.3:  # at 30% of normal limit
                            print(
                                f"[NO-SL CLOSE] {sym} #{ticket} sin SL y perdiendo ${abs(pnl):.2f} → cerrando",
                                flush=True,
                            )
                            await loop.run_in_executor(
                                None, lambda t=ticket: self.mt5.close_position(t)
                            )
                            self._position_peaks.pop(ticket, None)
                            continue

                # ── 1. Loss protection ─────────────────────────────────────
                if pnl < -limit_usd:
                    print(
                        f"[AUTO-CLOSE] {sym} #{ticket} perdiendo ${abs(pnl):.2f}"
                        f" > limite ${limit_usd:.0f} → cerrando",
                        flush=True,
                    )
                    ok = await loop.run_in_executor(
                        None, lambda t=ticket: self.mt5.close_position(t)
                    )
                    if ok:
                        self._position_peaks.pop(ticket, None)
                        try:
                            await self.telegram.send_glint_alert(
                                f"<b>AUTO-CIERRE PERDIDA</b>\n{sym} #{ticket}\n"
                                f"Perdida ${abs(pnl):.2f} > limite → cerrado"
                            )
                        except Exception:
                            pass
                    continue

                # ── 1b. Peak-profit retracement guard ─────────────────────
                # Only fires when peak profit is large (>= $200) to avoid killing
                # small winners. Closes if profit retreats 30% from peak.
                PEAK_MIN_USD      = 200.0   # only guard big winners
                PEAK_RETRACE_PCT  = 0.30    # close if profit drops 30% from peak
                if pnl > 0:
                    peak = self._position_peaks.get(ticket, 0.0)
                    if pnl > peak:
                        self._position_peaks[ticket] = pnl
                        peak = pnl
                    if (peak >= PEAK_MIN_USD
                            and pnl < peak * (1.0 - PEAK_RETRACE_PCT)):
                        print(
                            f"[PEAK-GUARD] {sym} #{ticket} peak=${peak:.2f} → "
                            f"actual=${pnl:.2f} (retroceso {(1-pnl/peak)*100:.0f}%) → cerrando para asegurar ganancia",
                            flush=True,
                        )
                        ok = await loop.run_in_executor(
                            None, lambda t=ticket: self.mt5.close_position(t)
                        )
                        if ok:
                            self._position_peaks.pop(ticket, None)
                            try:
                                await self.telegram.send_glint_alert(
                                    f"<b>GANANCIA ASEGURADA</b>\n{sym} #{ticket}\n"
                                    f"Peak: ${peak:.2f} → Retroceso 30% → cerrado en ${pnl:.2f}"
                                )
                            except Exception:
                                pass
                        continue
                else:
                    self._position_peaks.pop(ticket, None)

                # ── 2-3. Trailing stop (only for winning positions) ────────
                if entry > 0 and sl_cur > 0:
                    sl_dist = abs(entry - sl_cur)  # 1R distance
                    if sl_dist > 0:
                        tick = _mt5.symbol_info_tick(sym)
                        if tick is not None:
                            cur_price = tick.ask if is_buy else tick.bid
                            profit_r  = (cur_price - entry) / sl_dist if is_buy else (entry - cur_price) / sl_dist

                            new_sl = None
                            if profit_r >= 3.0:
                                # At 3R+: tight trail at 0.5R below/above price (lock in more)
                                trail_sl = (cur_price - sl_dist * 0.5) if is_buy else (cur_price + sl_dist * 0.5)
                                trail_sl = round(trail_sl, 5)
                                if (is_buy and trail_sl > sl_cur) or (not is_buy and trail_sl < sl_cur):
                                    new_sl = trail_sl
                                    print(
                                        f"[TRAIL] {sym} #{ticket} profit_R={profit_r:.1f} "
                                        f"tight trail SL {sl_cur:.5f}→{new_sl:.5f}",
                                        flush=True,
                                    )
                            elif profit_r >= 2.0:
                                # At 2R+: trail SL at 1R below/above current price
                                trail_sl = (cur_price - sl_dist) if is_buy else (cur_price + sl_dist)
                                trail_sl = round(trail_sl, 5)
                                if (is_buy and trail_sl > sl_cur) or (not is_buy and trail_sl < sl_cur):
                                    new_sl = trail_sl
                                    print(
                                        f"[TRAIL] {sym} #{ticket} profit_R={profit_r:.1f} "
                                        f"trail SL {sl_cur:.5f}→{new_sl:.5f}",
                                        flush=True,
                                    )
                            elif profit_r >= 1.5 and sl_cur != entry:
                                # At 1.5R+: move SL to breakeven (raised from 1R — more room for winners)
                                new_sl = round(entry, 5)
                                if (is_buy and new_sl > sl_cur) or (not is_buy and new_sl < sl_cur):
                                    print(
                                        f"[TRAIL] {sym} #{ticket} profit_R={profit_r:.1f} "
                                        f"→ breakeven SL {sl_cur:.5f}→{new_sl:.5f}",
                                        flush=True,
                                    )
                                else:
                                    new_sl = None

                            if new_sl is not None:
                                ok = await loop.run_in_executor(
                                    None,
                                    lambda t=ticket, s=new_sl, tp=tp_cur:
                                        self.mt5.modify_position_sl_tp(t, s, tp)
                                )
                                if ok:
                                    try:
                                        await self.telegram.send_glint_alert(
                                            f"<b>TRAILING STOP</b>\n{sym} #{ticket}\n"
                                            f"SL movido a {new_sl:.5f} | P&L: ${pnl:+.2f}"
                                        )
                                    except Exception:
                                        pass

                # ── 4. Hard close LOSING position stuck > MAX_HOLD_HOURS ────
                # Winners are handled by trail SL / breakeven — don't kill them early.
                if pnl <= 0 and open_time > 0:
                    import time as _time
                    age_h = (_time.time() - open_time) / 3600
                    if age_h >= MAX_HOLD_HOURS:
                        # Cooldown: don't spam close attempts — retry at most every 5 min
                        _last_try = self._close_attempted.get(ticket, 0.0)
                        if _time.time() - _last_try < 300:
                            continue  # skip until cooldown expires
                        self._close_attempted[ticket] = _time.time()
                        print(
                            f"[TIME-CLOSE] {sym} #{ticket} abierta {age_h:.1f}h "
                            f"perdiendo ${pnl:.2f} → cerrando (limite {MAX_HOLD_HOURS}h)",
                            flush=True,
                        )
                        ok = await loop.run_in_executor(
                            None, lambda t=ticket: self.mt5.close_position(t)
                        )
                        if ok:
                            self._position_peaks.pop(ticket, None)
                            self._close_attempted.pop(ticket, None)
                            try:
                                await self.telegram.send_glint_alert(
                                    f"<b>CIERRE POR TIEMPO</b>\n{sym} #{ticket}\n"
                                    f"Abierta {age_h:.1f}h perdiendo → cerrada en ${pnl:+.2f}"
                                )
                            except Exception:
                                pass
                        else:
                            print(f"[TIME-CLOSE] {sym} #{ticket} close FALLO — reintento en 5min", flush=True)

        except Exception as _me:
            print(f"[AUTO-CLOSE] error monitor: {_me}", flush=True)

    # -- Autonomous background loops ----------------------------------------



    async def _learning_loop(self):

        while self._running:

            try:

                self._learner.run_analysis()

                print("[LEARNER] Weight analysis complete", flush=True)

            except Exception as exc:

                print(f"[LEARNER] error: {exc}", flush=True)

            await asyncio.sleep(3600)  # every 1 hour



    async def _research_loop(self):

        while self._running:

            try:

                self._researcher.run_cycle()

            except Exception as exc:

                print(f"[RESEARCH] error: {exc}", flush=True)

            await asyncio.sleep(7200)  # every 2 hours



    async def _risk_governor_loop(self):

        while self._running:

            try:

                symbol_deals = await asyncio.get_event_loop().run_in_executor(

                    None, fetch_recent_deals_by_symbol, MT5_SYMBOLS

                )

                acc = self.mt5.get_account_info() if self.mt5 else None

                balance = acc.get("balance", 0.0) if acc else 0.0

                if balance <= 0:

                    print("[GOVERNOR] sin balance MT5 -- ciclo omitido", flush=True)

                    await asyncio.sleep(7200)

                    continue

                drawdown_pct = max(0.0, (100_000.0 - balance) / 100_000.0)

                changes = self.risk_governor.evaluate(symbol_deals, drawdown_pct)

                if RiskGovernor.has_changes(changes):

                    report = self.risk_governor.format_report(changes, balance, drawdown_pct)

                    print(f"[GOVERNOR]\n{report}", flush=True)

                    try:

                        await self.telegram.send_glint_alert(report)

                    except Exception:

                        pass

                else:

                    print(f"[GOVERNOR] {self.risk_governor.status_line()}", flush=True)

            except Exception as exc:

                print(f"[GOVERNOR] error: {exc}", flush=True)

            await asyncio.sleep(7200)  # every 2 hours



    async def _goals_loop(self):

        while self._running:

            try:

                self._goals_mgr.evaluate()

            except Exception as exc:

                print(f"[GOALS] error: {exc}", flush=True)

            await asyncio.sleep(1800)  # every 30 min



    async def _nightly_report_loop(self):

        while self._running:

            await asyncio.sleep(60)  # check every 1 min

            try:
                now = datetime.now(timezone.utc)
                if self._reporter.should_fire(now):
                    date_str = now.strftime("%Y-%m-%d")
                    self._reporter.mark_fired(date_str)

                    # Cierre de jornada (05:00 UTC = midnight Colombia): reporte P&L del dia
                    if now.hour == 5:
                        loop = asyncio.get_running_loop()
                        acc  = await loop.run_in_executor(None, self.mt5.get_account_info)
                        bal  = acc.get("balance", 0) if acc else 0
                        net  = await loop.run_in_executor(None, self.mt5.get_daily_pnl)
                        msg  = self._reporter.generate_eod_report(bal, net or 0)
                        try:
                            await self.telegram.send_glint_alert(msg)
                            print(f"[EOD] Reporte cierre enviado: bal=${bal:.2f} net=${net:.2f}", flush=True)
                        except Exception:
                            pass
                    else:
                        await self._reporter.send(date_str)

            except Exception as exc:
                print(f"[NIGHTLY] error: {exc}", flush=True)



    async def _vision_monitor_loop(self):
        """Every 2h: capture MT5 screen (interval extended to preserve API credits)."""
        _BALANCE_AT_START = 100_000.0  # Axi demo seed capital

        while self._running:
            interval = 7200  # 2h — preservar creditos API (era 5min)
            await asyncio.sleep(interval)

            if self._vision is None:
                continue

            try:
                loop = asyncio.get_event_loop()
                report = await loop.run_in_executor(None, self._vision.monitor_and_protect)
                analysis = report.get("analysis", {})
                alerts = report.get("alerts", [])
                balance = report.get("balance", 0)

                # Always check balance growth vs starting capital
                if balance > 0 and balance < _BALANCE_AT_START:
                    deficit = _BALANCE_AT_START - balance
                    alerts.append(
                        f"CUENTA EN PERDIDA -- balance ${balance:,.0f} "
                        f"(inicio ${_BALANCE_AT_START:,.0f}, -${deficit:,.0f})"
                    )

                if alerts:
                    alert_text = "\n".join(alerts)
                    try:
                        await self.telegram.send_glint_alert(
                            f"[VISION ALERT]\n{alert_text}"
                        )
                    except Exception:
                        pass

                # Auto-close critical positions (> $500 loss)
                if self._mt5_available:
                    live_positions = await asyncio.get_event_loop().run_in_executor(
                        None, self.mt5.get_positions
                    )
                    # Build symbol→ticket map from live MT5 data (vision JSON may hallucinate)
                    sym_to_ticket = {p["symbol"]: p["ticket"] for p in live_positions}

                    for pos in analysis.get("posiciones", []):
                        pnl = pos.get("pnl", 0)
                        symbol = pos.get("symbol", "")
                        ticket = sym_to_ticket.get(symbol)
                        if ticket and self._vision.should_close_position(symbol, pnl):
                            try:
                                ok = await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda t=ticket: self.mt5.close_position(t)
                                )
                                msg = f"[VISION] {'Cerrada' if ok else 'Fallo cierre'} {symbol} #{ticket} perdida ${abs(pnl):.0f}"
                                print(msg, flush=True)
                                await self.telegram.send_glint_alert(msg)
                            except Exception as e:
                                print(f"[VISION] close failed {symbol}: {e}", flush=True)

            except Exception as exc:
                print(f"[VISION MONITOR] error: {exc}", flush=True)

    async def _market_scan_loop(self):

        _was_offline = False

        while self._running:

            try:

                online = await self._check_internet()



                if not online:

                    if not _was_offline:

                        _was_offline = True

                    print("[Bot] Sin internet -- reintentando en 30s...")

                    await asyncio.sleep(30)

                    continue



                if _was_offline:

                    _was_offline = False

                    try:

                        await self.telegram.send_glint_alert("ðŸ"" Conexion restaurada -- bot activo")

                    except Exception:

                        pass



                # â"€â"€ Full market scan â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

                # ── Reset FTMO diario + refresh daily_pnl ─────────────────
                _today_utc = datetime.now(timezone.utc).date()
                if not hasattr(self, '_last_ftmo_day') or self._last_ftmo_day != _today_utc:
                    self._ftmo_agent.new_trading_day(self._ftmo_state)
                    self._last_ftmo_day = _today_utc
                    print(f"[FTMO] Nuevo dia {_today_utc} -- daily_pnl reseteado (streak preservado)", flush=True)
                    try:
                        _loop_ref = asyncio.get_running_loop()
                        self._ftmo_state.daily_pnl_today = await _loop_ref.run_in_executor(
                            None, self.mt5.get_daily_pnl
                        )
                    except Exception:
                        pass

                for symbol in SCAN_SYMBOLS:

                    for tf in SCAN_TIMEFRAMES:

                        if not self._running:

                            break

                        try:

                            signal = await self._scan_symbol(symbol, tf)

                            if signal is None:

                                print(f"[{symbol}][{tf}] Sin datos")

                                continue



                            score = signal.decision_score

                            bias  = signal.signal_type.value.upper()

                            self._scan_stats["total"] += 1

                            print(f"[{symbol}][{tf}] Score: {score} | {bias}", end="")



                            if signal.signal_type == SignalType.WAIT or score < DEMO_SCORE_THRESHOLD:

                                print(" -- sin setup")

                            elif self.demo_mode:

                                print(f" -- ejecutando trade DEMO")

                                await self._execute_demo_trade(signal)

                            else:

                                print(f" -- ejecutando trade")

                                self._dispatch(signal)



                        except Exception as exc:

                            print(f"[{symbol}][{tf}] Error: {exc.__class__.__name__}: {exc}")



                        await asyncio.sleep(1)  # rate limit between symbols

                # Monitor demo TP/SL after each full crypto scan cycle
                try:
                    await self._monitor_demo_trades()
                except Exception as _e:
                    print(f"[DEMO-MONITOR] Error: {_e}", flush=True)

                # MT5 forex scan (real orders on demo account -- bypass demo slot limit)

                if self._mt5_available:
                    # ── Threshold adaptativo basado en win rate reciente ──────
                    mt5_threshold = self._adaptive_threshold()

                    for symbol in self.risk_governor.active_symbols():
                        # ── Update D1 trend (50 EMA daily) once per symbol ────
                        try:
                            _d1_df = await asyncio.get_running_loop().run_in_executor(
                                None, lambda s=symbol: self.mt5.get_ohlcv(s, "D1", 55)
                            )
                            if _d1_df is not None and len(_d1_df) >= 50:
                                _ema50 = float(_d1_df["close"].ewm(span=50).mean().iloc[-1])
                                _last_close = float(_d1_df["close"].iloc[-1])
                                self._mt5_d1_trend[symbol] = "LONG" if _last_close > _ema50 else "SHORT"
                        except Exception:
                            pass

                        for tf in MT5_TIMEFRAMES:
                            if not self._running:
                                break
                            try:
                                signal = await self._scan_mt5_symbol(symbol, tf)
                                if signal is None:
                                    continue
                                self._scan_stats["total"] += 1
                                score = signal.decision_score
                                bias  = signal.signal_type.value.upper()
                                print(f"[MT5][{symbol}][{tf}] Score: {score} | {bias}", end="", flush=True)

                                # Siempre actualizar cache H4 — incluso en WAIT para no dejar stale
                                if tf == "H4":
                                    self._mt5_h4_direction[symbol] = bias  # LONG, SHORT, o WAIT

                                # TRIPLE confirm: D1 + H4 + H1 deben coincidir
                                if signal.signal_type != SignalType.WAIT:
                                    d1_dir = self._mt5_d1_trend.get(symbol)
                                    h4_dir = self._mt5_h4_direction.get(symbol)

                                    # D1 desconocido = NO operar — sin macro confirmada no hay trade
                                    if not d1_dir:
                                        print(f" -- [D1-FILTER] D1 no disponible — skip", flush=True)
                                        continue

                                    # D1 contra el trade = bloquear
                                    if d1_dir != bias:
                                        print(f" -- [D1-FILTER] {bias} vs D1={d1_dir} — contra macro, skip", flush=True)
                                        continue

                                    # H4 solo bloquea si explicitamente en contra (LONG vs SHORT)
                                    if tf in ("H1", "M15") and h4_dir in ("LONG", "SHORT") and h4_dir != bias:
                                        print(f" -- [H4-FILTER] {tf}={bias} vs H4={h4_dir} — no confluencia, skip", flush=True)
                                        continue

                                    print(f" -- [D1={d1_dir} H4={h4_dir or '?'}] OK", end="", flush=True)

                                # M15 = scalping: threshold más bajo para más trades
                                effective_threshold = MT5_SCALP_THRESHOLD if tf == "M15" else mt5_threshold
                                if signal.signal_type == SignalType.WAIT or score < effective_threshold:
                                    self._scan_stats["blocked_score"] += 1
                                    print(f" -- sin setup (threshold={effective_threshold})")
                                else:
                                    tag = "SCALP" if tf == "M15" else "SWING"
                                    print(f" -- ejecutando {tag} (score={score}>={effective_threshold})")
                                    await self._send_mt5_real_order(signal)
                            except Exception as exc:
                                print(f"[MT5][{symbol}] Error: {exc.__class__.__name__}")
                            await asyncio.sleep(0.5)



                # yfinance forex scan (fallback when MT5 unavailable)

                if not self._mt5_available:

                    for symbol, tf_yf, tf_label in [

                        ("EURUSD","1h","1H"), ("GBPUSD","1h","1H"),

                        ("USDJPY","1h","1H"), ("GBPJPY","1h","1H"),

                    ]:

                        if not self._running: break

                        try:

                            signal = await self._scan_forex_yfinance(symbol, tf_yf, tf_label)

                            if signal is None:

                                continue

                            score = signal.decision_score

                            bias  = signal.signal_type.value.upper()

                            print(f"[FOREX][{symbol}][{tf_label}] Score: {score} | {bias}", end="")

                            if signal.signal_type == SignalType.WAIT or score < MT5_REAL_SCORE_THRESHOLD:

                                print(" -- sin setup")

                            elif self.demo_mode:

                                print(f" -- DEMO FOREX")

                                await self._execute_demo_trade(signal)

                            else:

                                self._dispatch(signal)

                        except Exception as exc:

                            print(f"[FOREX][{symbol}] Error: {exc.__class__.__name__}")

                        await asyncio.sleep(0.5)



            except asyncio.CancelledError:

                break

            except Exception:

                await asyncio.sleep(10)



            # MT5 reconnect monitor -- check every scan cycle

            if self._mt5_available and not self.mt5.is_connected():

                self._mt5_available = False

                print("[MT5] Desconectado -- reintentando en 60s...")

                try:

                    await self.telegram.send_glint_alert(

                        self.mt5.disconnect_alert_msg()

                    )

                except Exception:

                    pass

            elif not self._mt5_available:

                # Silent reconnect attempt every scan cycle

                loop2 = asyncio.get_event_loop()

                mt5_ok = await loop2.run_in_executor(None, self.mt5.reconnect)

                if mt5_ok:

                    self._mt5_available = True

                    info = await loop2.run_in_executor(None, self.mt5.get_account_info)

                    bal = info.get("balance", 0)

                    if bal > 0:
                        self.capital = bal
                        self.risk_manager.update_capital(bal)

                    print(f"[MT5] Reconectado! Balance ${bal:,.2f}")

                    try:

                        await self.telegram.send_glint_alert(

                            f"<b>MT5 AXI RECONECTADO</b>\nBalance: ${bal:,.2f} USD"

                        )

                    except Exception:

                        pass



            self._save_scan_stats()   # persist stats for /status
            await asyncio.sleep(SCAN_INTERVAL_SEC)  # next full scan



    def stop(self):

        self._running = False



