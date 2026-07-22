import asyncio
import json
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

from strategies.event_driven import EventDrivenStrategy

from connectors.economic_calendar import currencies_for_symbol, get_high_impact_window

from smc.liquidity_sweep import check_setup as _silver_bullet_check



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

from strategies.ftmo_agent import FTMOAgent, ChallengeType, ChallengeState, ChallengeStatus, FTMORules

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
from agents.axi_select_guard import AxiSelectGuard
from agents.axi_select_tracker import AxiSelectTracker
from agents.axi_capital_adjuster import AxiCapitalAdjuster
from agents.consistency_enforcer import ConsistencyEnforcer
from memory.episodic_db import query_similar_episodes



# Score thresholds

DEMO_SCORE_THRESHOLD     = 999
DEMO_MAX_POSITIONS       = 0
SCAN_INTERVAL_SEC        = 30
CONSERVATIVE_MODE        = False
CONSERVATIVE_SCORE_MIN   = 75
CONSERVATIVE_PAIRS       = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY"]

# ── CONFIGURACIÓN SIMPLE — solo lo esencial ──────────────────────────
# UN solo modo: SCALP M15 con 0.1L
# Score >= 85 (calidad alta), TP=$10, SL=-$4, max 10 simultáneas
# Sin swings, sin recovery modes, sin aceleración — solo scalps limpios
MT5_REAL_SCORE_THRESHOLD = 95   # techo absoluto WR<40% (fallback en excepciones) — backtest 2026-07-01: 95 NO mejora WR vs 80, solo reduce volumen
MT5_SCALP_THRESHOLD      = 105  # subido de 100→105
MT5_SCORE_AUTO_REDUCE    = 80   # recalibrado 2026-07-01: barrido thr x RR en 2 años reales muestra 80+RR3.0 = optimo (WR=41.7%, P(pasar 5%)=28.4% vs 8.5% con 90-95)
MT5_SCORE_REDUCE_AFTER_H = 4
MAX_SCALP_POSITIONS      = 2    # max 2 simultáneos (era 3)
MAX_OPEN_POSITIONS       = 3    # 2026-07-17: backtest_multiyear.py confirmo 2 veces
                                 # (sesiones separadas) que MAX_OPEN=3 supera a 2:
                                 # P(mes>=5%) 44%->49%, E[mensual] $4104->$5287.
                                 # Subido de nuevo (era 3 originalmente, se bajo a 2 sin
                                 # evidencia registrada de por que).
MIN_RR                   = 3.0  # subido de 2.5→3.0: con WR=32% necesitas RR>=3 para ser rentable
DAILY_PROFIT_TARGET      = 250.0  # $250/dia → 5% mensual Axi Select
INITIAL_CAPITAL          = 100_000.0

# Recovery — simplificado: solo para emergencias
RECOVERY_SCALP_TP        = 10.0  # igual que normal
RECOVERY_SCALP_SL        = -4.0  # igual que normal
RECOVERY_MAX_SCALPS      = 3
RECOVERY_TRIGGER_LOSS    = -150.0  # recovery si pierde $150 en el día (era -50: demasiado agresivo)
RECOVERY_DRAWDOWN_FROM_PEAK = 3000.0  # Axi 5% daily = $4,850 — solo recovery en emergencia real
ACCEL_TRIGGER_PROFIT     = 50.0   # aceleración si gana $50 en el día
ACCEL_SCALP_TP           = 10.0
ACCEL_SCALP_SL           = -4.0
ACCEL_MAX_SCALPS         = 5
SCALP_MAX_DOLLAR_RISK    = 50.0

# Horas bloqueadas — backtest 2 años (700 dias, 6 pares) demuestra:
# 13:00 UTC = WR 29%, avg -$97/trade (1,223 trades) → SEÑALES RANCIAS overnight → BLOQUEAR
# 14:00 UTC = WR 61%, avg +$102/trade (578 trades) → NY open GOLD window
# 17-19 UTC  = WR 24-28%, avg -$102 a -$120 → POST-NY fading, reducir trades
# Estrategia: iniciar a 14:00 UTC (9am Colombia) cuando NY open momentum confirma BOS real
DEAD_HOURS_UTC           = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                             17, 18, 19}  # 14-17 UTC ventana de oro. 17-19 WR=24-28% bloqueado



# Symbols and timeframes to scan

SCAN_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]  # reactivado 2026-07-06: usuario confirmo que quiere cripto operando en Binance

SCAN_TIMEFRAMES = ["4h", "1h"]  # 4h first so H4 trend is cached before 1h filter runs



# MT5 forex/indices symbols — expanded for more opportunities
# Asian pairs (AUDUSD, USDJPY, GBPJPY) active 00-09 UTC
# European pairs (EURUSD, GBPUSD) active 07-16 UTC
# US indices (NAS100, US30) active 13-20 UTC
# Gold (XAUUSD) active 07-20 UTC

# Universo completo de pares MT5 (usado para enrutar señales MT5 vs Binance).
# La lista de pares ACTIVAMENTE escaneados la decide RiskGovernor en tiempo
# real (self.risk_governor.active_symbols()) — ver core/risk_governor.py.
MT5_SYMBOLS      = ["USDCAD", "EURUSD", "NZDUSD", "USDCHF", "EURAUD", "GBPCAD"]  # ampliado 2026-07-05: screening backtest 2y (scripts/backtest_new_pairs_screen.py) con las mismas reglas del bot (thr=80/90, RR=3.0, partial+BE@1R) mostro edge positivo real: USDCHF (+$7,296 WR=58%), EURAUD (+$3,828 WR=56%), GBPCAD (+$3,023 WR=58%) en 2 anos -- EURCAD/EURGBP/AUDCAD tambien probados y rechazados (negativos)
# GBPUSD removido 2026-07-09: auditoria de episodes.db (591 trades reales, toda la historia)
# mostro GBPUSD como el PEOR par activo -- 147 trades, WR=25.9%, PF=0.53, neto -$887.55 (la
# mayor perdida individual de cualquier simbolo). EURUSD es el UNICO par con edge real
# demostrado (PF=1.11, +$130.86). Sin GBPUSD, el conjunto de pares activos con datos
# suficientes pasa de PF negativo a PF~1.01.
MT5_TIMEFRAMES   = ["H4", "H1"]  # H4 swing principal | H1 swing adicional | M15 scalps DESACTIVADOS (destruian capital)

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

        self.event_driven   = EventDrivenStrategy()

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
        self._mt5_h4_direction: Dict[str, str] = {}       # symbol → "LONG" | "SHORT" | "WAIT"
        self._mt5_h4_just_confirmed: Dict[str, bool] = {}  # symbol → True si H4 acaba de confirmar
        self._mt5_d1_trend: Dict[str, str] = {}            # symbol → "LONG" | "SHORT" (D1 50EMA)

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
            # BUG-DD-TIERS-DISABLED (2026-07-21, risk-management expert panel):
            # dd_tiers=() meant risk_multiplier() was provably always 1.0 no
            # matter how close the account got to the 8% total drawdown
            # ceiling (5.6% safety-block on new entries, strategies/ftmo_agent.py)
            # -- full size right up to a binary hard stop, no graduated taper.
            # Tiers below cut size well before the 5.6% new-entry block: by
            # 6% drawdown, size is already at 25%, so the hard stop at 5.6%
            # is reached with sizing already well reduced, not at full risk.
            dd_tiers=(
                (0.06, 0.25),
                (0.045, 0.50),
                (0.03, 0.75),
            ),
            min_trades=30,       # necesita 30 trades antes de suspender (evita suspension por 1 mal dia)
            suspend_wr=0.10,     # solo suspende si WR < 10% sostenido
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

        # Risk-gate (Axi Select rules) enforcement

        # Axi Select runtime guards
        self._axi_guard      = AxiSelectGuard()
        self._axi_tracker    = AxiSelectTracker()
        self._axi_adjuster   = AxiCapitalAdjuster()
        self._axi_enforcer   = ConsistencyEnforcer()
        # BUG-AXI-GUARD-RESTART: paused-today state now lives in AxiSelectGuard
        # itself (persisted to disk) so a pm2 restart mid-day can't silently
        # un-pause the bot or reset the -4% loss baseline. See axi_select_guard.py.

        self._risk_gate_agent = FTMOAgent()

        # Bug found 2026-07-07: this used to build FTMO's TWO_STEP rules
        # (10% total DD, 5% daily loss) even though the user only trades
        # Axi Select, not FTMO. The engine (drawdown/daily-loss/consecutive-
        # loss gating, well tested) is generic -- only the numbers were
        # wrong. Now built directly with Axi Select's real risk limits
        # (agents/axi_select_agent.py: MAX_TOTAL_DRAWDOWN_PCT=8%,
        # MAX_DAILY_DRAWDOWN_PCT=3%). profit_target_pct is set unreachably
        # high because Axi Select is an ongoing funded account, not a
        # timed pass/fail challenge -- it should never enter PASSED status.
        _risk_gate_rules = FTMORules(
            challenge_type=ChallengeType.TWO_STEP,
            initial_balance=100_000.0,  # Axi Select account size — NOT startup capital param
            profit_target_pct=100.0,    # unreachable -- Axi has no challenge "pass" state
            max_daily_loss_pct=0.03,    # Axi Select real limit (was FTMO's 5%)
            max_total_drawdown_pct=0.08,  # Axi Select real limit (was FTMO's 10%)
            trailing_drawdown=False,
            min_trading_days=0,
            profit_split_pct=0.80,
        )
        self._risk_gate_state = ChallengeState(
            rules=_risk_gate_rules,
            status=ChallengeStatus.ACTIVE,
            current_balance=100_000.0,
            start_date=datetime.now(timezone.utc).date(),
            trading_days=0,
            daily_pnl_today=0.0,
            total_pnl=0.0,
            max_drawdown_reached_pct=0.0,
            consecutive_losses=0,
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
        # Symbol cooldown tras SL: (symbol, direction) → timestamp del SL
        # Persiste en disco para sobrevivir reinicios
        _sl_time_file = os.path.join("memory", "sl_cooldown_state.json")
        try:
            _sl_raw = json.load(open(_sl_time_file)) if os.path.exists(_sl_time_file) else {}
            import time as _tsl; _now = _tsl.time()
            self._symbol_sl_time: Dict[str, float] = {k: v for k, v in _sl_raw.items() if _now - v < 14400}
        except Exception:
            self._symbol_sl_time: Dict[str, float] = {}
        # Daily profit target tracking
        self._daily_pnl_date: str = ""           # "YYYY-MM-DD" UTC
        self._daily_realized_pnl: float = 0.0   # closed trades today
        self._daily_target_hit: bool = False     # DAILY_PROFIT_TARGET hit — day locked, no new trades
        self._daily_protect_hit: bool = False    # reserved (unused)
        # Cumulative/total drawdown emergency force-close — fires once, does
        # NOT reset daily (unlike AxiSelectGuard) since a total-DD breach
        # means the challenge is already failing, not "wait for tomorrow"
        self._dd_force_closed: bool = False
        # ticket -> intended dollar risk at open time (swings only), used to
        # scale LOSS-LIMIT proportionally instead of a flat balance percentage
        self._position_intended_risk: Dict[int, float] = {}
        # Scalp daily target: $60 acumulado en scalps → cierra todos los scalps
        self._scalp_realized_today: float = 0.0
        self._scalp_daily_hit: bool = False
        self._scalp_pnl_date: str = ""
        self._scalp_peak_today: float = 0.0  # max alcanzado hoy — si cae a $60 cierra
        # High-water mark: balance máximo histórico — recuperar si cae > $500 del pico
        # Usar capital real de startup, no INITIAL_CAPITAL=$100K (que nunca tuvimos)
        self._balance_peak: float = self.capital

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
            _realized_val = float(_realized) if (_realized is not None and float(_realized) != 0.0) else None

            # Fallback: if MT5 history returns 0 on restart, use balance-based estimate.
            # Saves today's start balance on first startup; subsequent restarts use the diff.
            _start_bal_file = os.path.join("memory", "daily_start_balance.json")
            try:
                _start_data = json.load(open(_start_bal_file)) if os.path.exists(_start_bal_file) else {}
            except Exception:
                _start_data = {}
            if _start_data.get("date") != _today_str:
                # First startup of the day — save starting balance
                _start_data = {"date": _today_str, "balance": bal}
                try:
                    json.dump(_start_data, open(_start_bal_file, "w"))
                except Exception:
                    pass
            if _realized_val is None:
                # MT5 history returned 0 — read real balance directly to compute diff.
                # `bal` above may be a connector fallback; read account_info raw here.
                try:
                    import MetaTrader5 as _mt5raw
                    _raw_info = _mt5raw.account_info()
                    _real_bal = float(_raw_info.balance) if _raw_info else None
                except Exception:
                    _real_bal = None
                _start_bal = _start_data.get("balance", 0.0)
                if _real_bal and _real_bal > 0 and _start_bal > 0:
                    _realized_val = max(0.0, _real_bal - _start_bal)
                    if _realized_val > 0:
                        print(f"[STARTUP] PnL via balance-diff: ${_realized_val:.2f} (bal=${_real_bal:.0f} start=${_start_bal:.0f})", flush=True)
                if not _realized_val:
                    _realized_val = 0.0

            self._daily_pnl_date     = _today_str
            self._daily_realized_pnl = _realized_val if _realized_val is not None else 0.0
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



        # Solo OBs dentro del 1% — OBs lejanos son irrelevantes para scalps M15
        current_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        _max_poi_dist  = current_close * 0.01 if current_close > 0 else float("inf")
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

        # Premium/Discount Zone Filter — solo comprar en descuento, vender en premium
        # Bloquea entradas long cuando precio ya subio (zona premium) y viceversa
        _pd_ok = True
        if len(df) >= 50 and current_close > 0:
            _range_high = float(df["high"].rolling(50).max().iloc[-1])
            _range_low  = float(df["low"].rolling(50).min().iloc[-1])
            _range_mid  = (_range_high + _range_low) / 2.0
            if is_bullish and current_close > _range_mid:
                _pd_ok = False  # No comprar en zona premium
            if is_bearish and current_close < _range_mid:
                _pd_ok = False  # No vender en zona descuento

        # Liquidity Sweep Detection — precio barro equal highs/lows y revirtio
        # Confirmacion institucional: el sweep precede al movimiento real
        _has_sweep = False
        if len(df) >= 20:
            _rh = df["high"].rolling(20).max()
            _rl = df["low"].rolling(20).min()
            _swept_high = (float(df["high"].iloc[-1]) > float(_rh.iloc[-2])) and \
                          (float(df["close"].iloc[-1]) < float(_rh.iloc[-2]))
            _swept_low  = (float(df["low"].iloc[-1]) < float(_rl.iloc[-2])) and \
                          (float(df["close"].iloc[-1]) > float(_rl.iloc[-2]))
            _has_sweep = (is_bearish and _swept_high) or (is_bullish and _swept_low)

        # Displacement Candle en BOS reciente — filtra BOS falsos causados por velas pequenas
        # Solo cuenta un BOS como valido si fue roto por una vela de rango expandido (>1.5x ATR)
        _has_displacement_bos = any(b.get("is_displacement", False) for b in bos_list)

        # OTE Zone (Optimal Trade Entry) — Fibonacci 62-79% del swing que creo el OB
        # ICT Unicorn: solo entrar cuando precio retrocede al 62-79% del impulso bullish/bearish
        # Sin OTE = entrada prematura (precio no llego al punto institucional)
        _in_ote = False
        if poi_zones and len(df) >= 5:
            _poi = poi_zones[0]
            _ob_idx = _poi.get("index", 0)
            if _ob_idx > 0 and _ob_idx + 1 < len(df):
                # El impulso que creo el OB: desde el OB hasta el maximo del swing siguiente
                if is_bullish:
                    _swing_low  = float(df["low"].iloc[_ob_idx])
                    _swing_high = float(df["high"].iloc[_ob_idx:_ob_idx+10].max()) if _ob_idx+10 <= len(df) else float(df["high"].iloc[_ob_idx:].max())
                    _ote_low  = _swing_high - (_swing_high - _swing_low) * 0.79
                    _ote_high = _swing_high - (_swing_high - _swing_low) * 0.62
                    _in_ote   = _ote_low <= current_close <= _ote_high
                else:
                    _swing_high = float(df["high"].iloc[_ob_idx])
                    _swing_low  = float(df["low"].iloc[_ob_idx:_ob_idx+10].min()) if _ob_idx+10 <= len(df) else float(df["low"].iloc[_ob_idx:].min())
                    _ote_low  = _swing_low + (_swing_high - _swing_low) * 0.62
                    _ote_high = _swing_low + (_swing_high - _swing_low) * 0.79
                    _in_ote   = _ote_low <= current_close <= _ote_high

        # BUG-SETUP-QUALITY-GATES-UNENFORCED (2026-07-21, trading-strategy
        # expert panel): _has_displacement_bos and _in_ote were both computed
        # with comments explicitly stating they're the ICT/SMC quality
        # precondition for BOS and OB entries respectively ("filtra BOS
        # falsos causados por velas pequenas", "ICT Unicorn: solo entrar
        # cuando precio retrocede al 62-79% del impulso") -- but has_setup
        # never actually required them, so a weak non-displacement BOS or an
        # OB entry outside the OTE zone qualified identically to a genuine
        # one. Now each factor requires its own documented quality gate: OB
        # only counts if price is in its OTE retracement zone, BOS only
        # counts if it broke on a displacement candle. FVG has no documented
        # quality gate in this codebase, so it's unchanged. This tightens
        # qualification to match what the code already claimed to enforce --
        # it will reduce trade frequency on weak/premature setups, which is
        # the intended effect, not a side effect.
        has_setup = (
            (is_bullish or is_bearish)
            and (has_fvg or (has_ob and _in_ote) or (has_bos and _has_displacement_bos))
            and _pd_ok
        )



        direction_word = "bullish" if is_bullish else ("bearish" if is_bearish else "neutral")

        analysis_text = f"{direction_word} trend {struct.structure_type.value}"

        if has_bos:    analysis_text += " BOS confirmado"

        if choch_list: analysis_text += " CHoCH detectado"

        if has_ob:     analysis_text += " order block presente"

        if has_fvg:    analysis_text += " FVG presente"

        if _has_sweep:             analysis_text += " liquidity_sweep_confirmado"
        if _has_displacement_bos:  analysis_text += " displacement_BOS_confirmado"
        if _in_ote:                analysis_text += " OTE_zone_activa"
        if not _pd_ok:             analysis_text += " zona_premium_descuento_bloqueado"
        if has_setup:              analysis_text += " setup valido"



        return {

            "bias": struct.bias,

            "has_ob": has_ob,

            "has_fvg": has_fvg,

            "has_bos": has_bos,

            "has_choch": bool(choch_list),

            "has_setup":           has_setup,

            "has_sweep":           _has_sweep,

            "has_displacement_bos": _has_displacement_bos,

            "in_ote":              _in_ote,

            "pd_ok":               _pd_ok,

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

        _agent_names = ["lunar","elliott","chaos","edge","footprint","instflow","micro","fed","onchain","geo","retail","alt","energy","momentum","billwilliams"]
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

        def _lunar():   return 0  # ELIMINADO: sin evidencia estadistica de edge real
        def _elliott(): return 0  # ELIMINADO 2026-07-06: validado contra 584 trades reales
            # (scripts/validate_elliott_agent.py) -- con ventanas de 200 velas H1 (el
            # tamaño real que usa el bot en vivo), _find_swings() casi siempre cuenta
            # >=5 swings, asi que el bonus SIEMPRE es +10 sin excepcion en los 584
            # trades -- cero poder discriminante, suma lo mismo a ganadores y
            # perdedores por igual. Mismo criterio que elimino Lunar/Chaos/Energy.
        def _chaos():   return 0  # ELIMINADO: sin evidencia estadistica de edge real
        def _edge():
            # Fix 2026-07-06: calculate_full_edge() was called with only symbol+prices,
            # so trades=None -> Kelly/Sharpe/Monte Carlo/walk-forward all ran on a fake
            # [0.0]*20 series instead of this symbol's real closed-trade history.
            # Pull the last 50 real closed trades for this symbol to feed those modules.
            _real_trades = []
            try:
                _rows = self._episodic_conn.execute(
                    "SELECT pnl FROM episodes WHERE symbol=? AND result IN ('WIN','LOSS') "
                    "ORDER BY id DESC LIMIT 50",
                    (signal.symbol,),
                ).fetchall()
                _real_trades = [{"pnl": r[0]} for r in _rows if r[0] is not None]
            except Exception:
                pass
            edge = self._edge.calculate_full_edge(
                symbol=signal.symbol, prices=prices, trades=_real_trades
            )
            return self._edge.get_decision_pts(edge)
        def _footprint():
            if signal.symbol not in SCAN_SYMBOLS:
                return 0
            fp_candle = self._footprint.build_live_footprint(
                signal.symbol, candle_open=signal.entry or 0
            )
            if fp_candle is None:
                return 0
            direction = "long" if signal.signal_type == SignalType.LONG else "short"
            return self._footprint.score_for_trade(fp_candle, direction, signal.entry or 0)
        def _instflow(): return self._inst_flow.score_adjustment(signal.symbol, bias)
        def _micro():    return self._microstructure.score_adjustment(signal.symbol, signal.entry or 0.0)
        def _fed():      return self._fed.score_adjustment(signal.symbol, bias)
        def _onchain():  return self._onchain.score_adjustment(signal.symbol, bias, signal.entry or 0.0)
        def _geo():      return self._geopolitical.score_adjustment(signal.symbol, bias)
        def _retail():   return self._retail_psych.score_adjustment(signal.symbol, df, bias)
        def _alt():      return self._alt_data.score_adjustment(signal.symbol, bias)
        def _energy():  return 0  # ELIMINADO: numerologia/tarot sin evidencia de edge
        def _momentum():  return 0  # DESACTIVADO 2026-07-09: backtest A/B contra 2
            # años de datos reales (scripts/backtest_multiyear.py) mostro que estas
            # penalizaciones (RSI/Bollinger/Estocastico/volumen) cortan el volumen de
            # trades a la mitad (833->1767 al desactivarlas) sin mejorar la calidad lo
            # suficiente para compensar -- P(pasar Axi 5%) bajo de 36.4% a 43.2% al
            # quitarlas. Codigo y tests de smc/momentum.py se mantienen, solo se
            # desconecta su contribucion al score en vivo.
        def _billwilliams(): return 0  # DESACTIVADO 2026-07-09: mismo hallazgo que
            # _momentum -- ver backtest A/B arriba. Codigo y tests de
            # smc/bill_williams.py se mantienen, solo se desconecta la contribucion.

        raw_tasks = [_lunar, _elliott, _chaos, _edge, _footprint, _instflow, _micro, _fed, _onchain, _geo, _retail, _alt, _energy, _momentum, _billwilliams]
        tasks = [_make(name, fn) for name, fn in zip(_agent_names, raw_tasks)]

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(fn) for fn in tasks]
            results = [f.result() for f in as_completed(futures)]

        # Mayoría de agentes debe estar de acuerdo — si >6 dan negativo, señal mala
        positive_agents = sum(1 for r in results if isinstance(r, (int, float)) and r > 0)
        negative_agents = sum(1 for r in results if isinstance(r, (int, float)) and r < 0)
        if negative_agents > positive_agents:
            # Mayoría en contra — reducir bonus significativamente
            bonus = sum(results) * 0.3
        else:
            bonus = sum(results)

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

        # ── Top trader rules bonus (Druckenmiller / PTJ / Soros / ICT) ─────
        trader_bonus = 0
        import datetime as _dt
        _hour_utc = _dt.datetime.utcnow().hour
        _bias = signal.signal_type
        # Soros: momentum fuerte H4 (3+ velas consecutivas mismo lado) = +8
        if not df.empty and len(df) >= 4:
            last3 = df["close"].iloc[-3:].values
            if _bias == SignalType.LONG and all(last3[i] < last3[i+1] for i in range(2)):
                trader_bonus += 8
            elif _bias == SignalType.SHORT and all(last3[i] > last3[i+1] for i in range(2)):
                trader_bonus += 8
        # ICT: NY kill zone 12-15 UTC = +5
        if 12 <= _hour_utc <= 15:
            trader_bonus += 5
        # London kill zone 07-10 UTC = +3
        elif 7 <= _hour_utc <= 10:
            trader_bonus += 3
        # ICT Confluence bonuses — investigacion bots rentables (Unicorn, DRL 70%+ WR)
        _smc_cache = self._df_cache.get(signal.symbol)
        if _smc_cache is not None:
            try:
                _smc_lite = self._run_smc_lite(_smc_cache)
                # OTE Zone (Fibonacci 62-79%) — ICT Unicorn: +20pts
                if _smc_lite.get("in_ote"):
                    trader_bonus += 20
                    print(f"[ICT-OTE] {signal.symbol}: precio en zona 62-79% Fib +20pts", flush=True)
                # Displacement BOS — BOS roto por vela institucional: +15pts
                if _smc_lite.get("has_displacement_bos"):
                    trader_bonus += 15
                    print(f"[ICT-DISP] {signal.symbol}: BOS con displacement candle +15pts", flush=True)
                # Liquidity Sweep confirmado: +10pts (ya calcula en sweep gate)
                if _smc_lite.get("has_sweep"):
                    trader_bonus += 10
                    print(f"[ICT-SWEEP] {signal.symbol}: liquidity sweep confirmado +10pts", flush=True)
            except Exception:
                pass
        if trader_bonus > 0:
            print(f"[TRADER-RULES] {signal.symbol}: +{trader_bonus}pts (momentum+killzone+ICT)", flush=True)
        final_bonus = bonus_clamped + trader_bonus

        print(
            f"[ENRICH] {signal.symbol} base={base} bonus={final_bonus:+d} "
            f"final={base + final_bonus} | {signal.signal_type.value.upper()}",
            flush=True,
        )
        return final_bonus



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

        # Hard cooldown check from disk — survives restarts
        import time as _tsc
        _cd_file = os.path.join("memory", "sl_cooldown_state.json")
        try:
            _cd_data = json.load(open(_cd_file)) if os.path.exists(_cd_file) else {}
            for _cd_key, _cd_ts in _cd_data.items():
                if _cd_key.startswith(symbol + "_"):
                    _elapsed = _tsc.time() - _cd_ts
                    if _elapsed < 14400:  # 4 hours
                        _left = int((14400 - _elapsed) / 60)
                        print(f"[COOLDOWN] {symbol}: bloqueado {_left}min (cooldown disco)", flush=True)
                        return None
        except Exception:
            pass

        loop = asyncio.get_event_loop()

        df = await loop.run_in_executor(None, lambda: self.mt5.get_ohlcv(symbol, timeframe, 200))

        if df is None or df.empty or len(df) < 50:

            return None

        self._df_cache[symbol] = df  # available for _claude_confirm_trade
        smc = self._run_smc_lite(df)

        # H4 structural direction cache — actualizado desde análisis SMC ANTES del momentum filter
        # Esto evita que el momentum filter (pullback temporal) setee H4=WAIT cuando la estructura es LONG/SHORT
        if timeframe == "H4":
            _smc_struct = smc.get("bias", "neutral")
            if _smc_struct == "bullish":
                self._mt5_h4_direction[symbol] = "LONG"
            elif _smc_struct == "bearish":
                self._mt5_h4_direction[symbol] = "SHORT"
            # Si neutral: no actualizar (preservar dirección previa conocida)

        current_price = float(df["close"].iloc[-1])

        # Momentum filter eliminado — en SMC un pullback ES el punto de entrada
        # El filtro bloqueaba setups bullish en retracements (exactamente cuando hay que comprar)

        # Hard trend filter: SMA50 vs SMA200 on current timeframe data
        # Only trade WITH the dominant trend — never against it
        if len(df) >= 200:
            _sma50  = float(df["close"].rolling(50).mean().iloc[-1])
            _sma200 = float(df["close"].rolling(200).mean().iloc[-1])
            _real_trend = "UP" if _sma50 > _sma200 else "DOWN"
            _smc_bias = smc.get("bias", "neutral")
            if _real_trend == "UP" and _smc_bias == "bearish":
                print(f"[TREND-BLOCK] {symbol} {timeframe}: SMC bearish pero SMA50={_sma50:.5f}>SMA200={_sma200:.5f} — tendencia REAL es UP, bloqueando SELL", flush=True)
                from agents.signal_agent import SignalType as _ST2
                return type('S', (), {'signal_type': _ST2.WAIT, 'decision_score': 0})()
            if _real_trend == "DOWN" and _smc_bias == "bullish":
                print(f"[TREND-BLOCK] {symbol} {timeframe}: SMC bullish pero SMA50={_sma50:.5f}<SMA200={_sma200:.5f} — tendencia REAL es DOWN, bloqueando BUY", flush=True)
                from agents.signal_agent import SignalType as _ST2
                return type('S', (), {'signal_type': _ST2.WAIT, 'decision_score': 0})()

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

    async def _send_mt5_real_order(self, signal: TradeSignal):

        """Send a real order to MT5 demo -- strict quality filters applied."""

        order_type = "BUY" if signal.signal_type == SignalType.LONG else "SELL"
        _is_scalp  = (signal.timeframe == "M15")   # definido aqui para todo el metodo

        sl_val = signal.stop_loss if signal.stop_loss else 0.0

        tp_val = signal.take_profit if signal.take_profit else 0.0

        # ── 8D DIM 8: Portfolio correlation guard ──────────────────────
        # Blocks duplicate USD exposure (e.g. EURUSD + GBPUSD + AUDUSD all BUY)
        _8d_result = None  # BUG-LEARN-UNKNOWN fix: usado luego para poblar regime del episodio
        try:
            if not hasattr(self, "_eight_dim_agent"):
                from agents.eight_dim_agent import EightDimensionAgent
                self._eight_dim_agent = EightDimensionAgent()
            loop8 = __import__("asyncio").get_event_loop()
            _open8 = await loop8.run_in_executor(None, self.mt5.get_positions)
            _8d_result = self._eight_dim_agent.analyze(
                signal.symbol,
                self._df_cache.get(signal.symbol),
                _open8 or [],
                direction=order_type
            )
            if not _8d_result.allowed:
                print(f"[8D-BLOCK] {signal.symbol}: {_8d_result.reason}", flush=True)
                return
            _8d_regime = f"vol={_8d_result.vol_regime} trend={_8d_result.trend_regime}"
            print(f"[8D] {signal.symbol}: score_mult={_8d_result.score_mult:.2f} {_8d_regime}", flush=True)
        except Exception as _8d_exc:
            print(f"[8D] error (no bloqueo): {_8d_exc}", flush=True)

        # ── PAUSA MANUAL (Telegram /pause) ────────────────────────────
        if getattr(self.commander, "state", None) and getattr(self.commander.state, "paused", False):
            print(f"[PAUSE] {signal.symbol}: bot pausado manualmente via Telegram — skip", flush=True)
            return

        # ── NEWS BLACKOUT (FOMC/NFP) — regla documentada en CLAUDE.md
        # ("no_operar_noticias") que nunca se habia conectado al pipeline real.
        try:
            _news_risk = self.event_driven.get_risk_adjustment()
            if _news_risk <= 0.25 and not _is_scalp:
                print(f"[NEWS-BLACKOUT] {signal.symbol}: ventana FOMC/NFP activa — skip swing", flush=True)
                return
        except Exception as _news_exc:
            print(f"[NEWS-BLACKOUT] error (no bloqueo): {_news_exc}", flush=True)

        # ── NEWS BLACKOUT (calendario real, no solo FOMC/NFP US) — Forex
        # Factory feed gratis sin API key, cubre BCE/BOE/BOC/RBA/RBNZ/SNB/CPI
        # etc. Complementa el bloque anterior (que solo sabia de FOMC/NFP
        # hardcodeados) con eventos High-impact reales de CUALQUIER divisa
        # del par. Fallback silencioso si el feed no responde (cache stale).
        try:
            _pair_currencies = currencies_for_symbol(signal.symbol)
            if _pair_currencies and not _is_scalp:
                _hi_event = get_high_impact_window(_pair_currencies, window_minutes=30)
                if _hi_event:
                    print(
                        f"[NEWS-BLACKOUT] {signal.symbol}: {_hi_event['country']} "
                        f"'{_hi_event['title']}' (High impact) a {_hi_event['time']:%H:%M UTC} — skip swing",
                        flush=True,
                    )
                    return
        except Exception as _cal_exc:
            print(f"[NEWS-BLACKOUT] calendario real error (no bloqueo): {_cal_exc}", flush=True)

        # ── AXI SELECT GUARDS ──────────────────────────────────────────
        # Guard 1: emergency daily loss limit (-4%)
        if self._axi_guard.paused_today:
            print(f"[AXI-GUARD] {signal.symbol}: bot pausado — limite diario alcanzado hoy", flush=True)
            return
        # Guard 2: consistency rule — ningún día > 30% del profit mensual
        try:
            _ce_result = self._axi_enforcer.check(
                today_pnl    = self._daily_realized_pnl,
                monthly_pnl  = self._axi_tracker.get_status().monthly_pnl,
            )
            if _ce_result.should_block_new and not _is_scalp:
                print(f"[AXI-CONSISTENCY] {signal.symbol}: {_ce_result.reason}", flush=True)
                return
        except Exception as _ce_exc:
            print(f"[AXI-CONSISTENCY] error (no bloqueo): {_ce_exc}", flush=True)

        # Índices (NAS100/US30) NO scalp M15 — SL de 4 pips inválido (min 50pts)
        # NAS100 solo opera swings H4 con SL basado en ATR
        _is_index = any(x in signal.symbol for x in ("NAS100", "US30", "SPX", "DAX"))
        if _is_scalp and _is_index:
            print(f"[SKIP] {signal.symbol}: índice no permite scalp M15 (SL inválido)", flush=True)
            return

        # Cooldown tras SL: mismo símbolo+dirección no abre por 4 horas
        import time as _time_mod
        _sl_key = f"{signal.symbol}_{order_type}"
        _sl_ts  = self._symbol_sl_time.get(_sl_key, 0.0)
        _cooldown_h = 2.0
        if _time_mod.time() - _sl_ts < _cooldown_h * 3600:
            _mins_left = int((_cooldown_h * 3600 - (_time_mod.time() - _sl_ts)) / 60)
            print(f"[COOLDOWN] {signal.symbol} {order_type}: SL reciente — espera {_mins_left}min", flush=True)
            return

        # Meta swing (DAILY_PROFIT_TARGET) cumplida → SOLO scalps (M15) el resto del día
        # Los swings ya aseguraron el mínimo — no abrir más swings que se coman la ganancia
        if self._daily_target_hit and not _is_scalp:
            print(f"[MT5] {signal.symbol}: meta swing ${DAILY_PROFIT_TARGET:.0f} cumplida — solo scalps permitidos, skip swing", flush=True)
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



        # ── FILTER 2b: Spread máximo — límite por tipo de instrumento ───────
        # BUG #11: NAS100 al abrir tiene spread normal de 20-50pts — no usar límite forex
        # Forex: max 3 pips | Índices (NAS100/US30): max 80 pips
        try:
            import MetaTrader5 as _mt5sp
            _tick_sp = _mt5sp.symbol_info_tick(signal.symbol)
            if _tick_sp:
                _spread = abs(_tick_sp.ask - _tick_sp.bid)
                _sym_sp = _mt5sp.symbol_info(signal.symbol)
                if _sym_sp and _sym_sp.point > 0:
                    _spread_pips = _spread / (_sym_sp.point * 10)
                    _is_index = any(x in signal.symbol for x in ("NAS", "US30", "SPX", "DAX", "UK100"))
                    _max_spread = 80.0 if _is_index else 3.0
                    if _spread_pips > _max_spread:
                        print(f"[SPREAD] {signal.symbol}: spread={_spread_pips:.1f} > {_max_spread} max — skip", flush=True)
                        return
        except Exception:
            pass

        # ── FILTER 3: Horario muerto ──────────────────────────────────────

        now_utc = datetime.now(timezone.utc)

        if now_utc.hour in DEAD_HOURS_UTC:

            print(f"[MT5] {signal.symbol}: hora muerta {now_utc.hour}:00 UTC, skip", flush=True)

            return

        if now_utc.weekday() == 4 and now_utc.hour >= 16:  # viernes 16:00+ UTC — no abrir nuevos trades
            print(f"[MT5] {signal.symbol}: viernes 16:00+ UTC, no se abren nuevos trades, skip", flush=True)
            return



        # FILTER 3b eliminado — el triple confirm D1+H4+H1 del scan loop
        # ya valida la dirección H4. Redundante y causaba bloqueos falsos.

        # ── SCALP M15: volumen fijo 0.1L, SL=4pips($4), TP=12pips($12) ─────────
        # 0.1L: pip=$1 → SL=4pips=$4 max loss, TP=12pips=$12, cerrar en $10
        if _is_scalp:
            # Índices NO permiten scalp M15 — SL de 4 pips inválido (necesita 50+ pts)
            if any(x in signal.symbol for x in ("NAS100", "US30", "SPX", "DAX", "UK100")):
                print(f"[SKIP-IDX] {signal.symbol}: índice no permite scalp M15", flush=True)
                return
            try:
                import MetaTrader5 as _mt5s
                _tick_s = _mt5s.symbol_info_tick(signal.symbol)
                _scalp_price = (_tick_s.ask if order_type == "BUY" else _tick_s.bid) if _tick_s else 0.0
                _sym_s = _mt5s.symbol_info(signal.symbol)
                if _scalp_price > 0 and _sym_s:
                    _pip = _sym_s.point * 10  # 1 pip = 10 points (5-digit broker)
                    _sl_pips  = 8   # 8 pips SL = $8 max loss a 0.1L (era 4, muy ajustado)
                    _tp_pips  = 24  # 24 pips TP = $24, RR=3:1 (era 12)
                    if order_type == "BUY":
                        sl_val = round(_scalp_price - _sl_pips * _pip, 5)
                        tp_val = round(_scalp_price + _tp_pips * _pip, 5)
                    else:
                        sl_val = round(_scalp_price + _sl_pips * _pip, 5)
                        tp_val = round(_scalp_price - _tp_pips * _pip, 5)
                    print(f"[SCALP] {signal.symbol} {order_type} @{_scalp_price:.5f} SL={sl_val:.5f} TP={tp_val:.5f} (8pip/$8 SL, 24pip/$24 TP)", flush=True)
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
                _min_rr_req = 1.5 if _is_scalp else MIN_RR  # swings: usa MIN_RR real (3.0, subido 2026-06-30) — antes hardcodeado a 1.9, ignoraba el fix
                # BUG #12: Si RR bajo → ajustar TP para garantizar mínimo (no skip)
                # NAS100 al abrir: ATR enorme → SL wide → TP del signal queda cerca → RR=0.33
                if rr < _min_rr_req and sl_dist > 0 and not _is_scalp:
                    # Usar _min_rr_req + 0.1 para que el TP calculado supere el threshold
                    # aunque haya rounding de float (1.90 < 1.9 = False gracias al margen)
                    _tp_rr = _min_rr_req + 0.1
                    tp_val = round((_entry_ref + sl_dist * _tp_rr) if order_type == "BUY"
                                   else (_entry_ref - sl_dist * _tp_rr), 5)
                    tp_dist = abs(_entry_ref - tp_val)
                    rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
                    print(f"[TP-ADJ] {signal.symbol}: TP ajustado → RR={rr:.2f}", flush=True)
                if rr < _min_rr_req:
                    print(f"[MT5] {signal.symbol}: RR={rr:.2f} < {_min_rr_req} minimo, skip", flush=True)
                    return
                print(f"[RR-OK] {signal.symbol}: RR={rr:.2f} ({'SCALP' if _is_scalp else 'SWING'})", flush=True)



        # ── FILTER 5: Max 1 trade real por dia ───────────────────────────

        today_str = now_utc.strftime("%Y-%m-%d")

        trades_today = self._daily_trades.get(today_str, 0)

        # Sin limite de trades diarios — el mercado limita, no el contador



        # ── FILTER 6: Axi Select drawdown y perdida diaria ──────────────

        try:

            daily_pnl = await asyncio.get_running_loop().run_in_executor(

                None, self.mt5.get_daily_pnl

            )

            self._risk_gate_state.daily_pnl_today = daily_pnl
            # Keep daily_realized_pnl in sync with MT5 real closed P&L
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_pnl_date == today_utc and daily_pnl is not None:
                self._daily_realized_pnl = float(daily_pnl)

            acc_info = await asyncio.get_running_loop().run_in_executor(

                None, self.mt5.get_account_info

            )

            self._risk_gate_state.current_balance = acc_info.get("balance", self.capital)

            _equity = acc_info.get("equity", self._risk_gate_state.current_balance)
            can_trade, reason = self._risk_gate_agent.can_trade(self._risk_gate_state, equity=_equity)

            if not can_trade:

                print(f"[RISK-GATE] {signal.symbol}: BLOQUEADO -- {reason}", flush=True)

                try:

                    await self.telegram.send_glint_alert(

                        f"<b>RISK GATE BLOQUEO</b>\n{signal.symbol}: {reason}"

                    )

                except Exception:

                    pass

                return

        except Exception as _fe:

            print(f"[RISK-GATE] check error: {_fe}", flush=True)



        # ── FILTER 7: Max posiciones abiertas ────────────────────────────────
        loop = asyncio.get_running_loop()
        existing = await loop.run_in_executor(None, self.mt5.get_positions)

        # Scalp y swing tienen topes independientes
        # Modo recuperación: permite más scalps simultáneos para recuperar más rápido
        _current_bal_r  = self._risk_gate_state.current_balance or self.capital
        _recovery_mode  = (
            self._daily_realized_pnl <= RECOVERY_TRIGGER_LOSS
        ) and not self._daily_target_hit
        _accel_mode = (
            self._daily_realized_pnl >= ACCEL_TRIGGER_PROFIT and
            _current_bal_r >= self.capital * 0.98 and not _recovery_mode
        )
        _max_scalp_now = (RECOVERY_MAX_SCALPS if _recovery_mode
                          else ACCEL_MAX_SCALPS if _accel_mode
                          else MAX_SCALP_POSITIONS)
        if _is_scalp:
            scalp_open = [p for p in existing if p.get("volume", 1) <= 0.10]
            if len(scalp_open) >= _max_scalp_now:
                print(f"[MT5] {signal.symbol}: {len(scalp_open)} scalps abiertas (max={_max_scalp_now}{'🔄RECOVERY' if _recovery_mode else ''}), skip", flush=True)
                return
            # Max 1 scalp por símbolo — evita apilar 3 USDCAD o 4 GBPUSD
            sym_scalps = [p for p in scalp_open if p.get("symbol") == signal.symbol]
            if len(sym_scalps) >= 1:
                print(f"[MT5] {signal.symbol}: ya tiene scalp abierto — skip duplicado", flush=True)
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
            # MAX 1 POSICION POR SIMBOLO — sin excepciones, sin "segunda a score>=120"
            # El "escalado" causó duplicados USDCAD y EURUSD que destruyeron ganancias
            pnl_live = sym_open[0].get("profit", 0.0)
            print(
                f"[MT5] {signal.symbol}: posicion ya abierta ({pnl_live:+.2f} USD) -- skip duplicado",
                flush=True,
            )
            return



        # ── FILTER 7c: Posiciones abiertas perdiendo ─────────────────────────
        if existing:
            _bal = self._risk_gate_state.current_balance or self.capital
            _total_pnl = sum(p.get("profit", 0.0) for p in existing)
            _loss_limit = _bal * 0.025  # bloquear nuevas entradas si portfolio pierde >2.5% ($2,425 con $97K)
            for p in existing:
                _pnl  = p.get("profit", 0.0)
                _pct  = abs(_pnl) / _bal * 100 if _bal > 0 else 0
                _tag  = f"perdiendo ${abs(_pnl):.2f} ({_pct:.2f}%)" if _pnl < 0 else f"ganando ${_pnl:.2f} ({_pct:.2f}%)"
                print(f"[LIVE-POS] {p.get('symbol','?')} {p.get('type','?')} {_tag}", flush=True)
            if _total_pnl < -_loss_limit:
                print(
                    f"[FILTER-LOSS] {signal.symbol}: skip -- "
                    f"portfolio perdiendo ${abs(_total_pnl):.2f} (limite=${_loss_limit:.0f} = 2.5% balance)",
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
        live_capital = self._risk_gate_state.current_balance if self._risk_gate_state.current_balance > 1000 else self.capital
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
            if volume == 0.0:
                print(f"[SKIP-VOL] {signal.symbol}: VolumeCalculator devolvio 0 (capital={live_capital:.0f}) — skip", flush=True)
                return

        # Swing: riesgo adaptativo según déficit diario
        _shortfall = DAILY_PROFIT_TARGET - self._daily_realized_pnl
        _now_h = __import__('datetime').datetime.utcnow().hour
        if _shortfall > 200 and _now_h >= 13:
            # Detrás de meta por >$200 en horario activo: escalar riesgo
            MAX_DOLLAR_RISK = min(400.0, 200.0 + _shortfall * 0.3)
            print(f"[ADAPTIVE-SIZE] Deficit=${_shortfall:.0f} → MAX_RISK=${MAX_DOLLAR_RISK:.0f}", flush=True)
        elif _shortfall <= 0:
            MAX_DOLLAR_RISK = 100.0  # meta cumplida: proteger ganancias
        else:
            MAX_DOLLAR_RISK = 200.0
        if not _is_scalp and volume > 0 and sl_val > 0 and _entry_for_vol > 0:
            _sl_pips = abs(_entry_for_vol - sl_val)
            _sym_info = None
            try:
                import MetaTrader5 as _mt5r
                _sym_info = _mt5r.symbol_info(signal.symbol)
            except Exception:
                pass
            if _sym_info:
                # Use VolumeCalculator pip tables to avoid 100x error on NAS100/US30
                from core.volume_calculator import VolumeCalculator as _VC
                _base = _VC._norm(signal.symbol)
                _vc_pip_size  = _VC._PIP_SIZE.get(_base, _sym_info.point)
                _vc_pip_value = _VC._PIP_VALUE.get(_base, _sym_info.trade_contract_size * _sym_info.point)
                _sl_in_pips   = _sl_pips / _vc_pip_size
                _dollar_risk  = volume * _sl_in_pips * _vc_pip_value
                if _dollar_risk > MAX_DOLLAR_RISK and _sl_in_pips > 0:
                    _raw_vol = MAX_DOLLAR_RISK / (_sl_in_pips * _vc_pip_value)
                    _step = _sym_info.volume_step if _sym_info.volume_step > 0 else 0.01
                    volume = max(round(int(_raw_vol / _step) * _step, 8), _sym_info.volume_min)
                    print(f"[RISK-CAP] {signal.symbol}: riesgo ${_dollar_risk:.0f} > ${MAX_DOLLAR_RISK} — vol={volume}L", flush=True)
                    _dollar_risk = volume * _sl_in_pips * _vc_pip_value  # recompute after the cap for accurate LOSS-LIMIT sizing below
                _intended_risk_usd = _dollar_risk
        # Proteccion: swing con vol<0.11L seria tratado como scalp por el monitor — skip
        if not _is_scalp and volume < 0.11:
            print(f"[SKIP-MINVOL] {signal.symbol}: vol={volume:.2f}L < 0.11 minimo swing — skip (evita que monitor lo cierre como scalp)", flush=True)
            return

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

            # BUG-LOSS-LIMIT-FLAT (2026-07-21, risk-management expert panel):
            # the software backstop that force-closes a losing position
            # (LOSS-LIMIT) used one flat 0.8%-of-balance threshold for every
            # trade regardless of what risk_pct (0.25/0.5/1%) it was actually
            # sized for -- a 0.25%-risk trade (~$242) was only backstopped at
            # 0.8% (~$776, >3x intended) if the broker-side SL failed to hold
            # (documented as real in metatrader_connector.py). Store the real
            # intended risk per ticket so _manage_open_positions can scale
            # LOSS-LIMIT off of it instead of a flat balance percentage.
            if not _is_scalp:
                self._position_intended_risk[result["ticket"]] = _intended_risk_usd

            # ── Real cost capture (spread/slippage) — feeds future backtest cost model ──
            _spread_pips_ep = result.get("spread_pips")
            _slippage_pips_ep = None
            try:
                from core.volume_calculator import VolumeCalculator as _VCsl
                _base_sl = _VCsl._norm(signal.symbol)
                _pip_size_sl = _VCsl._PIP_SIZE.get(_base_sl, 0.0001)
                _fill_price_sl = result.get("price")
                if signal.entry and signal.entry > 0 and _fill_price_sl and _pip_size_sl > 0:
                    _slippage_pips_ep = round(abs(signal.entry - _fill_price_sl) / _pip_size_sl, 2)
            except Exception as _sl_exc:
                print(f"[SLIPPAGE] calc error (no bloqueo): {_sl_exc}", flush=True)

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

                    "setup_type": signal.trigger or "SMC",

                    "regime": _8d_result.trend_regime if _8d_result else "unknown",

                    "session": "+".join(__import__("core.session_manager", fromlist=["get_active_sessions"]).get_active_sessions()) or "unknown",

                    "slippage_pips": _slippage_pips_ep,

                    "spread_pips": _spread_pips_ep,

                }, conn=self._episodic_conn)

                self._open_episodes[result["ticket"]] = eid
                self._save_open_episodes()

            except Exception as _ep_err:

                print(f"[EPISODIC] record error: {_ep_err}", flush=True)

            # Solo contar SWINGS en el limite diario — scalps no agotan el contador
            if not _is_scalp:
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

        """Every 3s: log open positions + detect closures → update learning + risk-gate.
        CRITICO: 3s para detectar movimientos rapidos y proteger profits con BE/TRAIL.
        Era 60s — demasiado lento para mercados que mueven $18→$1 en segundos."""

        _known_tickets: set = set()
        _ticket_info: dict = {}  # ticket → (symbol, direction) para cooldown al cerrar
        # Tickets flagged for close-on-market-open (positions with no SL)
        # Populated dynamically: any position with SL=0 gets auto-closed on next open
        _close_when_open: set = set()
        _log_counter: int = 0  # solo imprime posiciones cada 10 ciclos (30s)
        # BUG-LEARN-NO-RESULT (2026-07-03): MT5 tarda en registrar el deal de cierre en
        # history_deals_get() -- con polling de 100ms, get_closing_deal() a veces no lo
        # encuentra en el primer intento. Antes: se descartaba el resultado para siempre
        # (result/pnl quedaban NULL, el AutonomousLearner nunca aprendia de trades reales).
        # Fix: reintentar por varios ciclos antes de darse por vencido.
        _pending_deal_lookup: dict = {}  # ticket -> {"episode_id": int|None, "attempts": int}
        _MAX_DEAL_LOOKUP_ATTEMPTS = 30  # ~3-6s de reintentos a 100-200ms/ciclo

        while self._running:

            await asyncio.sleep(0.1)  # 100ms — prácticamente tiempo real
            _log_counter += 1

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
                    # Track ticket → (symbol, direction) for cooldown on close
                    _ticket_info[p["ticket"]] = (p.get("symbol", ""), p.get("type", "BUY"))
                # Remove tickets that are no longer open (already closed externally)
                _close_when_open -= (set(_close_when_open) - current_tickets)

                # ── Auto-close flagged positions (closes when market accepts) ─
                for p in positions:
                    ticket = p.get("ticket", 0)
                    if ticket in _close_when_open:
                        pnl    = p.get("profit", 0.0)
                        symbol = p.get("symbol", "?")
                        ok = await self._close_guarded(
                            loop, ticket, "IDX-NO-SL",
                            f"<b>CIERRE AUTOMATICO</b>\n{symbol} #{ticket} sin SL -- cerrado\nP&L: ${pnl:+.2f} USD"
                        )
                        if ok:
                            _close_when_open.discard(ticket)
                            print(f"[AUTO-CLOSE] {symbol} #{ticket} cerrado (sin SL) | P&L: ${pnl:+.2f}", flush=True)
                        else:
                            print(f"[AUTO-CLOSE] {symbol} #{ticket} sin SL -- mercado cerrado, reintentando", flush=True)

                current_tickets = {p["ticket"] for p in positions}

                # ── Partial close 50% at 1:1 RR — DISABLED, kept only as a note ──
                # SIMPLIFY-2026-07-21: this used to be a `for p in positions: try:
                # ... continue [unconditional] ... [58 more lines]` block -- every
                # line after the continue was unreachable dead code (found while
                # simplifying, no test could have caught it since it never ran).
                # Audit 2026-07-06 on the full real trade book found partial-close
                # caps wins at ~0.5R while a full SL loses 1R (avg WIN $26.65 vs
                # avg LOSS $19.62, ratio 1.36:1 vs the RR=3.0 designed) because the
                # 50% remainder almost always drifts back to breakeven before
                # reaching the real TP. Left disabled — full TP with breakeven-via-
                # trailing below (2.0R trigger) instead of a premature partial.

                # ── Detect closed positions ───────────────────────────────

                closed = _known_tickets - current_tickets

                for ticket in closed:
                    self._position_intended_risk.pop(ticket, None)
                    # Cooldown 2h en cualquier cierre (SL, TP, o manual)
                    # Evita que el bot re-abra inmediatamente el mismo par
                    _closed_sym, _closed_dir = _ticket_info.pop(ticket, ("", "BUY"))
                    if _closed_sym:
                        import time as _tc
                        _cd_key = f"{_closed_sym}_{_closed_dir}"
                        self._symbol_sl_time[_cd_key] = _tc.time()
                        try:
                            json.dump(self._symbol_sl_time,
                                      open(os.path.join("memory", "sl_cooldown_state.json"), "w"))
                        except Exception:
                            pass
                        print(f"[COOLDOWN-SET] {_closed_sym} {_closed_dir}: 2h cooldown (posicion cerrada)", flush=True)

                    # BUG-LEARN-NO-RESULT fix: NO sacar de self._open_episodes todavia --
                    # si el bot se reinicia mientras el deal aun no aparece en MT5 history,
                    # _recover_orphaned_episodes() (corrido en cada arranque, ver run())
                    # necesita encontrar el ticket ahi para poder recuperarlo. Solo se saca
                    # de self._open_episodes cuando el deal se confirma (o se agotan los
                    # reintentos), mas abajo.
                    episode_id = self._open_episodes.get(ticket)
                    _pending_deal_lookup[ticket] = {"episode_id": episode_id, "attempts": 0}

                for _pend_ticket in list(_pending_deal_lookup.keys()):
                    _pend = _pending_deal_lookup[_pend_ticket]
                    ticket = _pend_ticket
                    episode_id = _pend["episode_id"]

                    deal = await loop.run_in_executor(
                        None, lambda t=ticket: self.mt5.get_closing_deal(t)
                    )

                    if not deal:
                        _pend["attempts"] += 1
                        if _pend["attempts"] >= _MAX_DEAL_LOOKUP_ATTEMPTS:
                            print(
                                f"[LEARN] WARNING #{ticket}: no se encontro deal de cierre "
                                f"tras {_pend['attempts']} intentos -- queda pendiente para "
                                f"_recover_orphaned_episodes() en el proximo restart",
                                flush=True,
                            )
                            del _pending_deal_lookup[ticket]
                        continue

                    del _pending_deal_lookup[ticket]
                    self._open_episodes.pop(ticket, None)
                    self._save_open_episodes()

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

                    # Update risk-gate state

                    try:

                        self._risk_gate_agent.record_trade(self._risk_gate_state, pnl)

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

                        f" | Balance: ${self._risk_gate_state.current_balance:,.2f}",

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



                # ── Log open positions (cada 10 ciclos = 30s para no spam) ──

                if positions and _log_counter % 300 == 0:  # log cada 30s (300 × 0.1s)

                    total_pnl = sum(p.get("profit", 0.0) for p in positions)
                    _bal_ref  = self._risk_gate_state.current_balance or self.capital
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

                    # ── META DIARIA: realizado + float >= $250 → solo cerrar SCALPS
                    # Los SWINGS continuan hasta su TP — son los que dan $247+ por trade
                    _neto_dia = self._daily_realized_pnl + total_pnl
                    if not self._daily_target_hit and _neto_dia >= DAILY_PROFIT_TARGET:
                        self._daily_target_hit = True
                        print(f"[META-DIA] Neto ${_neto_dia:.2f} >= ${DAILY_PROFIT_TARGET:.0f} — cerrando solo scalps, swings siguen al TP", flush=True)
                        for _p in list(positions):
                            # Solo cerrar scalps (vol <= 0.1L) — swings siguen corriendo
                            if _p.get("volume", 0) <= 0.10:
                                try:
                                    _ok = await self._close_guarded(loop, _p["ticket"], "META-DIA-SCALP")
                                    print(f"[META-DIA] Scalp cerrado {_p['symbol']} #{_p['ticket']} PnL=${_p.get('profit',0):.2f}", flush=True)
                                except Exception:
                                    pass
                        try:
                            await self.telegram.send_glint_alert(
                                f"<b>META DIARIA $250 ALCANZADA</b>\nNeto: ${_neto_dia:.2f}\nSwings activos hasta TP ✅"
                            )
                        except Exception:
                            pass

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

                elif not positions and _log_counter % 300 == 0:
                    # No open positions right now — clear any stale snapshot so a
                    # future restart's run_recovery() doesn't "recover" trades that
                    # already closed (positions_state.json was only ever written
                    # while positions existed, never cleared on close).
                    try:
                        from core.wakeup_recovery import clear_positions as _clear_pos
                        _clear_pos()
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

    async def _close_guarded(self, loop, ticket: int, reason: str,
                              telegram_html: str = None) -> bool:
        """Shared close path for every position-management guard.
        SIMPLIFY-2026-07-21: replaces 14 near-identical inline blocks
        (close call + peak cleanup + Telegram alert) scattered across
        _manage_open_positions -- see docs/superpowers/plans/
        2026-07-21-supervisor-close-consolidation.md for the audit that
        found them. Behavior is unchanged: same close_position(ticket,
        reason) call, same peak-pop-on-success, same best-effort alert.
        telegram_html=None for the 2 batch-close call sites (META-DIA-SCALP,
        META-SWING) that already send one alert for the whole batch after
        the loop instead of per-position -- consolidating the close+peak-pop
        there without forcing a per-position alert that never existed.
        """
        ok = await loop.run_in_executor(
            None, lambda t=ticket, r=reason: self.mt5.close_position(t, r)
        )
        if ok:
            self._position_peaks.pop(ticket, None)
            if telegram_html:
                try:
                    await self.telegram.send_glint_alert(telegram_html)
                except Exception:
                    pass
        return ok

    def _recover_orphaned_episodes(self) -> None:
        """On startup: backfill outcomes for tickets that closed during a prior restart.

        BUG-ORPHAN-SILENT-DROP (2026-07-20): both branches below used to
        unconditionally add the ticket to `removed` -- even when the closing
        deal wasn't found, and even when update_episode_result raised (the
        exception was swallowed by a bare `except: pass` with no log line).
        Either path stopped tracking the ticket forever, leaving its
        episodes.db row with result=NULL permanently and silently -- no
        warning, no retry on the next restart. Found via 2 real orphaned
        rows (EURAUD #76484092, GBPCAD #76484444, both closed 2026-07-17
        with deals that WERE present in MT5 history when queried manually
        3 days later) that episodes.db never recorded. Fix: only drop a
        ticket from tracking once its result is actually persisted; keep
        retrying indefinitely otherwise, with a visible warning each time
        so a repeat failure is never silent again.
        """
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
                if not d:
                    print(f"[LEARN] orphan ticket={ticket}: sin deal de cierre en 90 dias -- se reintenta en el proximo restart", flush=True)
                    continue
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
                    removed.append(ticket)
                except Exception as _upd_exc:
                    print(f"[LEARN] orphan ticket={ticket}: update_episode_result fallo ({_upd_exc}) -- se reintenta en el proximo restart", flush=True)
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
            # Recalibrado 2026-07-01 tras barrido thr x RR sobre 2 anos reales (EURUSD+USDCAD):
            # thr=80/RR=3.0 -> WR=41.7%, P(pasar 5% mensual)=28.4% (vs 8.5% con 90-95)
            # thr=95 NO demostro mejor calidad que 80 en los datos -- solo menos volumen.
            # Igual se mantiene selectividad dinamica: peor WR reciente -> mas estricto.
            if wr >= 0.65:
                thr = MT5_SCORE_AUTO_REDUCE - 2       # 78 — buen WR reciente, el mas permisivo
            elif wr >= 0.55:
                thr = MT5_SCORE_AUTO_REDUCE           # 80 — punto optimo del backtest
            elif wr >= 0.40:
                thr = MT5_SCORE_AUTO_REDUCE + 5        # 85
            else:
                thr = MT5_SCORE_AUTO_REDUCE + 10       # 90 — WR<40%, maxima selectividad (sin llegar al 95 que no aporta segun el backtest)
            print(f"[ADAPT-THR] WR={wr*100:.0f}% → threshold={thr} (optimo backtest=80)", flush=True)
            return thr
        except Exception as _e:
            return MT5_REAL_SCORE_THRESHOLD  # fallback al default

    async def _manage_open_positions(self):
        """
        Active position management:
        0a. Friday pre-close: close ALL open positions (winners + losers) by 19:30 UTC Friday (before 21:00 close) — avoids weekend gap risk on both sides
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

            bal       = self._risk_gate_state.current_balance or self.capital
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
                self._axi_tracker.record_day(self._daily_realized_pnl, capital=bal)
                # Si el PnL REAL está bajo el target (ej: ganó $277 luego NAS100 -$212 → $65)
                # el flag se resetea para que el bot pueda seguir operando y llegar a $250
                if self._daily_target_hit and self._daily_realized_pnl < DAILY_PROFIT_TARGET:
                    self._daily_target_hit = False
                    print(f"[META-RESET] PnL=${self._daily_realized_pnl:.2f} bajo meta $250 — bot puede operar swings", flush=True)

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

            # META = realizados + flotantes >= $250 — el bot incluye lo ya ganado hoy
            _total_combined = self._daily_realized_pnl + swing_float
            if not self._daily_target_hit and _total_combined >= DAILY_PROFIT_TARGET:
                self._daily_target_hit = True
                print(
                    f"[META-SWING] realizado=${self._daily_realized_pnl:.2f} + float=${swing_float:.2f}"
                    f" = ${_total_combined:.2f} >= ${DAILY_PROFIT_TARGET:.0f}"
                    f" — META CUMPLIDA, cerrando SWINGS",
                    flush=True,
                )
                for sp in list(swing_positions):
                    t_ticket = sp["ticket"]
                    t_sym    = sp.get("symbol", "?")
                    t_pnl    = sp.get("profit", 0.0)
                    ok = await self._close_guarded(loop, t_ticket, "META-SWING")
                    if ok:
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
            _current_bal   = self._risk_gate_state.current_balance or self.capital
            # Actualizar high-water mark (balance máximo histórico)
            if _current_bal > self._balance_peak:
                self._balance_peak = _current_bal
                print(f"[PEAK] Nuevo máximo histórico: ${self._balance_peak:,.2f}", flush=True)

            # Recovery solo por pérdida real del día — _below_peak eliminado porque
            # risk-gate inicia current_balance=$100K causando falso drawdown desde día 1
            # La protección multi-día ya la cubre: RiskGovernor + risk-gate drawdown limits
            _day_in_loss      = self._daily_realized_pnl <= RECOVERY_TRIGGER_LOSS
            _in_recovery      = _day_in_loss and not self._scalp_daily_hit

            # Estrategia 5: Modo Aceleración — dia muy bueno → maximizar
            _in_accel = (
                self._daily_realized_pnl >= ACCEL_TRIGGER_PROFIT and
                _current_bal >= self.capital * 0.98 and
                not _in_recovery and
                not self._scalp_daily_hit
            )

            if _in_recovery:
                SCALP_MIN_PROFIT = RECOVERY_SCALP_TP
                SCALP_MAX_LOSS   = RECOVERY_SCALP_SL
                # Log RECOVERY solo cada 300 ciclos (~30s) para no spam con monitor 100ms
                if getattr(self, "_recovery_log_count", 0) % 300 == 0:
                    if self._balance_peak > INITIAL_CAPITAL and _current_bal < self._balance_peak:
                        _gap = self._balance_peak - _current_bal
                        print(f"[RECOVERY] Cayó ${_gap:.0f} del pico ${self._balance_peak:,.0f} — recuperando", flush=True)
                    elif _current_bal < INITIAL_CAPITAL:
                        print(f"[RECOVERY] Balance ${_current_bal:,.0f} bajo $100K — recuperando capital base", flush=True)
                    else:
                        print(f"[RECOVERY] Dia ${self._daily_realized_pnl:.2f} — recuperando el dia", flush=True)
                self._recovery_log_count = getattr(self, "_recovery_log_count", 0) + 1
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
                    ok = await self._close_guarded(loop, sp["ticket"], "SCALP-DAY")
                    if ok:
                        print(f"[SCALP-DAY] {sp.get('symbol','?')} cerrado ${sp.get('profit',0):+.2f} (meta $60 scalp cumplida)", flush=True)
            else:
                # Gestión individual de cada scalp
                for sp in list(scalp_positions):
                    sp_pnl    = sp.get("profit", 0.0)
                    sp_ticket = sp["ticket"]
                    sp_sym    = sp.get("symbol", "?")
                    if sp_pnl >= SCALP_MIN_PROFIT:
                        ok = await self._close_guarded(loop, sp_ticket, "SCALP-TP")
                        if ok:
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
                        ok = await self._close_guarded(loop, sp_ticket, "SCALP-SL")
                        if ok:
                            self._scalp_realized_today += sp_pnl
                            print(f"[SCALP-SL] {sp_sym} #{sp_ticket} ${sp_pnl:+.2f} | total scalp hoy=${self._scalp_realized_today:.2f}", flush=True)

            # ── 0a. Friday pre-close: dump ALL losers before weekend ──────────
            # ── 0b-TRAIL: Swing trailing — SL físico en MT5 al superar peak proporcional al riesgo ─
            # Problema: bot cicla cada 30s, precio puede caer de +$18 a +$1 en segundos
            # Solución: cuando peak >= umbral → mover SL MT5 a entry (breakeven) inmediato
            # Esto protege a nivel broker, independiente del ciclo del bot
            # BUG-BE-TOO-EARLY (2026-07-03): umbral fijo de $10 disparaba en cuanto el
            # precio hacia ruido normal, mucho antes del 1:1 RR real (~$275-390 en estas
            # posiciones) donde el mecanismo de partial-close (linea ~2438) esta pensado
            # para actuar. Resultado: dos trades reales (USDCAD, NZDUSD) que llegaron a
            # peak $30-90 volvieron a breakeven exacto ($0.00 neto) sin nunca tener
            # oportunidad de alcanzar el partial ni el TP. Fix original: umbral = 50% del
            # riesgo real de la posicion (via SL distance), con piso de $10.
            #
            # BUG-BE-STILL-TOO-EARLY (2026-07-09): auditoria de episodes.db mostro avg
            # WIN=$27.72 vs avg LOSS=$22.02 (ratio real 1.26x) pese a que el sistema esta
            # calculado para RR=3.0 -- decenas de "wins" de $0.24-$8 en el historial son
            # exactamente el patron de romper a breakeven apenas a mitad de camino del
            # riesgo real. El umbral de 50% de 1R seguia siendo demasiado temprano -- el
            # propio comentario original de esta funcion (arriba) dice "mover a breakeven
            # cuando profit >= 1R", pero el codigo aplicaba 0.5R. Subido a 1.0R real para
            # que las ganadoras tengan el espacio que el docstring siempre dijo que debian
            # tener antes de asegurar breakeven.
            #
            # BUG-BE-KILLS-RR (2026-07-17): auditoria completa de episodes.db (593 trades
            # cerrados) confirmo que el problema persistia incluso con el umbral en 1.0R:
            # avg WIN=$27.72 vs avg LOSS=$20.15 -- RR real 1.38, contra el RR=3.0 de diseno.
            # Causa mecanica: al llegar a 1R el SL saltaba a entry EXACTO (0R). Cualquier
            # ruido normal de precio que retrocede hasta entry -- algo que pasa en casi
            # toda posicion que ya estuvo en +1R -- saca el trade en ~$0 antes de que
            # tenga oportunidad de acercarse al TP real (3R). Fix: subir el umbral a 2.0R
            # (mas espacio antes de tocar el stop) y, al dispararse, asegurar 0.5R de
            # ganancia en vez de breakeven exacto -- protege capital sin borrar la ganancia
            # entera en el primer retroceso.
            _be_moved = self.__dict__.setdefault("_breakeven_set", set())
            for sw in list(swing_positions):
                sw_pnl    = sw.get("profit", 0.0)
                sw_ticket = sw["ticket"]
                sw_sym    = sw.get("symbol", "?")
                sw_entry  = sw.get("price_open", 0.0)
                sw_type   = sw.get("type", "BUY")
                sw_tp     = sw.get("tp", 0.0)
                sw_sl     = sw.get("sl", 0.0)
                # Actualizar peak
                _peak = self._position_peaks.get(sw_ticket, 0.0)
                if sw_pnl > _peak:
                    self._position_peaks[sw_ticket] = sw_pnl
                    _peak = sw_pnl
                # Umbral proporcional al riesgo real (2.0R -- ver BUG-BE-KILLS-RR
                # arriba), piso $10
                _be_trigger = 10.0
                _be_sl_dist = 0.0
                if sw_sl and sw_entry:
                    try:
                        from core.volume_calculator import VolumeCalculator as _VC
                        _be_base      = _VC._norm(sw_sym)
                        _be_pip_size  = _VC._PIP_SIZE.get(_be_base, 0.0001)
                        _be_pip_value = _VC._PIP_VALUE.get(_be_base, 10.0)
                        _be_sl_dist   = abs(sw_entry - sw_sl)
                        _be_sl_pips   = _be_sl_dist / _be_pip_size if _be_pip_size else 0.0
                        _be_risk_usd  = sw.get("volume", 0.0) * _be_sl_pips * _be_pip_value
                        if _be_risk_usd > 0:
                            _be_trigger = max(10.0, _be_risk_usd * 2.0)
                    except Exception:
                        pass
                # Cuando peak >= umbral → asegurar 0.5R de ganancia en MT5 (no breakeven
                # exacto -- ver BUG-BE-KILLS-RR), una sola vez
                if _peak >= _be_trigger and sw_ticket not in _be_moved and sw_entry > 0:
                    _be_lock_price = sw_entry
                    if _be_sl_dist > 0:
                        _be_offset = _be_sl_dist * 0.5
                        _be_lock_price = sw_entry + _be_offset if sw_type == "BUY" else sw_entry - _be_offset
                    _be_ok = await loop.run_in_executor(
                        None, lambda t=sw_ticket, e=_be_lock_price, tp=sw_tp: self.mt5.modify_position_sl_tp(t, e, tp)
                    )
                    if _be_ok:
                        _be_moved.add(sw_ticket)
                        print(f"[BE-SET] {sw_sym} #{sw_ticket} peak=${_peak:.2f} → SL MT5 movido a +0.5R {_be_lock_price:.5f}", flush=True)
                # BUG-DOUBLE-PEAK-GUARD (2026-07-13): este cierre por software
                # (peak>=$100, retrocede 50%) duplicaba al guardia PEAK-GUARD de
                # mas abajo (linea ~3376: peak>=$200, retrocede 30%), pero con un
                # umbral MUCHO mas bajo -- por eso este disparaba primero y
                # aseguraba apenas $50 de una posicion con peak $100, sin dejarle
                # nunca la oportunidad de llegar al TP real (a veces +$200-400).
                # Auditoria de episodes.db: max win historico jamas alcanzado
                # fue $279.04 pese a TPs disenados para RR=3.0 -- este guardia
                # duplicado y prematuro es la causa mecanica. Eliminado; el
                # breakeven-set de arriba (linea 3181) ya protege el capital sin
                # cortar la ganancia, y PEAK-GUARD mas abajo protege peaks
                # grandes con un umbral realista.

            # ── 0b. Swing dollar-stop: solo cierre de emergencia — dejar que SL del broker actúe
            # No cerrar antes del SL: los swings USDCAD necesitan tiempo para llegar al TP (+$247)
            # El broker SL ya protege el capital. Auto-close solo si falla el broker SL (emergencia).
            SWING_MAX_LOSS = -150.0  # emergencia si broker SL falla — no interferir antes
            for sw in list(swing_positions):
                sw_pnl    = sw.get("profit", 0.0)
                sw_ticket = sw["ticket"]
                sw_sym    = sw.get("symbol", "?")
                sw_auto_close = sw_pnl <= SWING_MAX_LOSS  # solo emergencia — NO cerrar por tiempo
                if sw_auto_close:
                    ok = await self._close_guarded(
                        loop, sw_ticket, "SWING-STOP",
                        f"<b>SWING STOP -$50</b>\n{sw_sym} #{sw_ticket}\nCerrado en ${sw_pnl:.2f}"
                    )
                    if ok:
                        # Registrar cooldown: no reabrir este par/dirección por 4 horas
                        import time as _t; _sw_dir = "BUY" if sw.get("type") == "BUY" else "SELL"
                        self._symbol_sl_time[f"{sw_sym}_{_sw_dir}"] = _t.time()
                        try:
                            json.dump(self._symbol_sl_time, open(os.path.join("memory", "sl_cooldown_state.json"), "w"))
                        except Exception:
                            pass
                        print(f"[SWING-STOP] {sw_sym} #{sw_ticket} cerrado ${sw_pnl:+.2f} → COOLDOWN 4h (persistido)", flush=True)

            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() == 4:  # Friday
                past_cutoff = (now_utc.hour > FRIDAY_CLOSE_HOUR or
                               (now_utc.hour == FRIDAY_CLOSE_HOUR and now_utc.minute >= FRIDAY_CLOSE_MIN))
                if past_cutoff:
                    # Cierra TODO antes del fin de semana — ganadoras y perdedoras.
                    # Antes solo cerraba perdedoras; una ganadora abierta el viernes
                    # queda expuesta al mismo gap risk que una perdedora el lunes.
                    for lp in list(positions):
                        sym    = lp.get("symbol", "?")
                        ticket = lp["ticket"]
                        pnl    = lp.get("profit", 0.0)
                        estado = "perdiendo" if pnl < 0 else "ganando"
                        print(
                            f"[FRIDAY-CLOSE] {sym} #{ticket} {estado} ${pnl:.2f} "
                            f"— cerrando antes del fin de semana (19:30 UTC)",
                            flush=True,
                        )
                        ok = await self._close_guarded(
                            loop, ticket, "FRIDAY-CLOSE",
                            f"<b>CIERRE VIERNES</b> {sym} #{ticket}\n"
                            f"Cerrado antes del fin de semana.\n"
                            f"P&amp;L: ${pnl:.2f}"
                        )
                        if not ok:
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
                ok = await self._close_guarded(
                    loop, drag_ticket, "ANTI-DRAG",
                    f"<b>ANTI-DRAG CLOSE</b>\n{drag_sym} #{drag_ticket}\n"
                    f"Perdida ${abs(drag_pnl):.2f} cancelaba ganancias (neto ${net_pnl:+.2f})\n"
                    f"→ Perdedora cerrada. Ganadoras protegidas."
                )
                if ok:
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
                    await asyncio.sleep(1.0)
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
                            await self._close_guarded(loop, ticket, "NO-SL-CLOSE")
                            continue

                # ── 1. Loss protection ─────────────────────────────────────
                # BUG-LOSS-LIMIT-FLAT (2026-07-21): used to be one flat
                # 0.8%-of-balance threshold (limit_usd) for every trade
                # regardless of intended risk_pct -- see _position_intended_risk
                # comment at open time. Scale to 2x the trade's actual intended
                # risk when known (swings only); fall back to the flat balance
                # limit for scalps or any ticket opened before this fix.
                _intended = self._position_intended_risk.get(ticket)
                _ticket_limit_usd = max(10.0, _intended * 2.0) if _intended else limit_usd
                if pnl < -_ticket_limit_usd:
                    print(
                        f"[AUTO-CLOSE] {sym} #{ticket} perdiendo ${abs(pnl):.2f}"
                        f" > limite ${_ticket_limit_usd:.0f} → cerrando",
                        flush=True,
                    )
                    await self._close_guarded(
                        loop, ticket, "LOSS-LIMIT",
                        f"<b>AUTO-CIERRE PERDIDA</b>\n{sym} #{ticket}\n"
                        f"Perdida ${abs(pnl):.2f} > limite → cerrado"
                    )
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
                        await self._close_guarded(
                            loop, ticket, "PEAK-GUARD",
                            f"<b>GANANCIA ASEGURADA</b>\n{sym} #{ticket}\n"
                            f"Peak: ${peak:.2f} → Retroceso 30% → cerrado en ${pnl:.2f}"
                        )
                        continue
                else:
                    self._position_peaks.pop(ticket, None)

                # ── 1c. Stagnation guard ────────────────────────────────────
                # BUG-STAGNANT-NO-GUARD (2026-07-20): a position that never
                # loses enough to trip LOSS-LIMIT and never wins enough to
                # trip PEAK-GUARD ($200) had NO guard at all -- it could sit
                # open indefinitely, drifting near breakeven for a full day
                # or more, without ever reaching its real SL or TP. That's
                # capital tied up producing neither a result nor a learning
                # signal, and it occupies one of only MAX_OPEN_POSITIONS
                # slots that a fresh, better setup could use instead. Close
                # it once it's been open a long time and never showed real
                # movement in either direction.
                # 2026-07-20 (pedido usuario): bajado de 12h a 4h -- no quiere
                # posiciones ocupando un cupo todo el dia sin moverse. Y no
                # debe forzar el cierre a la primera perdida que encuentre:
                # una vez marcada estancada, prefiere cerrar en beneficio/
                # breakeven; si esta perdiendo, da un margen corto
                # (STAGNANT_GRACE_HOURS) esperando que suba a >=0 antes de
                # forzar el cierre en la menor perdida disponible en ese
                # momento -- nunca espera indefinidamente.
                STAGNANT_HOURS       = 4.0
                STAGNANT_PEAK_MAX    = 15.0   # never even reached this much peak profit
                STAGNANT_GRACE_HOURS = 2.0    # margen extra esperando pnl >= 0 antes de forzar
                if open_time > 0:
                    import time as _stagn_time
                    _now_stagn = _stagn_time.time()
                    age_h = (_now_stagn - open_time) / 3600.0
                    peak_seen = self._position_peaks.get(ticket, max(pnl, 0.0))
                    _stagn_flags = self.__dict__.setdefault("_stagnant_flagged", {})
                    if age_h >= STAGNANT_HOURS and peak_seen < STAGNANT_PEAK_MAX:
                        flagged_at = _stagn_flags.get(ticket)
                        if flagged_at is None:
                            _stagn_flags[ticket] = _now_stagn
                            flagged_at = _now_stagn
                        grace_elapsed_h = (_now_stagn - flagged_at) / 3600.0
                        should_close = pnl >= 0 or grace_elapsed_h >= STAGNANT_GRACE_HOURS
                        if should_close:
                            _motivo = "en breakeven/beneficio" if pnl >= 0 else f"forzado tras {grace_elapsed_h:.1f}h de margen"
                            print(
                                f"[STAGNANT] {sym} #{ticket} abierta {age_h:.1f}h, peak nunca superó "
                                f"${STAGNANT_PEAK_MAX:.0f} (max visto ${peak_seen:.2f}) → cerrando {_motivo}, "
                                f"actual ${pnl:+.2f} -- no llega a TP ni a SL",
                                flush=True,
                            )
                            ok = await self._close_guarded(
                                loop, ticket, "STAGNANT",
                                f"<b>CIERRE POR ESTANCAMIENTO</b>\n{sym} #{ticket}\n"
                                f"Abierta {age_h:.1f}h sin movimiento real → cerrada {_motivo} en ${pnl:+.2f}"
                            )
                            if ok:
                                _stagn_flags.pop(ticket, None)
                        # Flagged as stagnant (closing now or still in the grace
                        # window) -- either way, skip trailing-stop logic below,
                        # it doesn't apply to a position that never developed.
                        continue
                    else:
                        _stagn_flags.pop(ticket, None)

                # ── 2-3. Trailing stop (only for winning positions) ────────
                if entry > 0 and sl_cur > 0:
                    sl_dist = abs(entry - sl_cur)  # 1R distance
                    if sl_dist < 0.0001:           # SL ya en breakeven o muy cerca → skip trailing
                        sl_dist = 0.0              # fuerza skip del bloque abajo
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

                # ── 3b. Structure invalidation: LOSING position whose H4 bias
                # reversed against it. NOTE: H4=WAIT is deliberately NOT treated
                # as invalidation here -- the scan loop already preserves the
                # prior LONG/SHORT through WAIT reads to avoid closing on normal
                # pullback noise (see BUG #H4-WAIT-PERMANENT). Only a genuine
                # flip to the opposite direction counts -- that means the setup
                # that justified the trade is gone, not just quiet.
                # Added 2026-07-08: positions were holding to the full SL/36h
                # timer even after their own structural justification reversed,
                # with nothing acting on it until a human checked manually.
                if pnl < 0:
                    import time as _stime
                    _pos_dir = "LONG" if is_buy else "SHORT"
                    _opposite_dir = "SHORT" if is_buy else "LONG"
                    _cur_h4_dir = self._mt5_h4_direction.get(sym)
                    if _cur_h4_dir == _opposite_dir:
                        _last_try = self._close_attempted.get(ticket, 0.0)
                        if _stime.time() - _last_try >= 300:
                            self._close_attempted[ticket] = _stime.time()
                            print(
                                f"[STRUCTURE-INVALID] {sym} #{ticket} era {_pos_dir}, "
                                f"H4 ahora {_cur_h4_dir} -- perdiendo ${pnl:.2f} → cerrando",
                                flush=True,
                            )
                            ok = await self._close_guarded(
                                loop, ticket, "STRUCT-INVALID",
                                f"<b>CIERRE POR REVERSION DE ESTRUCTURA</b>\n{sym} #{ticket}\n"
                                f"Era {_pos_dir}, H4 ahora {_cur_h4_dir} → cerrada en ${pnl:+.2f}"
                            )
                            if ok:
                                self._close_attempted.pop(ticket, None)
                            continue

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
                        ok = await self._close_guarded(
                            loop, ticket, "TIME-CLOSE-36H",
                            f"<b>CIERRE POR TIEMPO</b>\n{sym} #{ticket}\n"
                            f"Abierta {age_h:.1f}h perdiendo → cerrada en ${pnl:+.2f}"
                        )
                        if ok:
                            self._close_attempted.pop(ticket, None)
                        else:
                            print(f"[TIME-CLOSE] {sym} #{ticket} close FALLO — reintento en 5min", flush=True)

        except Exception as _me:
            import traceback
            print(f"[AUTO-CLOSE] error monitor: {_me}\n{traceback.format_exc()[:300]}", flush=True)

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
        """Vision monitor — DESACTIVADO: sin creditos API Anthropic."""
        while self._running:
            await asyncio.sleep(86400)  # duerme 24h — efectivamente desactivado
        return  # desactivado — sin creditos API

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

                # ── Reset risk-gate diario + refresh daily_pnl ─────────────
                _today_utc = datetime.now(timezone.utc).date()
                if not hasattr(self, '_last_ftmo_day') or self._last_ftmo_day != _today_utc:
                    self._risk_gate_agent.new_trading_day(self._risk_gate_state)
                    self._last_ftmo_day = _today_utc
                    self._daily_target_hit  = False
                    self._daily_protect_hit = False
                    self._scalp_daily_hit   = False
                    self._scalp_realized_today = 0.0
                    print(f"[RISK-GATE] Nuevo dia {_today_utc} -- daily_pnl reseteado (streak preservado)", flush=True)
                    try:
                        _loop_ref = asyncio.get_running_loop()
                        self._risk_gate_state.daily_pnl_today = await _loop_ref.run_in_executor(
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

                # ── AXI SELECT: chequeo de capital y guard diario ──────────
                if self._mt5_available:
                    try:
                        _acc = await asyncio.get_running_loop().run_in_executor(
                            None, self.mt5.get_account_info
                        )
                        if _acc and _acc.get("equity", 0) > 0:
                            _eq  = _acc["equity"]
                            _bal = _acc.get("balance", _eq)
                            # AxiSelectGuard: emergencia si dia cae -4%
                            # (day_start_balance + paused_today are persisted
                            # to disk inside the guard -- survives pm2 restarts,
                            # see BUG-AXI-GUARD-RESTART)
                            _was_paused = self._axi_guard.paused_today
                            self._axi_guard.set_day_start(_bal)
                            _guard = self._axi_guard.check(_eq)
                            if _guard.should_close and not _was_paused:
                                print(f"[AXI-GUARD] EMERGENCY CLOSE: {_guard.reason}", flush=True)
                                try:
                                    await asyncio.get_running_loop().run_in_executor(
                                        None, self.mt5.close_all_positions
                                    )
                                    await self.telegram.send_glint_alert(
                                        f"<b>AXI GUARD — LIMITE DIARIO</b>\n{_guard.reason}\nBot pausado hasta manana."
                                    )
                                except Exception:
                                    pass
                            elif _guard.warning_level:
                                print(f"[AXI-GUARD] {_guard.reason}", flush=True)
                            # ── Cumulative/total drawdown force-close ──────────
                            # BUG-DD-NO-FORCE-CLOSE (2026-07-21, found by the
                            # risk-management expert panel): check_drawdown_limit
                            # (strategies/ftmo_agent.py) only ever fed into
                            # can_trade (blocks NEW entries in _send_mt5_real_order)
                            # -- it never closed positions already open. Unlike
                            # AxiSelectGuard (daily -4%, resets every UTC day),
                            # the cumulative 8% ceiling has no force-close at all,
                            # so a string of sub-4%-per-day losing days (e.g.
                            # -3.9% repeated) never trips the daily guard while
                            # still compounding past 8-10% total, with open
                            # positions left to ride their own SL/LOSS-LIMIT/
                            # TIME-CLOSE-36H exits. Fires once (not daily-reset,
                            # since this is a total-drawdown event, not a daily
                            # one) -- a real breach here means the challenge is
                            # already failing, not "wait for tomorrow".
                            if not self._dd_force_closed:
                                self._risk_gate_state.current_balance = _bal
                                _dd_ok, _dd_reason = self._risk_gate_agent.check_drawdown_limit(
                                    self._risk_gate_state, equity=_eq
                                )
                                if not _dd_ok:
                                    print(f"[DD-GUARD] EMERGENCY CLOSE: {_dd_reason}", flush=True)
                                    self._dd_force_closed = True
                                    try:
                                        await asyncio.get_running_loop().run_in_executor(
                                            None, self.mt5.close_all_positions
                                        )
                                        await self.telegram.send_glint_alert(
                                            f"<b>DRAWDOWN TOTAL — CIERRE DE EMERGENCIA</b>\n{_dd_reason}"
                                        )
                                    except Exception:
                                        pass
                            # AxiCapitalAdjuster: detecta si Axi escalo capital
                            _adj = self._axi_adjuster.check(_bal)
                            if _adj.adjusted:
                                print(f"[AXI-CAPITAL] {_adj.reason}", flush=True)
                                self._axi_tracker.set_capital(_bal)
                                try:
                                    await self.telegram.send_glint_alert(
                                        self._axi_adjuster.format_telegram(_adj)
                                    )
                                except Exception:
                                    pass
                    except Exception as _axi_chk_exc:
                        print(f"[AXI-CHECK] error: {_axi_chk_exc}", flush=True)

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

                                # Cache H4: actualizado desde análisis estructural en _scan_mt5_symbol.
                                # Aquí solo gestionamos el flag H4-NEW y evitamos sobreescribir con WAIT.
                                if tf == "H4":
                                    _h4_prev = self._mt5_h4_direction.get(symbol, "UNKNOWN")
                                    if bias in ("LONG", "SHORT"):
                                        # Señal con dirección clara → actualizar siempre
                                        self._mt5_h4_direction[symbol] = bias
                                    elif _h4_prev == "UNKNOWN":
                                        # Primera vez que se ve: setear WAIT
                                        self._mt5_h4_direction[symbol] = "WAIT"
                                    # Si bias==WAIT pero prev==LONG/SHORT → preservar dirección estructural
                                    # (el momentum filter bloqueó la señal pero la estructura sigue igual)
                                    # Si H4 acaba de confirmar (WAIT→LONG/SHORT): marcar como nuevo
                                    if _h4_prev in ("WAIT", "UNKNOWN") and bias in ("LONG", "SHORT"):
                                        self._mt5_h4_just_confirmed[symbol] = True
                                        print(f"[H4-NEW] {symbol}: H4 acaba de confirmar {bias} — 1 ciclo de espera", flush=True)
                                    elif bias in ("LONG", "SHORT"):
                                        self._mt5_h4_just_confirmed.pop(symbol, None)  # confirmado 2+ ciclos

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

                                    # APRENDIZAJE 24-Jun: H4 recién confirmado (WAIT→DIR) = esperar 1 ciclo
                                    # Evita swing chase cuando H4 acaba de girar (AUDUSD -$12.54)
                                    if self._mt5_h4_just_confirmed.get(symbol):
                                        print(f" -- [H4-NEW] {symbol} H4 recién confirmó — esperando 1 ciclo", flush=True)
                                        continue

                                    # MOMENTUM CHECK para swings H4: precio debe moverse hacia la señal
                                    # Si últimas 3 velas H4 van CONTRA la dirección → skip
                                    # Evita abrir EURUSD SELL cuando precio sube (múltiples pérdidas 25-Jun)
                                    if tf == "H4":
                                        try:
                                            _h4_df = await loop.run_in_executor(None, lambda s=symbol: self.mt5.get_ohlcv(s, "H4", 10))
                                            if _h4_df is not None and len(_h4_df) >= 4:
                                                _c_now  = float(_h4_df["close"].iloc[-1])
                                                _c_3ago = float(_h4_df["close"].iloc[-4])
                                                _h4_mom = "UP" if _c_now > _c_3ago else "DOWN"
                                                if bias == "SHORT" and _h4_mom == "UP":
                                                    print(f" -- [MOM-H4] SELL pero H4 subiendo — skip", flush=True)
                                                    continue
                                                if bias == "LONG" and _h4_mom == "DOWN":
                                                    print(f" -- [MOM-H4] BUY pero H4 bajando — skip", flush=True)
                                                    continue
                                        except Exception:
                                            pass

                                    print(f" -- [D1={d1_dir} H4={h4_dir or '?'}] OK", end="", flush=True)

                                # Solo swings: H4 (principal) y H1 (adicional)
                                # M15 DESACTIVADO — scalps destruian capital (Jun 25: 79 trades, -$336)
                                # Backtest 6meses: threshold=80 da 6-7 señales/dia, P(dia>=$250)=46%
                                # vs threshold=100 que da 2 señales/dia, P(dia>=$250)=31%
                                if tf == "H4":
                                    effective_threshold = mt5_threshold  # ADAPT-THR dinamico (75-80)
                                elif tf == "H1":
                                    # H1 swing: preferible con H4 confirmado, pero permite con score alto
                                    _h4_now = self._mt5_h4_direction.get(symbol, "WAIT")
                                    if _h4_now == "WAIT":
                                        if score >= 100:
                                            effective_threshold = 100  # D1+H1 triple confirm sin H4
                                        else:
                                            print(f" -- [H1-SKIP] H4=WAIT, score {score}<100 sin confirmacion")
                                            continue
                                    else:
                                        effective_threshold = max(MT5_SCORE_AUTO_REDUCE, mt5_threshold - 5)  # H1 siempre 5pts menos que H4, piso=90
                                else:
                                    continue  # M15 y cualquier otro TF: skip
                                # Killzone multiplier: threshold mas bajo en horas gold (14-16 UTC),
                                # mas alto en horas debiles (17-18 UTC) — backtest WR 61% vs 24-28%
                                from core.session_manager import session_multiplier as _kz_mult
                                _kz = _kz_mult()
                                _kz_threshold = max(MT5_SCORE_AUTO_REDUCE, int(effective_threshold / _kz))  # piso=90, coherente con fix 2026-06-30 (era 70)
                                if _kz != 1.0:
                                    print(f" -- [KZ] hora={__import__('datetime').datetime.now(__import__('datetime').timezone.utc).hour}UTC mult={_kz:.2f} thr={effective_threshold}->{_kz_threshold}", end="", flush=True)
                                    effective_threshold = _kz_threshold
                                # AutonomousLearner: aplicar el weight_adj real calculado cada hora
                                # (antes solo se calculaba y guardaba en episodes.db, nunca se usaba
                                # para ajustar ninguna decision en vivo -- hallazgo 2026-07-06)
                                #
                                # BUG-LEARN-THR-HARDCODED (2026-07-21): esta llamada pasaba los
                                # literales fijos "SMC"/"unknown"/"unknown" en vez del trigger real
                                # de la señal -- exactamente la misma clase de bug que
                                # BUG-TRIGGER-HARDCODED (2026-07-09), pero en este call site nunca
                                # se corrigió. Efecto: TODA señal, de cualquier símbolo o setup,
                                # consultaba siempre el mismo bucket ("SMC","unknown","unknown") --
                                # 485 muestras, WR=25%, TODAS anteriores al fix del trigger real del
                                # 09-jul (desde entonces episodes.db graba setup_type especifico como
                                # "BOS + CHoCH + OB + FVG", nunca mas "SMC") -- un bucket muerto y
                                # congelado que ya no puede recibir datos nuevos, aplicando SIEMPRE
                                # un 25% extra de exigencia al threshold (adj=0.80) sin relacion con
                                # la señal real que se estaba evaluando. Verificado en vivo el
                                # 2026-07-21: 5,321 "sin setup" contra 1,069 bloqueos direccionales
                                # combinados (D1/H4/TREND) en un solo dia -- el score casi nunca
                                # llegaba a evaluarse. Fix: usar signal.trigger real (regime/session
                                # quedan "unknown" -- no se calculan en este punto del scan, antes de
                                # saber si el trade sigue adelante -- asi que ya no calzan con el
                                # bucket viejo y el ajuste vuelve a neutral hasta que se acumule
                                # historial real por (trigger, regime, session) especifico).
                                # BUG-TREND-BLOCK-DUMMY-SIGNAL (2026-07-21, encontrado en vivo
                                # minutos despues del fix de arriba): el guard TREND-BLOCK (~linea
                                # 1493) devuelve un objeto minimo `type('S', (), {...})()` con solo
                                # signal_type/decision_score -- sin atributo .trigger. Cualquier
                                # simbolo bloqueado por TREND-BLOCK (NZDUSD, visto en vivo) tiraba
                                # AttributeError aqui mismo en cada ciclo. getattr con default en
                                # vez de asumir que todo objeto "signal" tiene el mismo shape.
                                _learner_thr = self._learner.effective_threshold(
                                    effective_threshold, getattr(signal, "trigger", "unknown"), "unknown", "unknown"
                                )
                                if _learner_thr != effective_threshold:
                                    print(f" -- [LEARN-THR] {effective_threshold}->{_learner_thr}", end="", flush=True)
                                    effective_threshold = _learner_thr
                                if signal.signal_type == SignalType.WAIT or score < effective_threshold:
                                    self._scan_stats["blocked_score"] += 1
                                    print(f" -- sin setup (threshold={effective_threshold})")
                                else:
                                    # ICT Silver Bullet gate (2026-07-09): dentro de la kill zone
                                    # activa (14 UTC = 10-11am ET, la unica que solapa con las
                                    # horas activas reales del bot), un trader ICT real exige la
                                    # confluencia COMPLETA (sweep+FVG+kill zone) en vez de un score
                                    # ponderado -- si falta cualquier pieza, no opera, sin importar
                                    # que tan alto sume el resto. Fuera de esa hora, sigue el
                                    # criterio de score de siempre (sin este gate adicional).
                                    if tf == "H1" and datetime.now(timezone.utc).hour == 14:
                                        try:
                                            _sb_df = await loop.run_in_executor(
                                                None, lambda s=symbol: self.mt5.get_ohlcv(s, "H1", 30)
                                            )
                                            _sb = _silver_bullet_check(_sb_df) if _sb_df is not None else None
                                            _sb_dir = "bullish" if bias == "LONG" else "bearish"
                                            if _sb is None or not _sb.valid or _sb.direction != _sb_dir:
                                                print(f" -- [SILVER-BULLET] confluencia incompleta (sweep+FVG+killzone) -- skip", flush=True)
                                                continue
                                            print(f" -- [SILVER-BULLET] confluencia completa confirmada", flush=True)
                                        except Exception as _sb_exc:
                                            print(f" -- [SILVER-BULLET] error verificando (no bloqueo): {_sb_exc}", flush=True)
                                    # BUG-EXEC-LOG-MISLEADING (2026-07-20): este print decia
                                    # "ejecutando SWING" pero _send_mt5_real_order todavia tiene
                                    # ~10 filtros propios (hora muerta, cooldown, RR, spread,
                                    # AXI guards, max posiciones...) que pueden abortar la orden
                                    # despues de este punto -- el log mentia sobre una accion que
                                    # no habia pasado. "intentando" refleja lo que de verdad se
                                    # sabe en este punto del codigo.
                                    print(f" -- intentando SWING (score={score}>={effective_threshold})")
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



