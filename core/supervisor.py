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

DEMO_SCORE_THRESHOLD     = 70   # simulated crypto demo trades
MT5_REAL_SCORE_THRESHOLD = 72   # sprint: 75→72 for 2-day Axi challenge
MT5_SCORE_AUTO_REDUCE    = 68   # fallback after 1h sin trades (sprint mode)
MT5_SCORE_REDUCE_AFTER_H = 1    # reduce faster in sprint (was 2h)
DEMO_MAX_POSITIONS       = 5
SCAN_INTERVAL_SEC        = 30

# Conservative mode disabled — 8 filters + Claude API confirmation are sufficient
CONSERVATIVE_MODE        = False   # was True — disabled now that pipeline is complete
CONSERVATIVE_SCORE_MIN   = 75
CONSERVATIVE_PAIRS       = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "US30"]
MAX_DAILY_TRADES         = 3       # sprint: 3/day for 2-day Axi challenge (was 2)
MAX_OPEN_POSITIONS       = 2       # max simultaneous open positions
MIN_RR                   = 2.0    # minimum risk:reward

# Dead hours (UTC) -- no new orders during low-liquidity windows
DEAD_HOURS_UTC           = set(range(22, 24)) | {0, 1}  # 22:00-01:59 UTC (was 21-02)



# Symbols and timeframes to scan

SCAN_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

SCAN_TIMEFRAMES = ["1h", "4h"]  # removed 5m/15m -- too noisy for quality setups



# MT5 forex/indices symbols

MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]
MT5_TIMEFRAMES   = ["H1", "H4"]  # H1 + H4 for more signal opportunities

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

        self._demo_trades: List[DemoTrade] = []

        self.mode    = config.operation_mode

        self._running = False

        # Autonomous mode

        self._episodic_conn = get_db()

        self._learner   = AutonomousLearner(conn=self._episodic_conn)

        self._researcher = ResearchAgent(conn=self._episodic_conn)

        self._goals_mgr  = GoalsManager(conn=self._episodic_conn)

        self._reporter   = NightlyReporter(conn=self._episodic_conn)

        self._open_episodes: Dict[int, int] = {}  # ticket -> episode_id

        # Load daily trade count from disk so pm2 restarts don't reset the limit
        self._daily_trades: Dict[str, int] = self._load_daily_trades()

        # FTMO / Axi rules enforcement

        self._ftmo_agent = FTMOAgent()

        self._ftmo_state = FTMOAgent.new_challenge(

            initial_balance=capital,

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

            print(f"  MT5:           CONECTADO -- Balance ${bal:,.2f}")

            print(f"  Forex:         {', '.join(MT5_SYMBOLS)}")

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



        await asyncio.gather(

            self.commander.start_polling(),

            self.glint.connect(),

            self._market_scan_loop(),

            self._position_monitor_loop(),

            self._learning_loop(),

            self._research_loop(),

            self._goals_loop(),

            self._nightly_report_loop(),

            self._vision_monitor_loop(),

        )



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



        poi_zones = []

        for ob in (bull_obs + bear_obs)[:3]:

            poi_zones.append(ob)



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

    def _load_daily_trades(self) -> Dict[str, int]:
        """Load daily MT5 trade count from disk — survives pm2 restarts."""
        import json
        try:
            with open(self._DAILY_TRADES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
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

        if signal.decision_score < 65:
            return 0

        bias = "bullish" if signal.signal_type == SignalType.LONG else "bearish"
        prices = list(df["close"].astype(float).values) if not df.empty else []

        def _lunar():
            try: return self._lunar.score_adjustment(bias)
            except Exception: return 0

        def _elliott():
            try:
                e = self._elliott.analyze(df, bias)
                return e.score_bonus
            except Exception: return 0

        def _chaos():
            try: return self._chaos.score_adjustment(df)
            except Exception: return 0

        def _edge():
            try:
                edge = self._edge.calculate_full_edge(symbol=signal.symbol, prices=prices)
                return self._edge.get_decision_pts(edge)
            except Exception: return 0

        def _footprint():
            if signal.symbol not in SCAN_SYMBOLS:
                return 0
            try:
                fp_candle = self._footprint.build_live_footprint(
                    signal.symbol, candle_open=signal.entry or 0, limit=500
                )
                direction = "long" if signal.signal_type == SignalType.LONG else "short"
                return self._footprint.score_for_trade(fp_candle, direction, signal.entry or 0)
            except Exception: return 0

        def _instflow():
            try: return self._inst_flow.score_adjustment(signal.symbol, bias)
            except Exception: return 0

        def _micro():
            try: return self._microstructure.score_adjustment(signal.symbol, signal.entry or 0.0)
            except Exception: return 0

        def _fed():
            try: return self._fed.score_adjustment(signal.symbol, bias)
            except Exception: return 0

        def _onchain():
            try: return self._onchain.score_adjustment(signal.symbol, bias, signal.entry or 0.0)
            except Exception: return 0

        def _geo():
            try: return self._geopolitical.score_adjustment(signal.symbol, bias)
            except Exception: return 0

        def _retail():
            try: return self._retail_psych.score_adjustment(signal.symbol, df, bias)
            except Exception: return 0

        def _alt():
            try: return self._alt_data.score_adjustment(signal.symbol, bias)
            except Exception: return 0

        def _energy():
            try:
                energy = self._energy.analyze(signal.symbol, signal.entry or 0.0, prices)
                return energy.to_decision_pts()
            except Exception: return 0

        tasks = [
            _lunar, _elliott, _chaos, _edge, _footprint,
            _instflow, _micro, _fed, _onchain, _geo,
            _retail, _alt, _energy,
        ]

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(fn) for fn in tasks]
            bonus = sum(f.result() for f in as_completed(futures))

        return int(max(-30, min(60, bonus)))



    async def _scan_symbol(self, symbol: str, timeframe: str) -> Optional[TradeSignal]:

        """

        Full pipeline for one symbol/timeframe:

        fetch â†' SMC lite â†' SignalAgent â†' DecisionFilter â†' return signal or None

        """

        loop = asyncio.get_event_loop()

        df = await loop.run_in_executor(

            None, lambda: self.binance.get_ohlcv(symbol, timeframe, limit=200)

        )

        if df.empty or len(df) < 50:

            return None



        smc = self._run_smc_lite(df)

        current_price = float(df["close"].iloc[-1])



        signal = self.signal_agent.evaluate(

            analysis_text = smc["analysis_text"],

            symbol        = symbol,

            timeframe     = timeframe,

            current_price = current_price,

            poi_zones     = smc["poi_zones"],

            glint_context = self._last_glint_text,

        )



        if signal.signal_type == SignalType.WAIT:

            return signal   # still return so we can log the score=0



        signal = self.route_signal(signal, df)



        # Institutional agent enrichment

        if signal.signal_type != SignalType.WAIT:

            agent_bonus = self._enrich_with_agents(signal, df)

            if agent_bonus != 0:

                signal.decision_score = max(0, min(150, signal.decision_score + agent_bonus))

                signal.score_breakdown["agents"] = agent_bonus



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

        )

        if signal.signal_type == SignalType.WAIT:

            return signal

        signal = self.route_signal(signal, df)



        # Institutional agent enrichment (Elliott + Chaos + Quant edge)

        if signal.signal_type != SignalType.WAIT:

            agent_bonus = self._enrich_with_agents(signal, df)

            if agent_bonus != 0:

                signal.decision_score = max(0, min(150, signal.decision_score + agent_bonus))

                signal.score_breakdown["agents"] = agent_bonus



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

        sl_val = signal.stop_loss if signal.stop_loss else 0.0

        tp_val = signal.take_profit if signal.take_profit else 0.0



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

        if now_utc.weekday() == 4 and now_utc.hour >= 16:  # viernes 16:00+ UTC

            print(f"[MT5] {signal.symbol}: viernes 16:00+ UTC, skip", flush=True)

            return



        # ── FILTER 4: RR minimo 1:2 ───────────────────────────────────────

        if tp_val > 0 and sl_val > 0 and signal.entry and signal.entry > 0:

            sl_dist = abs(signal.entry - sl_val)

            tp_dist = abs(signal.entry - tp_val)

            rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

            if rr < MIN_RR:

                print(f"[MT5] {signal.symbol}: RR={rr:.2f} < {MIN_RR} minimo, skip", flush=True)

                return



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

            acc_info = await asyncio.get_running_loop().run_in_executor(

                None, self.mt5.get_account_info

            )

            self._ftmo_state.current_balance = acc_info.get("balance", self.capital)

            can_trade, reason = self._ftmo_agent.can_trade(self._ftmo_state)

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



        # ── FILTER 7: Max posiciones abiertas (por simbolo y total) ─────────
        loop = asyncio.get_running_loop()
        existing = await loop.run_in_executor(None, self.mt5.get_positions)

        if len(existing) >= MAX_OPEN_POSITIONS:
            self._scan_stats["blocked_duplicate"] += 1
            print(f"[MT5] {signal.symbol}: {len(existing)} posiciones abiertas (max={MAX_OPEN_POSITIONS}), skip", flush=True)
            return

        sym_open = [p for p in existing if p["symbol"] == signal.symbol]
        if sym_open:

            pos = sym_open[0]

            pnl_live = pos.get("profit", 0.0)

            print(

                f"[MT5] {signal.symbol}: posicion {pos['type']} ya abierta "

                f"({pnl_live:+.2f} USD) -- skip nueva orden",

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

        volume = vc.calculate_volume(self.capital, signal.entry or sl_val, sl_val, signal.symbol)

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

                    "score": signal.decision_score,

                    "setup_type": "SMC",

                }, conn=self._episodic_conn)

                self._open_episodes[result["ticket"]] = eid

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

            print(f"[MT5 REAL] {signal.symbol} fallida: {result.get('error', '?')}", flush=True)



    async def _execute_demo_trade(self, signal: TradeSignal):

        """Record a simulated demo trade, notify via Telegram, log to SQLite."""

        # Expire demo trades older than 8 hours (one full trading session)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=8)
        self._demo_trades = [
            d for d in self._demo_trades
            if getattr(d, "opened_at", datetime.now(timezone.utc)) > cutoff
        ]

        # One slot per symbol — no duplicate positions on same pair
        if any(d.signal.symbol == signal.symbol for d in self._demo_trades):
            return

        if len(self._demo_trades) >= DEMO_MAX_POSITIONS:

            return



        demo = DemoTrade(signal, signal.decision_score)

        self._demo_trades.append(demo)



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
                current_tickets = {p["ticket"] for p in positions}
                for p in positions:
                    if p.get("sl", 0.0) == 0.0:
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



                # ── Detect closed positions ───────────────────────────────

                closed = _known_tickets - current_tickets

                for ticket in closed:

                    episode_id = self._open_episodes.pop(ticket, None)

                    deal = await loop.run_in_executor(

                        None, lambda t=ticket: self.mt5.get_closing_deal(t)

                    )

                    if not deal:

                        continue

                    pnl    = deal.get("profit", 0.0)

                    result = "WIN" if pnl > 0 else "LOSS"

                    # Update episodic memory

                    if episode_id:

                        try:

                            update_episode_result(

                                episode_id,

                                exit_price=deal.get("price", 0.0),

                                pnl=pnl,

                                result=result,

                                lesson=f"Score={self._open_episodes.get(ticket, '?')} -> {result} PnL={pnl:+.2f}",

                                conn=self._episodic_conn,

                            )

                        except Exception as _ue:

                            print(f"[LEARN] update error: {_ue}", flush=True)

                    # Update FTMO state

                    try:

                        self._ftmo_agent.record_trade(self._ftmo_state, pnl)

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

                    lines = [f"[POS] {len(positions)} abiertas | P&L vivo: {total_pnl:+.2f} USD"]

                    for p in positions:

                        lines.append(

                            f"  {p['symbol']} {p['type']} {p['volume']}lot "

                            f"P&L: {p.get('profit', 0.0):+.2f} USD"

                        )

                    print("\n".join(lines), flush=True)

            except Exception as exc:

                print(f"[POS MONITOR] error: {exc}", flush=True)



    # -- Autonomous background loops ----------------------------------------



    async def _learning_loop(self):

        while self._running:

            await asyncio.sleep(3600)  # every 1 hour

            try:

                self._learner.run_analysis()

                print("[LEARNER] Weight analysis complete", flush=True)

            except Exception as exc:

                print(f"[LEARNER] error: {exc}", flush=True)



    async def _research_loop(self):

        while self._running:

            await asyncio.sleep(7200)  # every 2 hours

            try:

                self._researcher.run_cycle()

            except Exception as exc:

                print(f"[RESEARCH] error: {exc}", flush=True)



    async def _goals_loop(self):

        while self._running:

            await asyncio.sleep(1800)  # every 30 min

            try:

                self._goals_mgr.evaluate()

            except Exception as exc:

                print(f"[GOALS] error: {exc}", flush=True)



    async def _nightly_report_loop(self):

        while self._running:

            await asyncio.sleep(60)  # check every 1 min

            try:

                now = datetime.now(timezone.utc)

                if self._reporter.should_fire(now):

                    date_str = now.strftime("%Y-%m-%d")

                    self._reporter.mark_fired(date_str)

                    await self._reporter.send(date_str)

            except Exception as exc:

                print(f"[NIGHTLY] error: {exc}", flush=True)



    async def _vision_monitor_loop(self):
        """Every 5 min: capture MT5 screen, alert on losing positions, auto-close if critical.
        In _vision_protect_mode: runs every 2 min instead."""
        _BALANCE_AT_START = 100_000.0  # Axi demo seed capital

        while self._running:
            interval = 120 if self._vision_protect_mode else 300
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



                # MT5 forex scan (real orders on demo account -- bypass demo slot limit)

                if self._mt5_available:
                    # Auto-reduce threshold after MT5_SCORE_REDUCE_AFTER_H hours without trade
                    last_ts = self._scan_stats.get("last_trade_ts")
                    if last_ts is not None:
                        hours_idle = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                        mt5_threshold = MT5_SCORE_AUTO_REDUCE if hours_idle > MT5_SCORE_REDUCE_AFTER_H else MT5_REAL_SCORE_THRESHOLD
                    else:
                        mt5_threshold = MT5_REAL_SCORE_THRESHOLD

                    for symbol in MT5_SYMBOLS:
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
                                if signal.signal_type == SignalType.WAIT or score < mt5_threshold:
                                    self._scan_stats["blocked_score"] += 1
                                    print(f" -- sin setup (threshold={mt5_threshold})")
                                else:
                                    print(f" -- ejecutando MT5 REAL (score={score}>={mt5_threshold})")
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



