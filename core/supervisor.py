import asyncio
import logging
from datetime import datetime, timezone
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

# Demo mode: lower score threshold so the bot actually trades while learning
DEMO_SCORE_THRESHOLD = 30
DEMO_MAX_POSITIONS   = 5
SCAN_INTERVAL_SEC    = 30

# Symbols and timeframes to scan
SCAN_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
SCAN_TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# MT5 forex/indices symbols
MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]
MT5_TIMEFRAMES   = ["H1", "H4"]
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

    Market data â†’ SMC analysis â†’ DecisionFilter (0-100)
        Demo:  score >= 40 â†’ execute simulated trade
        Live:  score >= 60 â†’ REDUCED | >= 75 â†’ FULL | >= 90 â†’ PREMIUM

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

    # â”€â”€ Callbacks from TelegramCommander â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_mode_change(self, mode: str):
        self.mode = mode
        print(f"[Mode] Cambiado a: {mode.upper()} vÃ­a Telegram")

    def _on_history_command(self, symbol: str) -> str:
        return self.historical.get_market_summary(symbol or "BTC")

    # â”€â”€ Glint callback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_glint_signal(self, glint: GlintSignal):
        if glint.is_actionable():
            self._last_glint_text = glint.text
            self._glint_buffer.append(glint.raw)
            if len(self._glint_buffer) > 20:
                self._glint_buffer.pop(0)
            asyncio.create_task(self.telegram.send_glint_alert(glint.format_alert()))
            print(f"[Glint] {glint.impact}: {glint.text[:80]}...")

    # â”€â”€ Decision pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            print(f"[FILTER] NO TRADE â€” {signal.notes}")
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

    # â”€â”€ Trade execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        print(f"  Size: {actual_size} (base {base_size} Ã— {signal.risk_multiplier}) | R:R 1:{validation['risk_reward']}")
        self.risk_manager.open_positions += 1

    def _reject_trade(self, signal: TradeSignal):
        print(f"[REJECT] {signal.symbol} â€” rechazado manualmente")

    # â”€â”€ Main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def run(self):
        self._running = True
        print("=" * 55)
        print("  SMC TRADING BOT â€” Claude AI + Glint + DecisionFilter")
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

        # MT5 startup check — try port 443 first for ISP compatibility
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
        )

    @staticmethod
    async def _check_internet() -> bool:
        """Fast TCP probe to 8.8.8.8:53 â€” no external libraries needed."""
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

    # â”€â”€ Technical SMC analysis (no API call) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _run_smc_lite(self, df: pd.DataFrame) -> dict:
        """
        Lightweight SMC analysis without Claude API â€” runs in the scan loop.
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

    async def _scan_symbol(self, symbol: str, timeframe: str) -> Optional[TradeSignal]:
        """
        Full pipeline for one symbol/timeframe:
        fetch â†’ SMC lite â†’ SignalAgent â†’ DecisionFilter â†’ return signal or None
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
        return signal

    # â”€â”€ Demo trade execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


    async def _scan_mt5_symbol(self, symbol: str, timeframe: str):
        """Fetch MT5 OHLCV, run SMC lite, return signal or None."""
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, lambda: self.mt5.get_ohlcv(symbol, timeframe, 200))
        if df is None or df.empty or len(df) < 50:
            return None
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

    async def _send_mt5_real_order(self, signal: TradeSignal):
        """Send a real order to MT5 demo — max 1 open position per symbol."""
        order_type = "BUY" if signal.signal_type == SignalType.LONG else "SELL"
        sl_val = signal.stop_loss if signal.stop_loss else 0.0
        tp_val = signal.take_profit if signal.take_profit else 0.0

        # Require valid SL — a trade without a stop loss has no defined exit
        if sl_val == 0.0:
            print(f"[MT5] {signal.symbol}: SL no definido, skip", flush=True)
            return

        # Guard: max 1 open position per symbol
        loop = asyncio.get_running_loop()
        existing = await loop.run_in_executor(None, self.mt5.get_positions)
        sym_open = [p for p in existing if p["symbol"] == signal.symbol]
        if sym_open:
            pos = sym_open[0]
            pnl_live = pos.get("profit", 0.0)
            print(
                f"[MT5] {signal.symbol}: posicion {pos['type']} ya abierta "
                f"({pnl_live:+.2f} USD) — skip nueva orden",
                flush=True,
            )
            return

        print(f"[MT5 ORDER] Enviando {signal.symbol} {order_type} sl={sl_val:.5f} tp={tp_val:.5f}", flush=True)
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.mt5.place_order(signal.symbol, order_type, 0.01, sl=sl_val, tp=tp_val),
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
        """Every 60s: log open MT5 positions with live P&L."""
        while self._running:
            await asyncio.sleep(60)
            if not self._mt5_available:
                continue
            try:
                loop = asyncio.get_running_loop()
                positions = await loop.run_in_executor(None, self.mt5.get_positions)
                if positions:
                    total_pnl = sum(p.get("profit", 0.0) for p in positions)
                    lines = [f"[POS] {len(positions)} posiciones abiertas | P&L vivo: {total_pnl:+.2f} USD"]
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

    async def _market_scan_loop(self):
        _was_offline = False
        threshold = DEMO_SCORE_THRESHOLD if self.demo_mode else 60

        while self._running:
            try:
                online = await self._check_internet()

                if not online:
                    if not _was_offline:
                        _was_offline = True
                    print("[Bot] Sin internet â€” reintentando en 30s...")
                    await asyncio.sleep(30)
                    continue

                if _was_offline:
                    _was_offline = False
                    try:
                        await self.telegram.send_glint_alert("ðŸ”„ Conexion restaurada â€” bot activo")
                    except Exception:
                        pass

                # â”€â”€ Full market scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                            print(f"[{symbol}][{tf}] Score: {score} | {bias}", end="")

                            if signal.signal_type == SignalType.WAIT or score < threshold:
                                print(" â€” sin setup")
                            elif self.demo_mode:
                                print(f" â€” ejecutando trade DEMO")
                                await self._execute_demo_trade(signal)
                            else:
                                print(f" â€” ejecutando trade")
                                self._dispatch(signal)

                        except Exception as exc:
                            print(f"[{symbol}][{tf}] Error: {exc.__class__.__name__}: {exc}")

                        await asyncio.sleep(1)  # rate limit between symbols

                # MT5 forex scan (real orders on demo account — bypass demo slot limit)
                if self._mt5_available:
                    for symbol in MT5_SYMBOLS:
                        for tf in MT5_TIMEFRAMES:
                            if not self._running:
                                break
                            try:
                                signal = await self._scan_mt5_symbol(symbol, tf)
                                if signal is None:
                                    continue
                                score = signal.decision_score
                                bias  = signal.signal_type.value.upper()
                                print(f"[MT5][{symbol}][{tf}] Score: {score} | {bias}", end="")
                                if signal.signal_type == SignalType.WAIT or score < threshold:
                                    print(" -- sin setup")
                                else:
                                    print(f" -- ejecutando MT5 REAL")
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
                            if signal.signal_type == SignalType.WAIT or score < threshold:
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

            # MT5 reconnect monitor — check every scan cycle
            if self._mt5_available and not self.mt5.is_connected():
                self._mt5_available = False
                print("[MT5] Desconectado — reintentando en 60s...")
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

            await asyncio.sleep(SCAN_INTERVAL_SEC)  # next full scan

    def stop(self):
        self._running = False

