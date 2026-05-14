import asyncio
from typing import Optional, List, Dict
from core.config import config
from core.risk_manager import RiskManager
from core.decision_filter import DecisionFilter, TradeGrade
from agents.signal_agent import TradeSignal, SignalType
from connectors.glint_connector import GlintSignal
from connectors.glint_browser import GlintBrowser
from dashboard.telegram_bot import TradingTelegramBot
from dashboard.telegram_commander import TelegramCommander
from training.historical_agent import HistoricalDataAgent


class TradingSupervisor:
    """
    Master orchestrator. Full pipeline:

    Glint signal → SMC analysis → ML + Sentiment → DecisionFilter (0-100)
        → < 60  : NO TRADE (log reason)
        → 60-74 : REDUCED (25% risk)
        → 75-89 : FULL (100% risk)
        → 90+   : PREMIUM (100% risk + Telegram alert 🔥)
    """

    def __init__(self, capital: float = 1000.0):
        self.config         = config
        self.capital        = capital
        self.risk_manager   = RiskManager(config, capital)
        self.historical     = HistoricalDataAgent()
        self.decision       = DecisionFilter(config, self.risk_manager,
                                             historical_agent=self.historical)
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
        self._glint_buffer: List[Dict] = []
        self._last_glint_text: str = ""
        self.mode    = config.operation_mode
        self._running = False

    # ── Callbacks from TelegramCommander ─────────────────────────────────

    def _on_mode_change(self, mode: str):
        self.mode = mode
        print(f"[Mode] Cambiado a: {mode.upper()} vía Telegram")

    def _on_history_command(self, symbol: str) -> str:
        return self.historical.get_market_summary(symbol or "BTC")

    # ── Glint callback ────────────────────────────────────────────────────

    def _on_glint_signal(self, glint: GlintSignal):
        if glint.is_actionable():
            self._last_glint_text = glint.text
            self._glint_buffer.append(glint.raw)
            if len(self._glint_buffer) > 20:
                self._glint_buffer.pop(0)
            asyncio.create_task(self.telegram.send_glint_alert(glint.format_alert()))
            print(f"[Glint] {glint.impact}: {glint.text[:80]}...")

    # ── Decision pipeline ─────────────────────────────────────────────────

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
            print(f"[FILTER] NO TRADE — {signal.notes}")
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

    # ── Trade execution ───────────────────────────────────────────────────

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
        print(f"  Size: {actual_size} (base {base_size} × {signal.risk_multiplier}) | R:R 1:{validation['risk_reward']}")
        self.risk_manager.open_positions += 1

    def _reject_trade(self, signal: TradeSignal):
        print(f"[REJECT] {signal.symbol} — rechazado manualmente")

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self):
        self._running = True
        print("=" * 55)
        print("  SMC TRADING BOT — Claude AI + Glint + DecisionFilter")
        print("=" * 55)
        print(f"  Modo:          {self.mode.upper()}")
        print(f"  Capital:       ${self.capital:,.2f}")
        print(f"  Riesgo max:    {self.config.max_risk_per_trade*100}% por trade")
        print(f"  Timeframes:    {', '.join(self.config.timeframes)}")
        print(f"  Score gates:   <60=NO | 60-74=25% | 75-89=100% | 90+=PREMIUM")
        print()

        await asyncio.gather(
            self.commander.start_polling(),
            self.glint.connect(),
            self._market_scan_loop(),
        )

    async def _market_scan_loop(self):
        while self._running:
            try:
                print("[Scan] Escaneando mercados...")
                await asyncio.sleep(60)
            except Exception as e:
                print(f"[Scan] Error: {e}")
                await asyncio.sleep(10)

    def stop(self):
        self._running = False
