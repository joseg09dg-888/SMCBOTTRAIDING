import asyncio
from typing import Optional
from core.config import config
from core.risk_manager import RiskManager
from agents.signal_agent import TradeSignal, SignalType
from connectors.glint_connector import GlintConnector, GlintSignal
from dashboard.telegram_bot import TradingTelegramBot


class TradingSupervisor:
    """
    Master orchestrator. Coordinates the flow:
    Glint signal → SMC Analysis → Signal → Risk check → Execute / Alert
    """

    def __init__(self, capital: float = 1000.0):
        self.config  = config
        self.capital = capital
        self.risk_manager = RiskManager(config, capital)
        self.telegram = TradingTelegramBot(
            on_approve=self._execute_trade,
            on_reject=self._reject_trade,
        )
        self.glint = GlintConnector(
            ws_url=config.glint_ws_url,
            session_token=config.glint_session_token,
            on_signal=self._on_glint_signal,
            min_impact="High",
        )
        self._last_glint_context: str = ""
        self.mode = config.operation_mode
        self._running = False

    def _on_glint_signal(self, signal: GlintSignal):
        if signal.is_actionable():
            self._last_glint_context = signal.text
            asyncio.create_task(self.telegram.send_glint_alert(signal.format_alert()))
            print(f"[Glint] {signal.impact}: {signal.text[:80]}...")

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
        size = self.risk_manager.calculate_position_size(
            entry=signal.entry,
            stop_loss=signal.stop_loss,
        )
        print(f"[EXECUTE] {signal.symbol} {signal.signal_type.value.upper()}")
        print(f"  Entry: {signal.entry} | SL: {signal.stop_loss} | TP: {signal.take_profit}")
        print(f"  Size: {size} | R:R 1:{validation['risk_reward']}")
        self.risk_manager.open_positions += 1

    def _reject_trade(self, signal: TradeSignal):
        print(f"[REJECT] {signal.symbol} — rechazado manualmente")

    async def run(self):
        self._running = True
        print("=" * 50)
        print("  SMC TRADING BOT — Claude AI + Glint")
        print("=" * 50)
        print(f"  Modo: {self.mode.upper()}")
        print(f"  Capital: ${self.capital:,.2f}")
        print(f"  Riesgo max: {self.config.max_risk_per_trade*100}% por trade")
        print(f"  Timeframes: {', '.join(self.config.timeframes)}")
        print()

        await asyncio.gather(
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
