import asyncio
from typing import Optional, Callable
from core.config import config
from agents.signal_agent import TradeSignal, SignalType


class TradingTelegramBot:
    """
    AUTO:   Sends notification AFTER executing.
    SEMI:   Sends signal + Approve/Reject buttons before executing.
    ALERTS: Notifications only, never executes.
    """

    def __init__(
        self,
        on_approve: Optional[Callable] = None,
        on_reject: Optional[Callable] = None,
    ):
        self.on_approve = on_approve
        self.on_reject  = on_reject
        self._pending: dict = {}
        self._bot = None

    def _get_bot(self):
        if self._bot is None:
            try:
                from telegram import Bot
                self._bot = Bot(token=config.telegram_bot_token)
            except Exception:
                self._bot = None
        return self._bot

    async def send_signal(self, signal: TradeSignal, mode: str = "auto"):
        bot = self._get_bot()
        if not bot or not config.telegram_chat_id:
            print(f"[Telegram] {signal.format_telegram()}")
            return

        text = signal.format_telegram()
        if mode == "semi":
            try:
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                keyboard = InlineKeyboardMarkup([[
                    InlineKeyboardButton("âœ… EJECUTAR", callback_data=f"approve:{id(signal)}"),
                    InlineKeyboardButton("âŒ RECHAZAR", callback_data=f"reject:{id(signal)}"),
                ]])
                self._pending[str(id(signal))] = signal
                await bot.send_message(
                    chat_id=config.telegram_chat_id,
                    text=f"â³ *SEÃ‘AL PENDIENTE DE APROBACIÃ“N*\n\n{text}",
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
            except Exception as e:
                print(f"[Telegram] semi-auto send failed: {e}")
        else:
            try:
                await bot.send_message(
                    chat_id=config.telegram_chat_id,
                    text=text,
                    parse_mode="HTML",
                )
            except Exception as e:
                print(f"[Telegram] send failed: {e}")

    async def send_glint_alert(self, signal_text: str):
        bot = self._get_bot()
        if not bot or not config.telegram_chat_id:
            print(f"[Telegram/Glint] {signal_text[:100]}")
            return
        try:
            await bot.send_message(
                chat_id=config.telegram_chat_id,
                text=signal_text,
                parse_mode="HTML",
            )
        except Exception as e:
            print(f"[Telegram] glint alert failed: {e}")

    async def send_trade_result(self, symbol: str, pnl: float, direction: str):
        emoji = "ðŸŸ¢" if pnl > 0 else "ðŸ”´"
        msg = (
            f"{emoji} *TRADE CERRADO â€” {symbol}*\n"
            f"DirecciÃ³n: {direction}\n"
            f"P&L: `{'+'if pnl>0 else ''}{pnl:.2f} USD`"
        )
        bot = self._get_bot()
        if not bot or not config.telegram_chat_id:
            print(f"[Telegram] {msg}")
            return
        try:
            await bot.send_message(
                chat_id=config.telegram_chat_id,
                text=msg,
                parse_mode="HTML",
            )
        except Exception as e:
            print(f"[Telegram] trade result failed: {e}")

    async def send_risk_alert(self, reason: str):
        msg = f"ðŸš¨ *ALERTA DE RIESGO*\n{reason}\n\nâ›” Bot pausado automÃ¡ticamente."
        bot = self._get_bot()
        if not bot or not config.telegram_chat_id:
            print(f"[Telegram/Risk] {msg}")
            return
        try:
            await bot.send_message(
                chat_id=config.telegram_chat_id,
                text=msg,
                parse_mode="HTML",
            )
        except Exception as e:
            print(f"[Telegram] risk alert failed: {e}")

    async def handle_callback(self, update, context):
        query = update.callback_query
        await query.answer()
        action, signal_id = query.data.split(":")
        signal = self._pending.pop(signal_id, None)
        if action == "approve" and signal and self.on_approve:
            await query.edit_message_text(
                f"âœ… *APROBADO â€” Ejecutando...*\n\n{signal.format_telegram()}",
                parse_mode="HTML",
            )
            self.on_approve(signal)
        elif action == "reject":
            await query.edit_message_text("âŒ *SeÃ±al rechazada. No se opera.*", parse_mode="HTML")
            if signal and self.on_reject:
                self.on_reject(signal)

