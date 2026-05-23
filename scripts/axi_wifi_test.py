"""Test Axi MT5 on normal WiFi + execute first demo trade."""
import MetaTrader5 as mt5, sys, asyncio
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
import os; os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN = 10042896
PWD   = 'IMSMCbot*3axi'
SRV   = 'Axi-US50-Demo'

print(f"Testing Axi MT5 on current network...")
result = mt5.initialize(login=LOGIN, password=PWD, server=SRV)
print(f"Conectado: {result}")
if not result:
    print(f"Error: {mt5.last_error()}")
    sys.exit(1)

info = mt5.account_info()
print(f"Balance: ${info.balance:,.2f} {info.currency}")
print(f"Server:  {info.server}")
print(f"Type:    {'DEMO' if info.trade_mode == 0 else 'REAL'}")
print()

# Activar y mostrar precios
print("Live prices:")
for sym in ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY']:
    mt5.symbol_select(sym, True)
    tick = mt5.symbol_info_tick(sym)
    if tick:
        print(f"  {sym}: bid={tick.bid:.5f} ask={tick.ask:.5f}")
    else:
        print(f"  {sym}: no tick data")

# Execute first demo trade — EURUSD LONG 0.01
print("\nExecuting first MT5 Axi demo trade...")
mt5.symbol_select('EURUSD', True)
tick = mt5.symbol_info_tick('EURUSD')
price = tick.ask
sl = round(price - 0.0050, 5)
tp = round(price + 0.0100, 5)

request = {
    'action':   mt5.TRADE_ACTION_DEAL,
    'symbol':   'EURUSD',
    'volume':   0.01,
    'type':     mt5.ORDER_TYPE_BUY,
    'price':    price,
    'sl':       sl,
    'tp':       tp,
    'deviation': 20,
    'magic':    234000,
    'comment':  'SMC Bot Demo Axi',
    'type_time':    mt5.ORDER_TIME_GTC,
    'type_filling': mt5.ORDER_FILLING_IOC,
}
trade_result = mt5.order_send(request)

if trade_result and trade_result.retcode == mt5.TRADE_RETCODE_DONE:
    print(f"PRIMER TRADE MT5 AXI EJECUTADO!")
    print(f"  Ticket: {trade_result.order}")
    print(f"  EURUSD LONG 0.01 @ {price:.5f}")
    print(f"  SL: {sl:.5f} | TP: {tp:.5f}")
else:
    retcode = trade_result.retcode if trade_result else "None"
    comment = trade_result.comment if trade_result else "None"
    print(f"Trade result: retcode={retcode} comment={comment}")

# Send Telegram notification
try:
    from dashboard.telegram_bot import TradingTelegramBot
    bot = TradingTelegramBot()
    msg = (
        f"<b>MT5 AXI DEMO CONECTADO CON WIFI</b>\n"
        f"Balance: ${info.balance:,.2f} {info.currency}\n"
        f"Server: {info.server}\n"
        f"EURUSD/GBPUSD/XAUUSD/USDJPY operando\n"
        f"Bot 24/7 activo — aprendiendo y operando"
    )
    asyncio.run(bot.send_glint_alert(msg))
    print("Telegram: sent!")
except Exception as e:
    print(f"Telegram: {e}")

mt5.shutdown()
print("\nMT5 AXI FUNCIONANDO CON WIFI NORMAL")
