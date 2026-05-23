"""Execute first Axi MT5 demo trade.
Reconnects to terminal after user clicks AutoTrading button to pick up the new state.
"""
import MetaTrader5 as mt5, sys, asyncio, time, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN = 10042896
PWD   = 'IMSMCbot*3axi'
SRV   = 'Axi-US50-Demo'

def connect():
    try: mt5.shutdown()
    except: pass
    time.sleep(0.5)
    return mt5.initialize(login=LOGIN, password=PWD, server=SRV)

def check_state():
    ti = mt5.terminal_info()
    ai = mt5.account_info()
    if ti and ai:
        return ti.trade_allowed, ti.tradeapi_disabled, ai.balance, ai.currency
    return False, True, 0, ""

# Initial connect
print("Connecting to Axi MT5...")
if not connect():
    print(f"Connect failed: {mt5.last_error()}")
    sys.exit(1)

ai = mt5.account_info()
print(f"Connected: {ai.login} | Balance: ${ai.balance:,.2f} {ai.currency}")

# Poll for AutoTrading — reconnect each time to get fresh state
print()
print("=" * 50)
print("ACCION REQUERIDA EN AXI MT5:")
print("  1. Busca boton 'Trading algoritmico' en toolbar")
print("  2. Debe estar VERDE (activo)")
print("  3. Si esta rojo/gris: haz click para activarlo")
print("=" * 50)
print()

for attempt in range(30):
    # Fresh reconnect every attempt to read current terminal state
    if attempt > 0:
        connect()

    allowed, disabled, balance, currency = check_state()
    print(f"  [{attempt+1}/30] trade_allowed={allowed} tradeapi_disabled={disabled}", end="", flush=True)

    if allowed and not disabled:
        print(" -> READY!")
        break

    print(f" -> waiting 3s...")
    mt5.shutdown()
    time.sleep(3)
else:
    print("\nTimeout. Verify AutoTrading button is GREEN in Axi MT5 toolbar.")
    sys.exit(1)

# Execute EURUSD LONG
print("\nExecuting EURUSD LONG demo trade...")
mt5.symbol_select('EURUSD', True)
time.sleep(0.5)

tick = mt5.symbol_info_tick('EURUSD')
sym  = mt5.symbol_info('EURUSD')
if not tick or not sym:
    print(f"Symbol info failed: {mt5.last_error()}")
    mt5.shutdown(); sys.exit(1)

point  = sym.point
digits = sym.digits
price  = round(tick.ask, digits)
sl     = round(price - 50 * point * 10, digits)   # -50 pips
tp     = round(price + 100 * point * 10, digits)  # +100 pips

print(f"  EURUSD ask={price:.5f} | SL={sl:.5f} (-50 pips) | TP={tp:.5f} (+100 pips)")

request = {
    'action':        mt5.TRADE_ACTION_DEAL,
    'symbol':        'EURUSD',
    'volume':        0.01,
    'type':          mt5.ORDER_TYPE_BUY,
    'price':         price,
    'sl':            sl,
    'tp':            tp,
    'deviation':     20,
    'magic':         234000,
    'comment':       'SMC Bot Demo Axi',
    'type_time':     mt5.ORDER_TIME_GTC,
    'type_filling':  mt5.ORDER_FILLING_IOC,
}
result = mt5.order_send(request)

if result and result.retcode == mt5.TRADE_RETCODE_DONE:
    filled_price = result.price
    print(f"\nPRIMER TRADE MT5 AXI EJECUTADO!")
    print(f"  Ticket:  {result.order}")
    print(f"  EURUSD LONG 0.01 @ {filled_price:.5f}")
    print(f"  SL: {sl:.5f} | TP: {tp:.5f}")
    print(f"  Balance: ${balance:,.2f} {currency}")

    # Telegram
    try:
        from dashboard.telegram_bot import TradingTelegramBot
        bot = TradingTelegramBot()
        msg = (
            f"<b>PRIMER TRADE MT5 AXI EJECUTADO</b>\n"
            f"EURUSD LONG 0.01 @ <code>{filled_price:.5f}</code>\n"
            f"SL: <code>{sl:.5f}</code> | TP: <code>{tp:.5f}</code>\n"
            f"Ticket: {result.order}\n"
            f"Balance: ${balance:,.2f} USD"
        )
        asyncio.run(bot.send_glint_alert(msg))
        print("  Telegram: sent!")
    except Exception as e:
        print(f"  Telegram: {e}")
else:
    code    = result.retcode if result else "None"
    comment = result.comment if result else ""
    errors  = {
        10027: "AutoTrading OFF — click green button in Axi MT5 toolbar",
        10013: "Invalid stops — SL/TP too close",
        10016: "Invalid stops",
        10006: "Order rejected by broker",
        10014: "Invalid volume",
    }
    print(f"\nTrade failed: {code} — {errors.get(code, comment)}")

mt5.shutdown()
