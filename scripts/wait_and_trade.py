"""Wait for AutoTrading button, then execute first Axi demo trade."""
import MetaTrader5 as mt5, time, sys, asyncio
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
import os; os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

LOGIN = 10042896
PWD   = 'IMSMCbot*3axi'
SRV   = 'Axi-US50-Demo'

print("Conectando a Axi MT5...")
if not mt5.initialize(login=LOGIN, password=PWD, server=SRV):
    print(f"ERROR: {mt5.last_error()}")
    sys.exit(1)

ti = mt5.terminal_info()
print(f"trade_allowed: {ti.trade_allowed}")

if not ti.trade_allowed:
    print()
    print("=" * 50)
    print("ACCION REQUERIDA EN AXI MT5:")
    print("  Busca el boton 'AutoTrading' en el toolbar")
    print("  (boton con flecha verde en la barra superior)")
    print("  Haz click para activarlo — se pone VERDE")
    print("=" * 50)
    print()
    print("Esperando que actives AutoTrading...", end="", flush=True)

    for i in range(120):
        time.sleep(2)
        try: mt5.shutdown()
        except: pass
        mt5.initialize(login=LOGIN, password=PWD, server=SRV)
        ti = mt5.terminal_info()
        if ti and ti.trade_allowed:
            print(f" ACTIVADO! (t={i*2}s)")
            break
        if i % 10 == 9:
            print(f".", end="", flush=True)
    else:
        print("\nTimeout. Ejecuta el script de nuevo despues de activar AutoTrading.")
        mt5.shutdown()
        sys.exit(1)

# Execute first demo trade
print("\nEjecutando primer trade demo MT5 Axi...")
mt5.symbol_select('EURUSD', True)
time.sleep(1)
tick = mt5.symbol_info_tick('EURUSD')
price = tick.ask
sl = round(price - 0.0050, 5)
tp = round(price + 0.0100, 5)

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
res = mt5.order_send(request)

if res and res.retcode == mt5.TRADE_RETCODE_DONE:
    print(f"PRIMER TRADE MT5 AXI EJECUTADO!")
    print(f"  Ticket: {res.order}")
    print(f"  EURUSD LONG 0.01 @ {price:.5f}")
    print(f"  SL: {sl:.5f} | TP: {tp:.5f}")
    # Also add GBPUSD
    mt5.symbol_select('GBPUSD', True)
    tick2 = mt5.symbol_info_tick('GBPUSD')
    if tick2 and tick2.ask > 0:
        req2 = dict(request, symbol='GBPUSD', price=tick2.ask,
                    sl=round(tick2.ask-0.0050,5), tp=round(tick2.ask+0.0100,5))
        res2 = mt5.order_send(req2)
        if res2 and res2.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  GBPUSD LONG 0.01 @ {tick2.ask:.5f} (ticket {res2.order})")
else:
    code = res.retcode if res else "None"
    comment = res.comment if res else "None"
    print(f"Trade failed: {code} {comment}")

# Telegram notification
try:
    from dashboard.telegram_bot import TradingTelegramBot
    bot = TradingTelegramBot()
    ai = mt5.account_info()
    msg = (
        f"<b>MT5 AXI DEMO CONECTADO</b>\n"
        f"Balance: ${ai.balance:,.2f} {ai.currency}\n"
        f"Server: {ai.server}\n"
        f"EURUSD/GBPUSD/USDJPY activos\n"
        f"Bot 24/7 activo — aprendiendo y operando"
    )
    asyncio.run(bot.send_glint_alert(msg))
    print("Telegram: sent!")
except Exception as e:
    print(f"Telegram: {e}")

mt5.shutdown()
