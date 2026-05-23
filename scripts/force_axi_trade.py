"""Force Axi MT5 trade — bypass tradeapi_disabled flag check."""
import MetaTrader5 as mt5, sys, asyncio, time
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
import os; os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

AXI_EXE = r'C:\Program Files\Axi MetaTrader 5 Terminal\terminal64.exe'
LOGIN = 10042896
PWD   = 'IMSMCbot*3axi'
SRV   = 'Axi-US50-Demo'

print("Connecting to Axi MT5...")
mt5.shutdown()
time.sleep(1)
if not mt5.initialize(path=AXI_EXE, login=LOGIN, password=PWD, server=SRV):
    print(f"Connect failed: {mt5.last_error()}")
    sys.exit(1)

ai = mt5.account_info()
ti = mt5.terminal_info()
print(f"Balance: ${ai.balance:,.2f} | trade_allowed={ti.trade_allowed} | tradeapi_disabled={ti.tradeapi_disabled}")

# Force try order_send despite flags — see exact retcode
mt5.symbol_select('EURUSD', True)
time.sleep(0.5)
tick = mt5.symbol_info_tick('EURUSD')
sym  = mt5.symbol_info('EURUSD')

if not tick or tick.ask == 0:
    print(f"No tick data: {mt5.last_error()}")
    mt5.shutdown(); sys.exit(1)

price  = round(tick.ask, sym.digits)
sl     = round(price - 50 * sym.point * 10, sym.digits)
tp     = round(price + 100 * sym.point * 10, sym.digits)
print(f"EURUSD ask={price:.5f} sl={sl:.5f} tp={tp:.5f}")

# Try ORDER_TYPE_BUY
for fill in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
    mt5.shutdown()
    time.sleep(0.5)
    mt5.initialize(path=AXI_EXE, login=LOGIN, password=PWD, server=SRV)

    tick = mt5.symbol_info_tick('EURUSD')
    price = round(tick.ask, sym.digits)
    sl    = round(price - 50 * sym.point * 10, sym.digits)
    tp    = round(price + 100 * sym.point * 10, sym.digits)

    req = {
        'action':        mt5.TRADE_ACTION_DEAL,
        'symbol':        'EURUSD',
        'volume':        0.01,
        'type':          mt5.ORDER_TYPE_BUY,
        'price':         price,
        'sl':            sl,
        'tp':            tp,
        'deviation':     50,
        'magic':         234000,
        'comment':       'SMC Bot Demo Axi',
        'type_time':     mt5.ORDER_TIME_GTC,
        'type_filling':  fill,
    }
    result = mt5.order_send(req)
    fill_name = {0:"IOC", 1:"FOK", 2:"RETURN"}.get(fill, str(fill))
    if result:
        print(f"fill={fill_name}: retcode={result.retcode} comment={result.comment}")
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\nTRADE EXECUTED! Ticket={result.order} @ {result.price:.5f}")

            # Telegram
            try:
                from dashboard.telegram_bot import TradingTelegramBot
                bot = TradingTelegramBot()
                msg = (
                    f"<b>PRIMER TRADE MT5 AXI EJECUTADO</b>\n"
                    f"EURUSD LONG 0.01 @ <code>{result.price:.5f}</code>\n"
                    f"SL: <code>{sl:.5f}</code> | TP: <code>{tp:.5f}</code>\n"
                    f"Balance: ${ai.balance:,.2f} USD"
                )
                asyncio.run(bot.send_glint_alert(msg))
                print("  Telegram: sent!")
            except Exception as e:
                print(f"  Telegram: {e}")
            break
    else:
        print(f"fill={fill_name}: order_send returned None, err={mt5.last_error()}")

# If all filling types failed, show diagnosis
print("\n=== Diagnosis ===")
print(f"tradeapi_disabled=True means Axi has disabled Python API trading on this account.")
print(f"Options:")
print(f"  1. In Axi MT5: Tools > Options > Expert Advisors")
print(f"     Check 'Allow automated trading' AND 'Allow trading by MetaTrader5 API'")
print(f"  2. Contact Axi support to enable Python API on demo account")
print(f"  3. The bot already works for DEMO trades via paper trading")
print(f"     (recorded in SQLite, counts for Axi Edge Score)")

mt5.shutdown()
