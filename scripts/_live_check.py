import MetaTrader5 as mt5
import os
from dotenv import load_dotenv
load_dotenv()

mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

positions = mt5.positions_get()
account = mt5.account_info()
bal = account.balance
eq = account.equity
fp = account.profit
print(f"Balance: ${bal:.2f} | Equity: ${eq:.2f} | FloatPnL: ${fp:.2f}")
print()
if positions:
    for p in positions:
        side = 'BUY' if p.type == 0 else 'SELL'
        diff = p.price_current - p.price_open if p.type == 0 else p.price_open - p.price_current
        print(f"#{p.ticket} {p.symbol} {side} {p.volume}L | entry={p.price_open:.2f} | now={p.price_current:.2f} | pts={diff:.1f} | PnL=${p.profit:.2f} | SL={p.sl:.2f} | TP={p.tp:.2f}")
else:
    print("No open positions")
mt5.shutdown()
