"""Force close a specific ticket using direct MT5 order."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MetaTrader5 as mt5
from core.config import Config

cfg = Config()
ok = mt5.initialize()
if not ok:
    print(f"ERROR init: {mt5.last_error()}")
    sys.exit(1)

ok = mt5.login(cfg.mt5_login, password=cfg.mt5_password, server=cfg.mt5_server)
if not ok:
    print(f"ERROR login: {mt5.last_error()}")
    sys.exit(1)

positions = mt5.positions_get()
if not positions:
    print("Sin posiciones abiertas.")
    mt5.shutdown()
    sys.exit(0)

print(f"Posiciones abiertas: {len(positions)}")
for p in positions:
    print(f"  {p.symbol} #{p.ticket} vol={p.volume} profit=${p.profit:.2f}")

for p in positions:
    sym = p.symbol
    ticket = p.ticket
    vol = p.volume
    direction = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(sym).bid if direction == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(sym).ask

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": vol,
        "type": direction,
        "position": ticket,
        "price": price,
        "deviation": 50,
        "magic": 234000,
        "comment": "force close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"CERRADA {sym} #{ticket} profit=${p.profit:.2f}")
    else:
        rc = result.retcode if result else "None"
        print(f"ERROR {sym} #{ticket} retcode={rc}")

mt5.shutdown()
