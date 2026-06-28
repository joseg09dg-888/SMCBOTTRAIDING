"""Check SL/TP and actual RR of today's orders."""
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import MetaTrader5 as mt5
from datetime import datetime, timezone

mt5.initialize()
from_date = datetime(2026, 6, 10, 17, 0, tzinfo=timezone.utc)
to_date   = datetime.now(timezone.utc)
orders = mt5.history_orders_get(from_date, to_date)

if orders:
    print("ORDENES HOY — SL/TP/RR REAL:")
    print()
    for o in sorted(orders, key=lambda x: x.time_setup):
        t  = datetime.fromtimestamp(o.time_setup, tz=timezone.utc)
        dr = "BUY" if o.type == 0 else "SELL"
        entry = o.price_open
        sl    = o.sl
        tp    = o.tp
        print(f"  {t.strftime('%H:%M')} {o.symbol} {dr} {o.volume_initial}L")
        print(f"    entry={entry:.4f}  SL={sl:.4f}  TP={tp:.4f}")
        if sl > 0 and tp > 0 and entry > 0:
            sl_d = abs(entry - sl)
            tp_d = abs(tp - entry)
            rr   = tp_d / sl_d if sl_d > 0 else 0
            print(f"    SL_dist={sl_d:.4f}  TP_dist={tp_d:.4f}  RR=1:{rr:.1f}")
        print()
else:
    print("Sin ordenes hoy")

mt5.shutdown()
