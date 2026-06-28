import MetaTrader5 as mt5, os, sys, time
sys.path.insert(0, '.')
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()
mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

print("MONITOR EN VIVO -- actualizando cada 30s (Ctrl+C para detener)\n")
last_pnls = {}

while True:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    acc = mt5.account_info()
    positions = mt5.positions_get()

    print(f"[{now}] Equity=${acc.equity:.2f} | Float=${acc.profit:+.2f}")
    if positions:
        for p in positions:
            side = 'BUY' if p.type == 0 else 'SELL'
            flag = ""
            prev = last_pnls.get(p.ticket)
            if prev is not None:
                delta = p.profit - prev
                flag = f" ({delta:+.2f})" if abs(delta) > 0.05 else ""
            alert = ""
            if p.symbol == "XAUUSD":
                dist_sl = abs(p.price_current - p.sl)
                pct_sl = dist_sl / abs(p.price_open - p.sl) * 100
                if pct_sl < 30:
                    alert = " <<< CERCA DEL SL!"
                elif p.profit > 0:
                    alert = " (ganando)"
            status = "GANANDO" if p.profit >= 0 else "PERDIENDO"
            print(f"  #{p.ticket} {p.symbol} {side} | {p.price_current:.2f} | ${p.profit:+.2f}{flag} {status}{alert} | SL={p.sl:.2f}")
            last_pnls[p.ticket] = p.profit
    else:
        print("  Sin posiciones abiertas")
    print()
    time.sleep(30)
