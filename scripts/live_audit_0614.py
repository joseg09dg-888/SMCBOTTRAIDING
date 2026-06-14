"""Live audit 2026-06-14: current balance + trades since last audit (06-11)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

login = int(os.getenv("MT5_LOGIN"))
password = os.getenv("MT5_PASSWORD")
server = os.getenv("MT5_SERVER")

if not mt5.initialize():
    print("initialize() failed:", mt5.last_error())
    sys.exit(1)

if not mt5.login(login, password=password, server=server):
    print("login failed:", mt5.last_error())
    sys.exit(1)

acc = mt5.account_info()
print("=" * 70)
print(f"BALANCE ACTUAL: ${acc.balance:,.2f} | EQUITY: ${acc.equity:,.2f}")
print(f"Drawdown desde $100,000: ${acc.balance-100000:,.2f} ({(acc.balance-100000)/100000*100:.2f}%)")
print("=" * 70)

positions = mt5.positions_get()
print(f"\nPOSICIONES ABIERTAS: {len(positions) if positions else 0}")
if positions:
    for p in positions:
        print(f"  #{p.ticket} {p.symbol} vol={p.volume} entry={p.price_open} "
              f"sl={p.sl} tp={p.tp} profit=${p.profit:.2f}")

# Deals since last audit (06-11 00:00 UTC)
since = datetime(2026, 6, 11, 0, 0)
until = datetime.now() + timedelta(days=1)
deals = mt5.history_deals_get(since, until)
print(f"\nDEALS desde {since} hasta ahora: {len(deals) if deals else 0}")

if deals:
    by_pos = {}
    for d in deals:
        if d.entry == 1:  # OUT (close)
            by_pos.setdefault(d.position_id, []).append(d)

    total_pnl = 0.0
    by_symbol = {}
    by_reason = {}
    rows = []
    for pid, ds in by_pos.items():
        for d in ds:
            total_pnl += d.profit + d.swap + d.commission
            sym = d.symbol
            by_symbol.setdefault(sym, {"n": 0, "pnl": 0.0, "wins": 0})
            by_symbol[sym]["n"] += 1
            by_symbol[sym]["pnl"] += d.profit + d.swap + d.commission
            if d.profit > 0:
                by_symbol[sym]["wins"] += 1
            by_reason.setdefault(d.reason, {"n": 0, "pnl": 0.0})
            by_reason[d.reason]["n"] += 1
            by_reason[d.reason]["pnl"] += d.profit + d.swap + d.commission
            rows.append((datetime.fromtimestamp(d.time), sym, d.position_id, d.volume,
                          d.price, d.profit, d.reason))

    print(f"\nTOTAL P&L desde 06-11: ${total_pnl:,.2f}")
    print(f"Total deals de cierre: {sum(v['n'] for v in by_symbol.values())}")

    print("\n--- POR SIMBOLO ---")
    for sym, v in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"]):
        wr = v["wins"] / v["n"] * 100 if v["n"] else 0
        print(f"  {sym:10s} n={v['n']:4d} WR={wr:5.1f}% pnl=${v['pnl']:+10.2f}")

    reason_names = {0: "CLIENT", 3: "EXPERT(bot)", 4: "SL", 5: "TP", 6: "SO", 1: "MOBILE", 2: "DEALER"}
    print("\n--- POR CLOSE REASON ---")
    for r, v in sorted(by_reason.items(), key=lambda x: x[1]["pnl"]):
        avg = v["pnl"]/v["n"] if v["n"] else 0
        print(f"  {reason_names.get(r, r):12s} n={v['n']:4d} pnl=${v['pnl']:+10.2f} avg=${avg:+.2f}")

    print("\n--- TODAS LAS OPERACIONES (orden cronologico) ---")
    for t, sym, pid, vol, price, profit, reason in sorted(rows, key=lambda x: x[0]):
        print(f"  {t} {sym:8s} #{pid} vol={vol} price={price} pnl=${profit:+8.2f} reason={reason_names.get(reason,reason)}")

mt5.shutdown()
