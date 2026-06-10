"""Detailed audit: before vs after improvements."""
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

mt5.initialize()

print("=" * 55)
print("AUDITORIA DETALLADA: ANTES vs DESPUES DE MEJORAS")
print("=" * 55)

cutoff = datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc)
from_date = datetime(2026, 5, 1, tzinfo=timezone.utc)
to_date   = datetime.now(timezone.utc)

deals = mt5.history_deals_get(from_date, to_date)
closing = [d for d in (deals or []) if d.profit != 0 and d.entry == 1]

before = [d for d in closing if datetime.fromtimestamp(d.time, tz=timezone.utc) < cutoff]
after  = [d for d in closing if datetime.fromtimestamp(d.time, tz=timezone.utc) >= cutoff]
last3  = [d for d in closing if datetime.fromtimestamp(d.time, tz=timezone.utc) > to_date - timedelta(days=3)]

def stats(deals_list, label):
    if not deals_list:
        print(f"\n{label}: Sin datos")
        return 0, 0, 0
    w  = sum(1 for d in deals_list if d.profit > 0)
    l  = sum(1 for d in deals_list if d.profit <= 0)
    gp = sum(d.profit for d in deals_list if d.profit > 0)
    lp = abs(sum(d.profit for d in deals_list if d.profit <= 0))
    wr = w/(w+l)*100 if (w+l) > 0 else 0
    pf = gp/lp if lp > 0 else 0
    net = gp - lp
    ok_wr = "PASS" if wr >= 60 else "FAIL(necesita 60%+)"
    ok_pf = "PASS" if pf >= 1.5 else "FAIL(necesita 1.5+)"
    avg_w = gp/w if w > 0 else 0
    avg_l = lp/l if l > 0 else 0
    print(f"\n{label}:")
    print(f"  Trades: {w+l}  |  Wins: {w}  |  Losses: {l}")
    print(f"  Win Rate:      {wr:.1f}%  [{ok_wr}]")
    print(f"  Profit Factor: {pf:.2f}   [{ok_pf}]")
    print(f"  Neto:          ${net:+.2f}")
    print(f"  Ganancia media/trade ganado: ${avg_w:.2f}")
    print(f"  Perdida media/trade perdido: ${avg_l:.2f}")
    return wr, pf, net

wr_b, pf_b, net_b = stats(before, "ANTES: mayo-2jun (sin H4 filter, sin ATR SL)")
wr_a, pf_a, net_a = stats(after,  "DESPUES: 3jun+ (CON H4+H1 dual confirm, ATR SL, partial close)")
if last3:
    wr_3, pf_3, net_3 = stats(last3, "ULTIMOS 3 DIAS")

print("\n--- POSICIONES ABIERTAS AHORA ---")
pos = mt5.positions_get()
if pos:
    floating = sum(p.profit for p in pos)
    print(f"Total flotante: ${floating:+.2f}")
    for p in pos:
        d = "BUY" if p.type == 0 else "SELL"
        s = "GANANDO" if p.profit >= 0 else "PERDIENDO"
        print(f"  {p.symbol} {d} {p.volume}L  P&L=${p.profit:+.2f} ({s})")
        print(f"    Entry={p.price_open:.4f}  SL={p.sl:.4f}  TP={p.tp:.4f}")
else:
    print("  Sin posiciones")

print("\n--- CUENTA MT5 ---")
info = mt5.account_info()
if info:
    net = info.balance - 100_000
    dd  = abs(min(net, 0)) / 100_000 * 100
    print(f"  Balance:  ${info.balance:,.2f}")
    print(f"  Equity:   ${info.equity:,.2f}")
    print(f"  Neto:     ${net:+,.2f} ({net/100_000*100:+.3f}%)")
    print(f"  Drawdown: {dd:.3f}% de $100K")

print("\n--- VEREDICTO ---")
if wr_a >= 60 and pf_a >= 1.5:
    print("MEJORAS FUNCIONANDO: WR y PF en rango Axi Select")
elif wr_a > wr_b:
    print(f"MEJORANDO: WR subio {wr_b:.0f}% -> {wr_a:.0f}% con los nuevos filtros")
    print(f"Necesita mas tiempo para llegar a 60%+")
else:
    print(f"PROBLEMA: WR={wr_a:.0f}% (antes={wr_b:.0f}%). Ajustar estrategia.")
    if wr_b < 30:
        print("El WR historico bajo indica señales de baja calidad antes de las mejoras")

mt5.shutdown()
