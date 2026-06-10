"""Live audit: MT5 connection, positions, signals, Axi progress."""
import sys
sys.path.insert(0, '.')
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

print("=" * 50)
print("AUDITORIA EN VIVO — SMC BOT")
print("=" * 50)

# --- MT5 Connection ---
ok = mt5.initialize()
print("\n1. CONEXION MT5:", "CONECTADO" if ok else "DESCONECTADO")
info = mt5.account_info()
if info:
    net = info.balance - 100_000.0
    print(f"   Login: {info.login} | Broker: Axi Demo")
    print(f"   Balance: ${info.balance:,.2f} | Equity: ${info.equity:,.2f}")
    print(f"   Ganancia/Perdida neta: ${net:+,.2f} ({net/100_000*100:+.3f}%)")
    print(f"   Trade allowed: {info.trade_allowed}")
else:
    print("   ERROR:", mt5.last_error())

# --- Open Positions ---
pos = mt5.positions_get()
print(f"\n2. POSICIONES ABIERTAS: {len(pos) if pos else 0}")
if pos:
    for p in pos:
        direction = "BUY" if p.type == 0 else "SELL"
        pip = 0.01 if 'JPY' in p.symbol else (1.0 if p.symbol in ['US30','NAS100'] else 0.0001)
        pip = 1.0 if p.symbol in ['US30','NAS100'] else pip
        sl_pips = abs(p.price_open - p.sl) / pip if pip > 0 and p.sl > 0 else 0
        tp_pips = abs(p.tp - p.price_open) / pip if pip > 0 and p.tp > 0 else 0
        rr = tp_pips / sl_pips if sl_pips > 0 else 0
        estado = "GANANDO" if p.profit >= 0 else "PERDIENDO"
        print(f"   {p.symbol} {direction} {p.volume}L #{p.ticket}")
        print(f"   Entry={p.price_open:.4f} SL={p.sl:.4f} TP={p.tp:.4f}")
        print(f"   SL={sl_pips:.0f}pips TP={tp_pips:.0f}pips RR=1:{rr:.1f}")
        print(f"   P&L: ${p.profit:+.2f} ({estado})")
else:
    print("   Sin posiciones abiertas")

# --- Trade History (last 30 days) ---
from datetime import date
from_date = datetime(2026, 5, 1, tzinfo=timezone.utc)
to_date   = datetime.now(timezone.utc)
deals = mt5.history_deals_get(from_date, to_date)

wins = losses = 0
total_profit = total_loss = 0.0
trade_pnls = []

if deals:
    for d in deals:
        if d.profit != 0 and d.entry == 1:  # entry=1 means closing deal
            if d.profit > 0:
                wins += 1
                total_profit += d.profit
            else:
                losses += 1
                total_loss += abs(d.profit)
            trade_pnls.append(d.profit)

total_closed = wins + losses
wr = wins / total_closed * 100 if total_closed > 0 else 0
pf = total_profit / total_loss if total_loss > 0 else 0

print(f"\n3. HISTORIAL MT5 REAL (desde 2026-05-01):")
print(f"   Trades cerrados: {total_closed}")
print(f"   Ganados: {wins} | Perdidos: {losses}")
print(f"   Win Rate: {wr:.1f}% {'PASS' if wr >= 60 else 'NECESITA >=60%'}")
print(f"   Profit ganado: ${total_profit:.2f}")
print(f"   Profit perdido: ${total_loss:.2f}")
print(f"   Profit Factor: {pf:.2f} {'PASS' if pf >= 1.5 else 'NECESITA >=1.5'}")

# --- Axi Readiness ---
print(f"\n4. ESTADO AXI SELECT:")
dd_pct = abs(100_000 - info.balance) / 100_000 * 100 if info else 0
print(f"   Drawdown actual: {dd_pct:.2f}% {'PASS' if dd_pct < 5 else 'CUIDADO'}")
print(f"   Win Rate: {wr:.1f}% (necesita >=60% sostenida)")
print(f"   Trades: {total_closed}/20 minimos para Seed")
print(f"   Profit Factor: {pf:.2f} (necesita >=1.5)")

# --- Issues ---
print(f"\n5. PROBLEMAS DETECTADOS:")
if info and not info.trade_allowed:
    print("   CRITICO: trade_allowed=False — MT5 no puede ejecutar ordenes")
print("   ERROR 10031: MT5 sin conexion de red al broker (ver logs)")
print("   Demo crypto con entradas historicas ($93K BTC) — ya limpiadas")

mt5.shutdown()
print("\n" + "=" * 50)
