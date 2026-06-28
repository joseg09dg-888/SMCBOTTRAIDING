import MetaTrader5 as mt5, os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from collections import defaultdict

load_dotenv()
mt5.initialize()
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

acc = mt5.account_info()
balance = acc.balance
equity  = acc.equity
print("=" * 60)
print("AUDITORIA COMPLETA SMC BOT")
print("=" * 60)
print(f"Balance:      ${balance:.2f}")
print(f"Equity:       ${equity:.2f}")
print(f"Float P&L:    ${acc.profit:.2f}")
print(f"Drawdown:     ${balance - 100000:.2f} ({(balance/100000-1)*100:.2f}%)")
print()

# ── POSICIONES ABIERTAS ────────────────────────────────────────
positions = mt5.positions_get()
now_utc = datetime.now(timezone.utc)
print("=== POSICIONES ABIERTAS ===")
if positions:
    for p in positions:
        side = 'BUY' if p.type == 0 else 'SELL'
        open_time = datetime.fromtimestamp(p.time, tz=timezone.utc)
        hours_open = (now_utc - open_time).total_seconds() / 3600
        pnl_status = "GANANDO" if p.profit > 0 else "PERDIENDO"
        print(f"  #{p.ticket} {p.symbol} {side} {p.volume}L")
        print(f"    Abierta hace: {hours_open:.1f}h")
        print(f"    Entry: {p.price_open:.5f} | Actual: {p.price_current:.5f}")
        print(f"    P&L: ${p.profit:.2f} ({pnl_status}) | SL: {p.sl:.5f} | TP: {p.tp:.5f}")
        dist_tp = abs(p.price_current - p.tp)
        dist_sl = abs(p.price_current - p.sl)
        print(f"    Distancia al TP: {dist_tp:.5f} | Distancia al SL: {dist_sl:.5f}")
else:
    print("  Sin posiciones abiertas")
print()

# ── HISTORIAL COMPLETO ────────────────────────────────────────
from_dt = datetime(2026, 5, 1, tzinfo=timezone.utc)
history = mt5.history_deals_get(from_dt, now_utc)

# Separate by bot vs manual
bot_wins, bot_losses, bot_pnl_list = [], [], []
man_wins, man_losses = [], []
symbol_stats = defaultdict(lambda: {'w':0,'l':0,'pnl':0.0,'hold_hours':[]})
hold_times_all = []

# Build position open times from orders
orders_history = mt5.history_orders_get(from_dt, now_utc)
ticket_to_open_time = {}
if orders_history:
    for o in orders_history:
        ticket_to_open_time[o.ticket] = o.time_setup

# Process deals (entries and exits)
# Group by position_id to compute hold time
pos_entry_time = {}
pos_entry_price = {}
pos_symbol = {}
pos_type = {}

for d in history:
    if d.entry == 0:  # entry deal
        pos_entry_time[d.position_id] = d.time
        pos_entry_price[d.position_id] = d.price
        pos_symbol[d.position_id] = d.symbol
        pos_type[d.position_id] = d.type

for d in history:
    if d.profit == 0 and d.commission == 0:
        continue
    if d.entry == 0:  # skip entry deals for P&L (they have 0 profit)
        continue

    pnl = d.profit + d.commission + d.swap
    comment = d.comment or ""
    is_bot = ("SMC" in comment or "smc" in comment.lower() or
              "NoSL" in comment or "PEAK" in comment or
              "sl " in comment.lower())

    # Hold time
    entry_t = pos_entry_time.get(d.position_id, d.time)
    hold_secs = d.time - entry_t
    hold_h = hold_secs / 3600

    sym = d.symbol

    if is_bot:
        if pnl > 0:
            bot_wins.append(pnl)
        else:
            bot_losses.append(pnl)
        bot_pnl_list.append(pnl)
        symbol_stats[sym]['pnl'] += pnl
        if pnl > 0:
            symbol_stats[sym]['w'] += 1
        else:
            symbol_stats[sym]['l'] += 1
        if hold_h > 0:
            symbol_stats[sym]['hold_hours'].append(hold_h)
            hold_times_all.append(hold_h)
    else:
        if pnl > 0:
            man_wins.append(pnl)
        else:
            man_losses.append(pnl)

print("=== ESTADISTICAS BOT (desde May 2026) ===")
total_bot = len(bot_wins) + len(bot_losses)
wr = len(bot_wins) / total_bot * 100 if total_bot > 0 else 0
avg_win  = sum(bot_wins)  / len(bot_wins)  if bot_wins  else 0
avg_loss = sum(bot_losses)/ len(bot_losses) if bot_losses else 0
pf = abs(sum(bot_wins) / sum(bot_losses)) if bot_losses else 999
avg_hold = sum(hold_times_all) / len(hold_times_all) if hold_times_all else 0
net_bot = sum(bot_pnl_list)

print(f"  Trades bot:   {total_bot} ({len(bot_wins)}W / {len(bot_losses)}L)")
print(f"  Win rate:     {wr:.1f}%")
print(f"  Avg ganancia: ${avg_win:.2f} | Avg perdida: ${avg_loss:.2f}")
print(f"  Profit factor:{pf:.2f}")
print(f"  Net bot P&L:  ${net_bot:.2f}")
print(f"  Hold promedio:{avg_hold:.1f}h")
print()

print("=== ESTADISTICAS MANUALES ===")
total_man = len(man_wins) + len(man_losses)
print(f"  Trades manual:{total_man} ({len(man_wins)}W / {len(man_losses)}L)")
print(f"  Net manual:   ${sum(man_wins)+sum(man_losses):.2f}")
print()

print("=== POR SIMBOLO (bot) ===")
for sym, s in sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
    tot = s['w'] + s['l']
    if tot == 0: continue
    wr_s = s['w']/tot*100
    avg_h = sum(s['hold_hours'])/len(s['hold_hours']) if s['hold_hours'] else 0
    print(f"  {sym:8s}: {s['w']}W/{s['l']}L WR={wr_s:.0f}% | P&L=${s['pnl']:+.2f} | hold_avg={avg_h:.1f}h")
print()

# ── PROYECCION PARA CHALLENGE ─────────────────────────────────
print("=== PROYECCION CHALLENGE AXI (5% mensual) ===")
target_pct  = 0.05
target_usd  = balance * target_pct
already_up  = balance - 97693.60  # desde que se arreglaron los bugs (ayer)
needed      = target_usd - already_up
max_daily_loss_usd = balance * 0.03  # 3% daily limit Axi Select
print(f"  Balance actual:         ${balance:.2f}")
print(f"  Meta 5% = ${target_usd:.2f}")
print(f"  Ya ganado (post-fix):   ${already_up:.2f}")
print(f"  Falta ganar:            ${needed:.2f}")
print(f"  Max perdida diaria:     ${max_daily_loss_usd:.2f} (3%)")
print()
print(f"  Con avg_win ${avg_win:.2f} y WR {wr:.0f}%:")
trades_needed = int(needed / (avg_win * wr/100 + avg_loss * (1-wr/100))) + 1 if (avg_win+abs(avg_loss)) > 0 else 999
print(f"    Trades totales necesarios: ~{trades_needed}")
print(f"    Trades ganadores necesarios: ~{int(needed/avg_win)+1 if avg_win>0 else 999}")
print(f"  Con 3 setups/dia (US30+USDCAD+GBPUSD):")
days_3 = trades_needed / 3
print(f"    Tiempo estimado: {days_3:.0f} dias ({days_3/5:.1f} semanas)")
print()
print(f"  MAX HOLD recomendado: 12h (posiciones no deben durar mas)")
print(f"  Hold actual promedio: {avg_hold:.1f}h")

mt5.shutdown()
