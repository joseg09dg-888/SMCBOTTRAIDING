import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("MT5 no disponible:", mt5.last_error())
    exit()

info = mt5.account_info()
print(f"=== CUENTA AXI DEMO ===")
print(f"Balance actual:  ${info.balance:,.2f}")
print(f"Equity actual:   ${info.equity:,.2f}")
print(f"P&L flotante:    ${info.profit:+,.2f}")
print(f"Balance inicial: $99,470.20")
print(f"P&L TOTAL:       ${info.balance - 99470.20:+,.2f}")

# All history
from_date = datetime(2026, 1, 1)
deals = mt5.history_deals_get(from_date, datetime.now())
if deals:
    print(f"\n=== HISTORIAL COMPLETO ({len(deals)} deals) ===")
    pnl_total = 0
    winners = 0
    losers = 0
    by_symbol = {}
    for d in deals:
        if d.profit != 0:
            pnl_total += d.profit
            if d.profit > 0:
                winners += 1
            else:
                losers += 1
            sym = d.symbol
            if sym not in by_symbol:
                by_symbol[sym] = {"pnl": 0, "count": 0, "wins": 0}
            by_symbol[sym]["pnl"] += d.profit
            by_symbol[sym]["count"] += 1
            if d.profit > 0:
                by_symbol[sym]["wins"] += 1
            dt = datetime.fromtimestamp(d.time)
            tipo = "BUY" if d.type == 0 else "SELL"
            print(f"  {dt.strftime('%m-%d %H:%M')} {d.symbol:10s} {tipo:4s} vol={d.volume} PNL={round(d.profit,2):+.2f}")

    print(f"\n=== RESUMEN POR SIMBOLO ===")
    for sym, data in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"]):
        wr = data["wins"]/data["count"]*100 if data["count"] > 0 else 0
        print(f"  {sym:10s} | trades={data['count']:3d} | WR={wr:.0f}% | PNL={data['pnl']:+.2f}")

    print(f"\n=== ESTADISTICAS ===")
    print(f"Winners: {winners} | Losers: {losers}")
    wr_global = winners/(winners+losers)*100 if (winners+losers) > 0 else 0
    print(f"Win Rate global: {wr_global:.1f}%")
    print(f"PNL total cerrado: ${pnl_total:+.2f}")

mt5.shutdown()
