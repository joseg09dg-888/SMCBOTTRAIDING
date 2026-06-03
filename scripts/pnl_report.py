"""P&L completo: balance, trades cerrados, ganadores vs perdedores."""
import sys; sys.path.insert(0, ".")
import MetaTrader5 as mt5
from datetime import datetime, timezone

mt5.initialize()
acc = mt5.account_info()

balance   = acc.balance
equity    = acc.equity
profit    = acc.profit
net       = balance - 100_000.0
net_pct   = net / 100_000.0 * 100

print("=" * 50)
print("RESUMEN DE CUENTA")
print("=" * 50)
print(f"  Capital inicial:  $100,000.00")
print(f"  Balance actual:   ${balance:,.2f}")
print(f"  Resultado neto:   ${net:+,.2f}  ({net_pct:+.3f}%)")
print(f"  P&L abierto:      ${profit:+.2f}")
print(f"  Equity total:     ${equity:,.2f}")

# Trades cerrados este mes
from_date = datetime(2026, 5, 1, tzinfo=timezone.utc)
to_date   = datetime.now(timezone.utc)
deals = mt5.history_deals_get(from_date, to_date) or []
closing = [d for d in deals if d.entry == 1 and d.symbol != ""]

wins   = [d for d in closing if d.profit > 0]
losses = [d for d in closing if d.profit < 0]
breakeven = [d for d in closing if d.profit == 0]

realized    = sum(d.profit + d.commission + d.swap for d in closing)
gross_win   = sum(d.profit for d in wins)
gross_loss  = sum(d.profit for d in losses)
pf = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")
wr = len(wins) / len(closing) * 100 if closing else 0

print()
print("=" * 50)
print("TRADES CERRADOS (Mayo 2026)")
print("=" * 50)
print(f"  Total trades:     {len(closing)}")
print(f"  Ganadores:        {len(wins)}  (${gross_win:+,.2f})")
print(f"  Perdedores:       {len(losses)}  (${gross_loss:+,.2f})")
print(f"  Win rate:         {wr:.1f}%")
print(f"  Profit factor:    {pf:.2f}")
print(f"  P&L realizado:    ${realized:+,.2f}")

print()
print("ULTIMOS 10 TRADES:")
print(f"  {'Fecha':<14} {'Par':<8} {'Dir':<5} {'Profit':>10}")
print(f"  {'-'*40}")
for d in closing[-10:]:
    dt  = datetime.fromtimestamp(d.time, tz=timezone.utc).strftime("%d-%b %H:%M")
    dir = "BUY" if d.type == 0 else "SELL"
    tag = "WIN" if d.profit > 0 else ("LOSS" if d.profit < 0 else "BE")
    print(f"  {dt:<14} {d.symbol:<8} {dir:<5} ${d.profit:>+8.2f}  {tag}")

mt5.shutdown()
