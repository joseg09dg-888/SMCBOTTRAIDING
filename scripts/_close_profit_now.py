"""Close all open positions in profit immediately."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.metatrader_connector import MT5Connector
from core.config import Config

cfg = Config()
mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)

if not mt5.connect():
    print("ERROR: no se pudo conectar a MT5")
    sys.exit(1)

positions = mt5.get_positions()
if not positions:
    print("Sin posiciones abiertas.")
    mt5.disconnect()
    sys.exit(0)

total_pnl = sum(p.get("profit", 0.0) for p in positions)
print(f"Posiciones abiertas: {len(positions)} | PnL total: ${total_pnl:.2f}")
print()

closed = 0
for p in positions:
    sym    = p.get("symbol", "?")
    ticket = p["ticket"]
    pnl    = p.get("profit", 0.0)
    size   = p.get("volume", p.get("size", 0))
    print(f"  {sym} #{ticket} vol={size} PnL=${pnl:+.2f} ... ", end="", flush=True)
    ok = mt5.close_position(ticket)
    if ok:
        print("CERRADA ✓")
        closed += 1
    else:
        print("ERROR al cerrar")

print(f"\nCerradas {closed}/{len(positions)} posiciones. PnL realizado: ${total_pnl:.2f}")
mt5.disconnect()
