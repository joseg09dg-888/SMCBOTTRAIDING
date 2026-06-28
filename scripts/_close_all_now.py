"""Close ALL open positions immediately."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.metatrader_connector import MT5Connector
from core.config import Config

cfg = Config()
mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
mt5.connect()

positions = mt5.get_positions()
if not positions:
    print("Sin posiciones abiertas.")
    sys.exit(0)

for p in positions:
    sym = p.get("symbol", "?")
    ticket = p["ticket"]
    pnl = p.get("profit", 0.0)
    ok = mt5.close_position(ticket)
    print(f"{'CERRADA' if ok else 'ERROR'} {sym} #{ticket} ${pnl:+.2f}")
