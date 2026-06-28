"""Check current positions and today's realized PnL, close all if daily target hit."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.metatrader_connector import MT5Connector
from core.config import Config

cfg = Config()
mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
mt5.connect()

daily_pnl = mt5.get_daily_pnl()
print(f"PnL realizado HOY: ${daily_pnl:.2f}")

positions = mt5.get_positions()
float_pnl = sum(p.get("profit", 0.0) for p in positions)
total = daily_pnl + float_pnl
print(f"Float PnL: ${float_pnl:.2f} | TOTAL HOY: ${total:.2f}")
print()

for p in positions:
    vol = p.get("volume", p.get("size", 0))
    print(f"  {p['symbol']} #{p['ticket']} vol={vol} PnL=${p.get('profit',0):+.2f}")

if total >= 150:
    print(f"\nMETA $150 ALCANZADA — cerrando todo...")
    for p in positions:
        sym = p.get("symbol", "?")
        ticket = p["ticket"]
        pnl = p.get("profit", 0.0)
        ok = mt5.close_position(ticket)
        print(f"  {sym} #{ticket} ${pnl:+.2f} -> {'OK' if ok else 'ERROR'}")
else:
    print(f"\nMeta no alcanzada todavia. Faltan ${150-total:.2f}")
