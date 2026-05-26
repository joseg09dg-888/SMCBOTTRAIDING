from connectors.metatrader_connector import MT5Connector
from core.config import config

mt5c = MT5Connector(config.mt5_login, config.mt5_password, config.mt5_server)
pnl  = mt5c.get_pnl_report(100000.0)

print("=== RESULTADO P&L ===")
for k, v in pnl.items():
    if k != "recent_trades":
        print(f"  {k}: {v}")

print("\nUltimas operaciones:")
for t in pnl.get("recent_trades", []):
    print(f"  {t['dt']} {t['symbol']} {t['direction']} "
          f"{t['volume']}lot @{t['price']:.3f} "
          f"notional=${t['notional_usd']:.2f} pnl={t['profit']:+.2f}")
