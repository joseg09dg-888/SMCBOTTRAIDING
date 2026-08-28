import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
from core.config import config
from connectors.metatrader_connector import MT5Connector
import MetaTrader5 as mt5

print(f"MT5_LOGIN presente: {bool(config.mt5_login)}")
print(f"MT5_SERVER: {config.mt5_server}")

conn = MT5Connector(login=int(config.mt5_login), password=config.mt5_password, server=config.mt5_server)
ok = conn.connect()
print(f"Conectado: {ok}")
if not ok:
    print(f"Error MT5: {mt5.last_error()}")
    sys.exit(1)

acc = mt5.account_info()
print(f"Cuenta: login={acc.login} server={acc.server} balance={acc.balance} {acc.currency}")

for pair in ["EURUSD", "USDCAD", "NZDUSD", "USDCHF", "EURAUD", "GBPCAD"]:
    mt5.symbol_select(pair, True)
    rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_H1, datetime.now(timezone.utc), 99999)
    if rates is None or len(rates) == 0:
        print(f"  {pair}: SIN DATOS ({mt5.last_error()})")
        continue
    first = datetime.fromtimestamp(rates[0]["time"])
    last = datetime.fromtimestamp(rates[-1]["time"])
    years = (last - first).days / 365.25
    print(f"  {pair}: {len(rates)} velas H1, desde {first.date()} hasta {last.date()} (~{years:.1f} años)")

mt5.shutdown()
