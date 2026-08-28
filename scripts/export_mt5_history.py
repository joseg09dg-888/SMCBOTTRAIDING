"""
Exporta la historia H1 + D1 real de MT5 (Axi) a CSV en data/historical/
para que quede respaldada en git y nunca se vuelva a perder con un cambio de PC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
import pandas as pd
import MetaTrader5 as mt5

import os
PAIRS = os.environ.get("PAIRS", "EURUSD,USDCAD,NZDUSD,USDCHF,EURAUD,GBPCAD").split(",")
OUT_DIR = Path(__file__).parent.parent / "data" / "historical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ok = mt5.initialize(timeout=15000)
if not ok:
    print(f"ERROR conectando: {mt5.last_error()}")
    sys.exit(1)

acc = mt5.account_info()
print(f"Conectado: login={acc.login} server={acc.server}")

summary = []
for pair in PAIRS:
    mt5.symbol_select(pair, True)

    rates_h1 = mt5.copy_rates_from(pair, mt5.TIMEFRAME_H1, datetime.now(timezone.utc), 99999)
    rates_d1 = mt5.copy_rates_from(pair, mt5.TIMEFRAME_D1, datetime.now(timezone.utc), 10000)

    for label, rates, tf in [("H1", rates_h1, "H1"), ("D1", rates_d1, "D1")]:
        if rates is None or len(rates) == 0:
            print(f"  {pair} {label}: SIN DATOS ({mt5.last_error()})")
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        out_path = OUT_DIR / f"{pair}_{tf}.csv"
        df.to_csv(out_path, index=False)
        years = (df["time"].iloc[-1] - df["time"].iloc[0]).days / 365.25
        print(f"  {pair} {label}: {len(df)} filas, {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()} (~{years:.1f} anios) -> {out_path}")
        summary.append((pair, tf, len(df), years))

mt5.shutdown()

print("\n=== RESUMEN ===")
for pair, tf, n, years in summary:
    print(f"{pair} {tf}: {n} filas (~{years:.1f} anios)")
