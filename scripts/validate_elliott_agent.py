"""
Validacion empirica del ElliottFibonacciAgent contra trades reales cerrados.
Para cada trade en episodes.db, reconstruye el score_bonus que Elliott habria
dado en el momento de la entrada (usando solo datos H1 disponibles hasta esa
fecha) y compara el WR real agrupado por nivel de bonus.
"""
import sqlite3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

from agents.elliott_agent import ElliottFibonacciAgent

TICKERS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X", "USDCHF": "USDCHF=X",
    "EURAUD": "EURAUD=X", "GBPCAD": "GBPCAD=X", "GBPJPY": "GBPJPY=X",
    "USDJPY": "USDJPY=X", "NAS100.fs": "^NDX", "XAUUSD": "GC=F",
    "US30": "^DJI",
}

conn = sqlite3.connect("memory/episodes.db")
trades = conn.execute(
    "SELECT symbol, direction, ts, result FROM episodes WHERE result IN ('WIN','LOSS')"
).fetchall()
print(f"[DATA] {len(trades)} trades cerrados en episodes.db")

# Descargar H1 una sola vez por simbolo (max 700 dias por limite yfinance)
symbols_needed = {t[0] for t in trades if t[0] in TICKERS}
h1_cache = {}
for sym in symbols_needed:
    try:
        df = yf.download(TICKERS[sym], period="729d", interval="1h",
                          progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.dropna(inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        h1_cache[sym] = df
        print(f"  {sym}: {len(df)} velas H1 descargadas")
    except Exception as e:
        print(f"  {sym}: fallo descarga -- {e}")

agent = ElliottFibonacciAgent()
buckets = {}  # bonus_value -> {"wins":0, "total":0}
skipped = 0

for symbol, direction, ts, result in trades:
    if symbol not in h1_cache:
        skipped += 1
        continue
    try:
        entry_time = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")
    except Exception:
        skipped += 1
        continue

    df_sym = h1_cache[symbol]
    hist = df_sym[df_sym.index < entry_time].tail(200)
    if len(hist) < 20:
        skipped += 1
        continue

    bias = "bullish" if (direction or "").upper() in ("LONG", "BUY") else "bearish"
    try:
        r = agent.analyze(hist, bias)
    except Exception:
        skipped += 1
        continue

    key = r.score_bonus
    if key not in buckets:
        buckets[key] = {"wins": 0, "total": 0}
    buckets[key]["total"] += 1
    if result == "WIN":
        buckets[key]["wins"] += 1

print(f"\n[SKIPPED] {skipped} trades sin datos suficientes")
print(f"\n{'Bonus Elliott':>15s} | {'Trades':>7s} | {'WR':>6s}")
print("-" * 40)
overall_total = sum(b["total"] for b in buckets.values())
overall_wins  = sum(b["wins"] for b in buckets.values())
for bonus in sorted(buckets.keys()):
    b = buckets[bonus]
    wr = b["wins"] / b["total"] * 100 if b["total"] else 0
    print(f"{bonus:>+15d} | {b['total']:>7d} | {wr:>5.1f}%")
print("-" * 40)
print(f"{'TOTAL':>15s} | {overall_total:>7d} | {overall_wins/max(1,overall_total)*100:>5.1f}%")

print("""
INTERPRETACION:
  Si Elliott tiene poder predictivo real, el WR deberia SUBIR con el bonus
  (bonus=+10 deberia ganar mas que bonus=-5 o bonus=0).
  Si el WR es parecido en todos los buckets (o invertido), el agente no
  esta aportando señal real -- solo ruido con apariencia de analisis.
""")
