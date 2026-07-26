"""
DETECTOR DE ANOMALIAS DE MERCADO -- 16 anios reales de MT5 (Axi)
==================================================================
Jose pidio memoria EXPLICITA, no solo promedios estadisticos: un registro
con fecha y hora exacta de cada movimiento anomalo real detectado en los
16.1 anios de velas H1 reales (no sinteticas, no yfinance) de los 6 pares
activos. No hay datos de tick/0.1s disponibles para ese periodo -- esto
usa la resolucion mas fina que SI existe de verdad: H1.

Detecta 3 tipos de anomalia por vela, todos relativos al ATR(14) de esa
vela en ese momento (asi es comparable entre pares y entre epocas de
volatilidad distinta):
  - RANGE:  (high-low) > UMBRAL x ATR14      -- vela con rango extremo
  - GAP:    |open - close_previo| > UMBRAL x ATR14 -- gap real (fin de
            semana, apertura de sesion, noticia)
  - BODY:   |close-open| > UMBRAL x ATR14     -- movimiento direccional
            extremo dentro de una sola vela (posible noticia/manipulacion)

Guarda todo en memory/market_anomalies_16y.json para que quede como
referencia permanente y consultable, no solo como analisis de una sesion.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone

PAIRS = ["EURUSD", "USDCAD", "NZDUSD", "USDCHF", "EURAUD", "GBPCAD"]
RANGE_THRESHOLD = 6.0   # vela con rango > 6x ATR14 = extrema
GAP_THRESHOLD   = 4.0   # gap > 4x ATR14 = extremo
BODY_THRESHOLD  = 5.0   # cuerpo > 5x ATR14 = movimiento direccional extremo
MAX_BARS = 99999        # limite real del terminal (ver backtest_multiyear.py)

print("=" * 72)
print("  DETECTOR DE ANOMALIAS -- 16 anios reales MT5 (Axi)")
print("=" * 72)

if not mt5.initialize():
    print(f"ERROR: mt5.initialize() fallo: {mt5.last_error()}")
    sys.exit(1)

all_anomalies = []

for pair in PAIRS:
    mt5.symbol_select(pair, True)
    rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_H1, datetime.now(timezone.utc), MAX_BARS)
    if rates is None or len(rates) == 0:
        print(f"  {pair}: sin datos, error={mt5.last_error()}")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")  # naive, matches live convention
    df.set_index("time", inplace=True)

    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    rng = h - l
    gap = (o - prev_c).abs()
    body = (c - o).abs()

    pair_count = 0
    for idx in range(20, len(df)):
        a = atr14.iloc[idx]
        if pd.isna(a) or a <= 0:
            continue
        ts = df.index[idx]
        events = []
        if rng.iloc[idx] > RANGE_THRESHOLD * a:
            events.append(("RANGE", round(rng.iloc[idx] / a, 2)))
        if gap.iloc[idx] > GAP_THRESHOLD * a:
            events.append(("GAP", round(gap.iloc[idx] / a, 2)))
        if body.iloc[idx] > BODY_THRESHOLD * a:
            events.append(("BODY", round(body.iloc[idx] / a, 2)))
        for etype, mult in events:
            all_anomalies.append({
                "pair": pair,
                "datetime_utc_naive": ts.strftime("%Y-%m-%d %H:%M"),
                "weekday": ts.strftime("%A"),
                "type": etype,
                "atr_multiple": mult,
                "open": float(o.iloc[idx]), "high": float(h.iloc[idx]),
                "low": float(l.iloc[idx]), "close": float(c.iloc[idx]),
                "atr14_at_time": round(float(a), 6),
            })
            pair_count += 1

    print(f"  {pair}: {len(df)} velas escaneadas -> {pair_count} anomalias detectadas")

mt5.shutdown()

all_anomalies.sort(key=lambda e: e["datetime_utc_naive"])

out = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": "MT5 real (Axi) H1, ~16.1 anios",
    "thresholds": {"RANGE_ATR_MULT": RANGE_THRESHOLD, "GAP_ATR_MULT": GAP_THRESHOLD, "BODY_ATR_MULT": BODY_THRESHOLD},
    "total_anomalies": len(all_anomalies),
    "by_pair": {p: sum(1 for e in all_anomalies if e["pair"] == p) for p in PAIRS},
    "by_type": {t: sum(1 for e in all_anomalies if e["type"] == t) for t in ("RANGE", "GAP", "BODY")},
    "events": all_anomalies,
}

out_path = os.path.join("memory", "market_anomalies_16y.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print(f"\n  Total anomalias detectadas: {len(all_anomalies)}")
print(f"  Por par: {out['by_pair']}")
print(f"  Por tipo: {out['by_type']}")
print(f"  Guardado en: {out_path}")

# Top 10 mas extremos, para lectura rapida
top10 = sorted(all_anomalies, key=lambda e: -e["atr_multiple"])[:10]
print("\n  TOP 10 MAS EXTREMOS:")
for e in top10:
    print(f"    {e['datetime_utc_naive']} ({e['weekday']}) {e['pair']} {e['type']} {e['atr_multiple']}x ATR")
