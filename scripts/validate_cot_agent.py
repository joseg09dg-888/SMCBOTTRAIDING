"""
Validacion empirica del COT bonus (institutional_flow_agent) contra trades reales.
Para cada trade con simbolo cubierto por CFTC, busca el reporte COT semanal mas
reciente ANTES de la fecha del trade y compara si el bias comercial coincidio
con la direccion del trade -- igual que hace get_combined_signal() en vivo.
"""
import sqlite3
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datetime import datetime, timezone

from agents.institutional_flow_agent import InstitutionalFlowAgent

COT_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
CONTRACT_NAMES = {
    "EURUSD": "EURO FX", "GBPUSD": "BRITISH POUND", "AUDUSD": "AUSTRALIAN DOLLAR",
    "NZDUSD": "NZ DOLLAR", "USDCAD": "CANADIAN DOLLAR", "USDJPY": "JAPANESE YEN",
}

conn = sqlite3.connect("memory/episodes.db")
trades = conn.execute(
    "SELECT symbol, direction, ts, result FROM episodes WHERE result IN ('WIN','LOSS')"
).fetchall()
trades = [t for t in trades if t[0] in CONTRACT_NAMES]
print(f"[DATA] {len(trades)} trades con simbolo cubierto por CFTC (de los totales)")

cache = {}  # (contract, date_str) -> commercial_net

def cot_net_before(contract, before_date):
    key = (contract, before_date)
    if key in cache:
        return cache[key]
    try:
        resp = requests.get(COT_URL, params={
            "$limit": 1,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$where": f"contract_market_name='{contract}' AND report_date_as_yyyy_mm_dd < '{before_date}'",
        }, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            cache[key] = None
            return None
        row = rows[0]
        net = int(row.get("dealer_positions_long_all", 0)) - int(row.get("dealer_positions_short_all", 0))
        cache[key] = net
        return net
    except Exception as e:
        cache[key] = None
        return None

buckets = {"agree": {"wins": 0, "total": 0}, "disagree": {"wins": 0, "total": 0}, "no_data": {"wins": 0, "total": 0}}

for i, (symbol, direction, ts, result) in enumerate(trades):
    contract = CONTRACT_NAMES[symbol]
    date_str = ts[:10]
    net = cot_net_before(contract, date_str)
    if i % 50 == 0:
        print(f"  ...procesando {i}/{len(trades)}")
    time.sleep(0.05)  # be nice to the API

    bias = "bullish" if net and net > 0 else ("bearish" if net and net < 0 else None)
    trade_bias = "bullish" if (direction or "").upper() in ("LONG", "BUY") else "bearish"

    if bias is None:
        key = "no_data"
    elif bias == trade_bias:
        key = "agree"
    else:
        key = "disagree"

    buckets[key]["total"] += 1
    if result == "WIN":
        buckets[key]["wins"] += 1

print(f"\n{'Grupo':>10s} | {'Trades':>7s} | {'WR':>6s}")
print("-" * 32)
for k in ("agree", "disagree", "no_data"):
    b = buckets[k]
    wr = b["wins"] / b["total"] * 100 if b["total"] else 0
    print(f"{k:>10s} | {b['total']:>7d} | {wr:>5.1f}%")

print("""
INTERPRETACION:
  Si COT tiene poder predictivo real, 'agree' deberia ganar bastante mas
  que 'disagree'. Si son parecidos, el bonus +15 no esta aportando señal.
""")
