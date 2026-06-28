"""Test de velocidad real del pipeline completo."""
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
mt5.initialize()

print("=== VELOCIDAD REAL DEL PIPELINE ===\n")

# 1. Obtener datos MT5
t0 = time.perf_counter()
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 200)
t1 = time.perf_counter()
print(f"Datos H1 200 velas:      {(t1-t0)*1000:.1f}ms")

import pandas as pd
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.rename(columns={'open':'open','high':'high','low':'low','close':'close','tick_volume':'volume'}, inplace=True)

# 2. SMC analysis
from smc.structure import MarketStructure
from smc.orderblocks import OrderBlockDetector
t0 = time.perf_counter()
ms = MarketStructure(df)
bos_list = ms.detect_bos()
choch_list = ms.detect_choch()
ob = OrderBlockDetector(df)
bull_obs = ob.detect_bullish_ob()
bear_obs = ob.detect_bearish_ob()
t1 = time.perf_counter()
print(f"SMC analysis completo:   {(t1-t0)*1000:.1f}ms  ({len(bos_list)} BOS, {len(bull_obs)+len(bear_obs)} OBs)")

# 3. Signal agent
from agents.signal_agent import SignalAgent
agent = SignalAgent()
analysis_text = "BOS alcista reciente. OB bullish en 1.0800. Setup valido."
t0 = time.perf_counter()
signal = agent.evaluate(analysis_text, df, symbol="EURUSD", timeframe="H1")
t1 = time.perf_counter()
print(f"SignalAgent.evaluate():  {(t1-t0)*1000:.1f}ms  → {signal.signal_type.value}")

# 4. DecisionFilter
import json
with open("memory/scores.db.json", "r") if os.path.exists("memory/scores.db.json") else open(os.devnull) as f:
    pass
# Score via SQLite
import sqlite3
try:
    conn = sqlite3.connect("memory/scores.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM scores")
    n_scores = c.fetchone()[0]
    conn.close()
    print(f"Scores en DB:            {n_scores} historicos")
except:
    print(f"Scores en DB:            N/A")

# 5. MT5 tick actual
t0 = time.perf_counter()
tick = mt5.symbol_info_tick("EURUSD")
t1 = time.perf_counter()
print(f"MT5 tick actual:         {(t1-t0)*1000:.1f}ms  → {tick.bid}/{tick.ask}")

# 6. Enrichment — cuanto tarda 1 agente real
from agents.chaos_agent import ChaosTheoryAgent
chaos = ChaosTheoryAgent()
t0 = time.perf_counter()
result = chaos.analyze(df, "EURUSD", "1h")
t1 = time.perf_counter()
print(f"ChaosAgent.analyze():    {(t1-t0)*1000:.1f}ms  → {str(result)[:50]}")

# 7. Estimado ciclo completo
print(f"\n=== TIEMPO ESTIMADO 1 CICLO COMPLETO ===")
print(f"7 pares × 2 timeframes × (datos+SMC+signal+score)")
print(f"Estimado: ~14 calls MT5 + 14 SMC + 14 signal + enrichment")
print(f"Con scan_interval=30s → mucho tiempo muerto esperando")

# 8. Latencia Glint
import json, os
if os.path.exists("memory/glint_session.json"):
    with open("memory/glint_session.json") as f:
        glint = json.load(f)
    print(f"\nGlint ultima actualizacion: {glint.get('last_update', 'N/A')}")
    print(f"Glint ultimo signal:        {glint.get('last_signal', 'N/A')}")
    signals = glint.get('signals', [])
    print(f"Glint signals guardados:    {len(signals)}")
    if signals:
        print(f"Ultimo: {signals[-1]}")

mt5.shutdown()
print("\nDone.")
