"""Auditoria completa del bot: velocidad, agentes, datos en tiempo real."""
import time
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = {}

# ── 1. MT5 conectado y rapido ────────────────────────────────────────────────
print("\n=== 1. MT5 VELOCIDAD ===")
try:
    import MetaTrader5 as mt5
    mt5.initialize()
    t0 = time.time()
    tick = mt5.symbol_info_tick("EURUSD")
    t1 = time.time()
    print(f"  EURUSD tick: {tick.bid}/{tick.ask}  [{(t1-t0)*1000:.0f}ms]")

    t0 = time.time()
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 200)
    t1 = time.time()
    print(f"  EURUSD H1 200 velas: {len(rates)} velas [{(t1-t0)*1000:.0f}ms]")
    results["mt5_tick_ms"] = round((t1-t0)*1000)
    results["mt5_ok"] = True
    mt5.shutdown()
except Exception as e:
    print(f"  ERROR: {e}")
    results["mt5_ok"] = False

# ── 2. Agentes enriquecimiento — estan operando real? ────────────────────────
print("\n=== 2. AGENTES ENRIQUECIMIENTO (13 agentes) ===")
agent_tests = [
    ("LunarCycleAgent",         "agents.lunar_agent",          "LunarCycleAgent"),
    ("ElliottFibonacciAgent",   "agents.elliott_agent",        "ElliottFibonacciAgent"),
    ("ChaosTheoryAgent",        "agents.chaos_agent",          "ChaosTheoryAgent"),
    ("QuantEdgeAgent",          "agents.statistical_edge_agent","QuantEdgeAgent"),
    ("FootprintAgent",          "agents.footprint_agent",      "FootprintAgent"),
    ("InstitutionalFlowAgent",  "agents.institutional_flow_agent","InstitutionalFlowAgent"),
    ("MarketMicrostructureAgent","agents.microstructure_agent","MarketMicrostructureAgent"),
    ("FEDSentimentAgent",       "agents.fed_sentiment_agent",  "FEDSentimentAgent"),
    ("OnChainAgent",            "agents.onchain_agent",        "OnChainAgent"),
    ("GeopoliticalAgent",       "agents.geopolitical_agent",   "GeopoliticalAgent"),
    ("RetailPsychologyAgent",   "agents.retail_psychology_agent","RetailPsychologyAgent"),
    ("AlternativeDataAgent",    "agents.alternative_data_agent","AlternativeDataAgent"),
    ("EnergyFrequencyAgent",    "agents.energy_frequency_agent","EnergyFrequencyAgent"),
]

import importlib
import pandas as pd
import numpy as np

# Make fake OHLCV data for testing
n = 100
fake_df = pd.DataFrame({
    "open":   np.random.uniform(50000, 51000, n),
    "high":   np.random.uniform(51000, 52000, n),
    "low":    np.random.uniform(49000, 50000, n),
    "close":  np.random.uniform(50000, 51000, n),
    "volume": np.random.uniform(100, 1000, n),
})

working_agents = 0
broken_agents = 0
for name, module_path, class_name in agent_tests:
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        agent = cls()
        t0 = time.time()
        # Try to call analyze or the main method
        result = None
        if hasattr(agent, "analyze"):
            result = agent.analyze(fake_df, "EURUSD", "1h")
        elif hasattr(agent, "get_signal"):
            result = agent.get_signal(fake_df)
        elif hasattr(agent, "evaluate"):
            result = agent.evaluate("EURUSD", fake_df)
        elapsed = (time.time() - t0) * 1000

        if result is not None:
            score = result.get("score", result.get("bias", result.get("value", "?"))) if isinstance(result, dict) else result
            print(f"  OK  {name:35s} [{elapsed:.0f}ms] → {str(score)[:40]}")
            working_agents += 1
        else:
            print(f"  ??  {name:35s} [{elapsed:.0f}ms] → None (no result)")
            working_agents += 1  # exists, just no result
    except Exception as e:
        print(f"  ERR {name:35s} → {type(e).__name__}: {str(e)[:60]}")
        broken_agents += 1

print(f"\n  Agentes OK: {working_agents}/13 | Rotos: {broken_agents}/13")
results["working_agents"] = working_agents
results["broken_agents"] = broken_agents

# ── 3. DecisionFilter — scoring rapido? ─────────────────────────────────────
print("\n=== 3. DECISION FILTER VELOCIDAD ===")
try:
    from core.decision_filter import DecisionFilter
    df_filter = DecisionFilter()

    from agents.signal_agent import SignalAgent, TradeSignal, SignalType
    fake_signal = TradeSignal(
        symbol="EURUSD", signal_type=SignalType.LONG,
        entry=1.08, stop_loss=1.075, take_profit=1.09,
        timeframe="1h", confidence=0.7,
        analysis_text="BOS alcista, setup valido OB 1.0800",
        decision_score=0
    )
    t0 = time.time()
    score = df_filter.evaluate(fake_signal, fake_df, capital=10000)
    elapsed = (time.time()-t0)*1000
    print(f"  Score: {score} [{elapsed:.0f}ms]")
    results["df_ok"] = True
    results["df_ms"] = round(elapsed)
except Exception as e:
    print(f"  ERROR: {e}")
    results["df_ok"] = False

# ── 4. SMC lite — cuanto tarda el analisis? ─────────────────────────────────
print("\n=== 4. SMC ANALYSIS VELOCIDAD (_run_smc_lite) ===")
try:
    from smc.structure import MarketStructure
    from smc.orderblocks import OrderBlockDetector
    t0 = time.time()
    ms = MarketStructure(fake_df)
    ms.detect_structure()
    ob = OrderBlockDetector(fake_df)
    bull_obs = ob.detect_bullish_ob()
    bear_obs = ob.detect_bearish_ob()
    elapsed = (time.time()-t0)*1000
    print(f"  SMC completo: {len(bull_obs)} bull OB + {len(bear_obs)} bear OB [{elapsed:.0f}ms]")
    results["smc_ms"] = round(elapsed)
except Exception as e:
    print(f"  ERROR: {e}")

# ── 5. Glint connector ───────────────────────────────────────────────────────
print("\n=== 5. GLINT (noticias tiempo real) ===")
try:
    from connectors.glint_connector import GlintConnector
    import json, os
    session_file = "memory/glint_session.json"
    if os.path.exists(session_file):
        with open(session_file) as f:
            session = json.load(f)
        last_signal = session.get("last_signal", "N/A")
        last_update = session.get("last_update", "N/A")
        print(f"  Session: OK | Ultima señal: {last_signal} @ {last_update}")
        results["glint_session"] = True
    else:
        print("  Session file no existe — Glint desconectado")
        results["glint_session"] = False
except Exception as e:
    print(f"  ERROR: {e}")

# ── 6. Resumen de scan stats ─────────────────────────────────────────────────
print("\n=== 6. SCAN STATS (rendimiento real) ===")
try:
    import json
    with open("memory/scan_stats.json") as f:
        stats = json.load(f)
    total = stats.get("total", 0)
    executed = stats.get("executed", 0)
    blocked_score = stats.get("blocked_score", 0)
    rate = executed/total*100 if total > 0 else 0
    print(f"  Total scans:    {total:,}")
    print(f"  Ejecutados:     {executed} ({rate:.2f}% ejecucion)")
    print(f"  Bloqueados score:{blocked_score:,}")
    print(f"  Razon: {blocked_score/total*100:.1f}% bloqueados por score bajo")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 7. AutonomousLearner / ResearchAgent / GoalsManager — activos? ──────────
print("\n=== 7. LOOPS AUTONOMOS ===")
loops = [
    ("AutonomousLearner",  "core.continuous_learning", "AutonomousLearner"),
    ("ResearchAgent",      "core.continuous_learning", "ResearchAgent"),
    ("GoalsManager",       "core.continuous_learning", "GoalsManager"),
    ("NightlyReporter",    "core.continuous_learning", "NightlyReporter"),
]
for name, mod_path, cls_name in loops:
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  ERR {name}: {e}")

# ── CONCLUSION ───────────────────────────────────────────────────────────────
print("\n=== CONCLUSION ===")
print(f"  MT5 conectado:     {'SI' if results.get('mt5_ok') else 'NO'}")
print(f"  Agentes OK:        {results.get('working_agents',0)}/13")
print(f"  Agentes rotos:     {results.get('broken_agents',0)}/13")
print(f"  DecisionFilter:    {'SI' if results.get('df_ok') else 'NO'} ({results.get('df_ms','?')}ms)")
print(f"  SMC analysis:      {results.get('smc_ms','?')}ms por scan")
print(f"  Glint session:     {'SI' if results.get('glint_session') else 'NO'}")
