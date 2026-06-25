import json, sys
sys.path.insert(0, '.')

path = "memory/risk_governor_state.json"
with open(path, encoding='utf-8') as f:
    state = json.load(f)

for sym in ["EURUSD", "AUDUSD", "GBPUSD"]:
    removed = state["suspended"].pop(sym, None)
    if removed:
        print("Desbloqueado:", sym)

state["risk_multiplier"] = 1.0

with open(path, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2, ensure_ascii=False)

print("Suspendidos restantes:", list(state["suspended"].keys()))
print("risk_multiplier: 1.0")
