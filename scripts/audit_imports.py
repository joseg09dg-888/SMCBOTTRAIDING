import sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv
load_dotenv()

modules = [
    'core.config', 'core.supervisor',
    'core.risk_manager', 'core.learning_engine',
    'agents.analysis_agent', 'agents.signal_agent',
    'agents.footprint_agent', 'agents.statistical_edge_agent',
    'connectors.binance_connector', 'connectors.metatrader_connector',
    'dashboard.telegram_commander', 'dashboard.telegram_bot',
    'strategies.pairs_trading', 'strategies.event_driven',
    'execution.smart_execution', 'backtesting.lean_backtest',
    'deployment.cloud_setup', 'deployment.health_monitor',
]
ok, fail = [], []
for m in modules:
    try:
        __import__(m)
        print(f'OK  {m}')
        ok.append(m)
    except Exception as e:
        print(f'ERR {m}: {e}')
        fail.append((m, str(e)))
print(f'\n{len(ok)}/{len(modules)} OK')
if fail:
    print('BROKEN:')
    for m, e in fail:
        print(f'  {m}: {e}')
