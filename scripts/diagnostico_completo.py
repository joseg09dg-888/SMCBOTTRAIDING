"""Diagnóstico completo del SMC Trading Bot."""
import sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

print('=== DIAGNÓSTICO REAL ===\n')

modulos = {
    'Config':           'core.config',
    'Supervisor':       'core.supervisor',
    'Risk Manager':     'core.risk_manager',
    'SMC Structure':    'smc.structure',
    'Order Blocks':     'smc.orderblocks',
    'Volume Profile':   'smc.volume_profile',
    'Analysis Agent':   'agents.analysis_agent',
    'Signal Agent':     'agents.signal_agent',
    'Footprint':        'agents.footprint_agent',
    'Statistical Edge': 'agents.statistical_edge_agent',
    'Binance':          'connectors.binance_connector',
    'MT5':              'connectors.metatrader_connector',
    'Glint':            'connectors.glint_browser',
    'Telegram':         'dashboard.telegram_commander',
    'FTMO':             'strategies.ftmo_agent',
    'Pairs Trading':    'strategies.pairs_trading',
    'Event Driven':     'strategies.event_driven',
    'Smart Execution':  'execution.smart_execution',
}

ok, fail = [], []
for name, mod in modulos.items():
    try:
        __import__(mod)
        print(f'OK  {name}')
        ok.append(name)
    except Exception as e:
        print(f'ERR {name}: {str(e)[:80]}')
        fail.append((name, str(e)[:80]))

print(f'\n{len(ok)}/{len(modulos)} modulos OK')
if fail:
    print('\nROTOS:')
    for n, e in fail:
        print(f'  {n}: {e}')

# Test Binance real data
print('\n=== TEST BINANCE REAL ===')
try:
    from connectors.binance_connector import BinanceConnector
    from core.config import config
    b = BinanceConnector(config.binance_api_key, config.binance_api_secret, True)
    df = b.get_ohlcv('BTCUSDT', '1h', 3)
    if not df.empty:
        print(f'OK Binance: BTC={float(df.close.iloc[-1]):,.2f} ({len(df)} velas)')
    else:
        print('ERR Binance: sin datos')
except Exception as e:
    print(f'ERR Binance: {e}')

# Test SMC pipeline
print('\n=== TEST SMC PIPELINE ===')
try:
    from connectors.binance_connector import BinanceConnector
    from core.config import config
    from smc.structure import MarketStructure
    from smc.orderblocks import OrderBlockDetector, FVGDetector
    from agents.signal_agent import SignalAgent

    b = BinanceConnector(config.binance_api_key, config.binance_api_secret, True)
    df = b.get_ohlcv('BTCUSDT', '1h', 200)
    ms = MarketStructure(df)
    struct = ms.analyze()
    bos = ms.detect_bos()
    bull_obs = OrderBlockDetector(df).find_bullish_obs()
    bear_obs = OrderBlockDetector(df).find_bearish_obs()

    is_bull = struct.bias == 'bullish'
    is_bear = struct.bias == 'bearish'
    if not (is_bull or is_bear) and bos:
        d = bos[-1].get('direction','')
        if d == 'bullish': is_bull = True
        elif d == 'bearish': is_bear = True

    dw = 'bullish' if is_bull else ('bearish' if is_bear else 'neutral')
    at = f'{dw} trend {struct.structure_type.value}'
    if bos: at += ' BOS confirmado'
    if bull_obs or bear_obs: at += ' order block presente setup valido'

    cp = float(df['close'].iloc[-1])
    sig = SignalAgent(0.55).evaluate(at, 'BTCUSDT', '1h', cp, (bull_obs if is_bull else bear_obs)[:3])

    print(f'OK Structure: {struct.structure_type.value} bias={struct.bias}')
    print(f'OK BOS: {len(bos)} | OBs: {len(bull_obs)+len(bear_obs)}')
    print(f'OK Signal: {sig.signal_type.value} entry={sig.entry:.2f} conf={sig.confidence}')
except Exception as e:
    print(f'ERR Pipeline: {e}')

# Test DecisionFilter
print('\n=== TEST DECISION FILTER ===')
try:
    from core.decision_filter import DecisionFilter
    from core.risk_manager import RiskManager
    from core.config import config
    rm = RiskManager(config, 1000.0)
    df_filter = DecisionFilter(config, rm)
    print(f'OK DecisionFilter inicializado')
except Exception as e:
    print(f'ERR DecisionFilter: {e}')

# Test FTMO
print('\n=== TEST FTMO ===')
try:
    from strategies.ftmo_agent import FTMOAgent, ChallengeType
    agent = FTMOAgent()
    state = FTMOAgent.new_challenge(10000.0, ChallengeType.TWO_STEP)
    state = agent.record_trade(state, 50.0)
    income = agent.calculate_monthly_income(200000, 0.05, 0.90)
    print(f'OK FTMO: balance={state.current_balance} income={income["net_monthly"]}/mes')
except Exception as e:
    print(f'ERR FTMO: {e}')

print('\n=== RESUMEN ===')
print(f'Modulos: {len(ok)}/{len(modulos)} OK')
