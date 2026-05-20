"""Add missing constants to supervisor.py cleanly."""
import os, ast
os.chdir(r'C:\Users\jose-\projects\trading_agent')

content = open('core/supervisor.py', encoding='utf-8').read()
print(f"File: {len(content)} chars")

# Find and replace the constants block
old_demo = 'DEMO_SCORE_THRESHOLD = 35'
new_block = (
    'DEMO_SCORE_THRESHOLD = 30\n'
    'DEMO_MAX_POSITIONS   = 5\n'
    'SCAN_INTERVAL_SEC    = 30\n'
    '\n'
    '# Symbols and timeframes to scan\n'
    'SCAN_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]\n'
    'SCAN_TIMEFRAMES = ["5m", "15m", "1h", "4h"]\n'
    '\n'
    '# MT5 forex/indices symbols\n'
    'MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]\n'
    'MT5_TIMEFRAMES   = ["H1", "H4"]\n'
    'MT5_MIN_VOLUME   = 0.01\n'
    '\n'
    '# yfinance forex (fallback when MT5 unavailable)\n'
    'YFINANCE_FOREX   = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",\n'
    '                    "USDJPY": "USDJPY=X", "GBPJPY": "GBPJPY=X"}\n'
)

# Find where the old DEMO threshold is
idx = content.find(old_demo)
if idx > 0:
    # Find end of the old block (up to the class definition)
    block_end = content.find('\nclass DemoTrade', idx)
    if block_end > 0:
        old_block = content[idx:block_end]
        print(f"Old block: {repr(old_block[:100])}")
        content = content[:idx] + new_block + content[block_end:]
        print("Constants block replaced")
    else:
        # Just replace the threshold line
        content = content.replace(old_demo, 'DEMO_SCORE_THRESHOLD = 30')
        print("Threshold replaced (block end not found)")
else:
    print(f"WARNING: '{old_demo}' not found in file")
    print("Content around line 22:")
    for i, l in enumerate(content.splitlines()[20:30], 21):
        print(f"  {i}: {repr(l[:60])}")

# Add SCAN_INTERVAL_SEC if missing
if 'SCAN_INTERVAL_SEC' not in content:
    content = content.replace(
        'DEMO_MAX_POSITIONS   = 5\n',
        'DEMO_MAX_POSITIONS   = 5\nSCAN_INTERVAL_SEC    = 30\n'
    )
    print("SCAN_INTERVAL_SEC added")

ast.parse(content)
open('core/supervisor.py', 'w', encoding='utf-8').write(content)
print("Written. Checks:")
print(f"  SCAN_INTERVAL_SEC: {'SCAN_INTERVAL_SEC' in content}")
print(f"  SCAN_SYMBOLS ADAUSDT: {'ADAUSDT' in content}")
print(f"  MT5_SYMBOLS: {'MT5_SYMBOLS' in content}")
print(f"  YFINANCE_FOREX: {'YFINANCE_FOREX' in content}")
print(f"  save_score: {'save_score' in content}")
print(f"  _scan_forex_yfinance: {'_scan_forex_yfinance' in content}")
print("supervisor.py: SYNTAX OK")
