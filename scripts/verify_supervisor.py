import ast
content = open('core/supervisor.py', encoding='utf-8').read()
ast.parse(content)
print('supervisor.py: OK')
checks = [
    ('MT5Connector import',    'from connectors.metatrader_connector import MT5Connector'),
    ('MT5_SYMBOLS',            'MT5_SYMBOLS'),
    ('_mt5_available',         '_mt5_available = False'),
    ('_scan_mt5_symbol method','async def _scan_mt5_symbol'),
    ('MT5 startup check',      'mt5_ok = await loop.run_in_executor'),
    ('MT5 scan block',         'MT5 forex/indices scan'),
    ('_execute_demo_trade',    'async def _execute_demo_trade'),
]
for name, snippet in checks:
    status = 'OK' if snippet in content else 'MISSING'
    print(f'  {status}: {name}')
