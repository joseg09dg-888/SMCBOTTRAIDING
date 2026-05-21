"""Add /axi command to telegram_commander.py."""
import ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')
content = open('dashboard/telegram_commander.py', encoding='utf-8').read()

if '"/axi"' not in content:
    content = content.replace(
        '"/ftmo":             "Estado FTMO challenge y criterios para cuentas fondeadas",',
        '"/ftmo":             "Estado FTMO challenge y criterios para cuentas fondeadas",\n'
        '    "/axi":              "Estado AXI SELECT - programa fondeo GRATIS hasta $1M",',
    )
    content = content.replace(
        '"/ftmo":             self._cmd_ftmo,',
        '"/ftmo":             self._cmd_ftmo,\n'
        '            "/axi":              self._cmd_axi,',
    )
    ast.parse(content)
    open('dashboard/telegram_commander.py', 'w', encoding='utf-8').write(content)
    print('Added /axi to COMMANDS and handlers')
else:
    print('/axi already present')
