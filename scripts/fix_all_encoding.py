"""Fix smart quotes and encoding in all Python files."""
import ast, glob, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

FILES = [
    'dashboard/telegram_commander.py',
    'strategies/ftmo_agent.py',
]

for path in FILES:
    if not os.path.exists(path):
        print(f"SKIP (missing): {path}")
        continue
    raw = open(path, 'rb').read()
    orig = raw
    # Replace smart quotes with straight quotes
    raw = raw.replace('“'.encode('utf-8'), b'"')   # left "
    raw = raw.replace('”'.encode('utf-8'), b'"')   # right "
    raw = raw.replace('‘'.encode('utf-8'), b"'")   # left '
    raw = raw.replace('’'.encode('utf-8'), b"'")   # right '
    raw = raw.replace('–'.encode('utf-8'), b'-')   # en dash
    raw = raw.replace('—'.encode('utf-8'), b'-')   # em dash
    # Also fix replacement character
    raw = raw.replace(b'\xef\xbf\xbd', b'?')           # U+FFFD
    if raw != orig:
        open(path, 'wb').write(raw)
        print(f"Fixed: {path}")
    else:
        print(f"Clean: {path}")
    # Verify
    try:
        ast.parse(open(path, encoding='utf-8').read())
        print(f"  Syntax OK")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR line {e.lineno}: {e.msg}")
