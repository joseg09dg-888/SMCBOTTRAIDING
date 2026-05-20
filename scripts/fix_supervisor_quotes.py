"""Fix smart quotes in supervisor.py."""
import ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')
raw = open('core/supervisor.py', 'rb').read()
raw = raw.replace('"'.encode('utf-8'), b'"')
raw = raw.replace('"'.encode('utf-8'), b'"')
raw = raw.replace("'".encode('utf-8'), b"'")
raw = raw.replace("'".encode('utf-8'), b"'")
raw = raw.replace('–'.encode('utf-8'), b'-')
raw = raw.replace('—'.encode('utf-8'), b'-')
open('core/supervisor.py', 'wb').write(raw)
try:
    ast.parse(open('core/supervisor.py', encoding='utf-8').read())
    print("supervisor.py: OK")
except SyntaxError as e:
    print(f"STILL BROKEN line {e.lineno}: {e.msg}")
    lines = open('core/supervisor.py', encoding='utf-8').readlines()
    if e.lineno:
        for i, l in enumerate(lines[max(0,e.lineno-3):e.lineno+2], max(0,e.lineno-3)+1):
            print(f"  {i}: {repr(l[:80])}")
