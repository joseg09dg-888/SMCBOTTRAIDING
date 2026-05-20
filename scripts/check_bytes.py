"""Check and fix supervisor.py bytes."""
import os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

raw = open('core/supervisor.py', 'rb').read()
lines = raw.split(b'\n')
line28 = lines[27]
print(f"Line 28 hex: {line28[:80].hex()}")
print(f"Line 28 repr: {repr(line28[:60])}")

# Fix: replace all multi-byte quote characters
fixes = [
    (b'\xe2\x80\x9c', b'"'),   # U+201C left double quote
    (b'\xe2\x80\x9d', b'"'),   # U+201D right double quote
    (b'\xe2\x80\x98', b"'"),   # U+2018 left single quote
    (b'\xe2\x80\x99', b"'"),   # U+2019 right single quote
    (b'\xe2\x80\x93', b'-'),   # U+2013 en dash
    (b'\xe2\x80\x94', b'-'),   # U+2014 em dash
    (b'\xef\xbf\xbd', b'?'),   # U+FFFD replacement char
]

fixed = raw
for bad, good in fixes:
    count = fixed.count(bad)
    if count > 0:
        print(f"Replacing {bad.hex()} ({count}x)")
        fixed = fixed.replace(bad, good)

open('core/supervisor.py', 'wb').write(fixed)
print("Written")

# Verify
import ast
try:
    ast.parse(open('core/supervisor.py', encoding='utf-8').read())
    print("supervisor.py: SYNTAX OK")
except SyntaxError as e:
    print(f"STILL BROKEN: line {e.lineno}: {e.msg}")
    lines2 = open('core/supervisor.py', encoding='utf-8', errors='replace').readlines()
    for i, l in enumerate(lines2[max(0,e.lineno-3):e.lineno+2], max(0,e.lineno-3)+1):
        print(f"  {i}: {repr(l[:80])}")
