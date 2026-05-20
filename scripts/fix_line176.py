"""Fix remaining syntax error at line 176 of supervisor.py."""
import os, ast
os.chdir(r'C:\Users\jose-\projects\trading_agent')

raw = open('core/supervisor.py', 'rb').read()
lines = raw.split(b'\n')
line176 = lines[175]
print(f"Line 176 hex: {line176.hex()}")
print(f"Line 176 repr: {repr(line176)}")

# The issue: after replacing right-quote bytes, some string literals got broken
# Fix: replace any remaining non-ASCII bytes in string literals with safe ASCII

# First, fix the specific known pattern: â + random byte + " -> -
# â = \xc3\xa2, followed by anything, followed by " = makes broken string
# Replace pattern \xc3\xa2...  with -

fixed = raw

# Replace â (U+00E2, \xc3\xa2) which is garbage from encoding corruption
fixed = fixed.replace(b'\xc3\xa2', b'-')

# Replace remaining euro sign (U+20AC, \xe2\x82\xac) if any
fixed = fixed.replace(b'\xe2\x82\xac', b'-')

# Replace any remaining UTF-8 multi-byte sequences that aren't valid Python identifiers
# in string literals: \xc3-\xc7 range (Latin extended chars)
import re

# More targeted: fix the NO TRADE line specifically
# Find all occurrences of patterns like "NO TRADE" followed by junk
old_pattern = b'NO TRADE \xc3\xa2'
new_pattern = b'NO TRADE -'
if old_pattern in fixed:
    fixed = fixed.replace(old_pattern, new_pattern)
    print("Fixed NO TRADE line")

# Also fix arrow patterns: â†' (U+2192 right arrow)
# â†' = \xe2\x86\x92 — keep as ->
fixed = fixed.replace(b'\xe2\x86\x92', b'->')
# â†" (left arrow \xe2\x86\x90) -> <-
fixed = fixed.replace(b'\xe2\x86\x90', b'<-')

open('core/supervisor.py', 'wb').write(fixed)
print("Written")

try:
    ast.parse(open('core/supervisor.py', encoding='utf-8').read())
    print("supervisor.py: SYNTAX OK")
except SyntaxError as e:
    print(f"BROKEN line {e.lineno}: {e.msg}")
    lines2 = open('core/supervisor.py', encoding='utf-8', errors='replace').readlines()
    for i, l in enumerate(lines2[max(0,e.lineno-3):e.lineno+2], max(0,e.lineno-3)+1):
        print(f"  {i}: {repr(l[:100])}")
