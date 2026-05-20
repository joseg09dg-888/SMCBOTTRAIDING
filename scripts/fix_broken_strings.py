"""Fix broken string literals caused by smart-quote replacements in supervisor.py."""
import os, ast
os.chdir(r'C:\Users\jose-\projects\trading_agent')

content = open('core/supervisor.py', encoding='utf-8', errors='replace').read()

# Pattern: f"[FILTER] NO TRADE --" {signal.notes}")
# Should be: f"[FILTER] NO TRADE -- {signal.notes}")
# The spurious " came from replacing the right-quote that was part of the em-dash encoding

import re

# Fix any pattern where a closing quote appeared mid-string before a {var}
# Look for: --" {   or  --- " {
# Replace the misplaced quote
fixes = [
    ('--" {signal.notes}")', '-- {signal.notes}")'),
    ('-- " {signal.notes}")', '-- {signal.notes}")'),
    # Common pattern: text --" {var}"  -> text -- {var}"
]
for bad, good in fixes:
    if bad in content:
        content = content.replace(bad, good)
        print(f"Fixed: {bad!r} -> {good!r}")

# More general: fix any f-string where a quote appears before {var}
# Pattern: "...--" {identifier}...
def fix_fstring_quotes(text):
    # Find patterns like --" {word} and replace with -- {word}
    pattern = r'--"(\s+\{[^}]+\})'
    replacement = r'--\1'
    fixed, count = re.subn(pattern, replacement, text)
    if count: print(f"Fixed {count} f-string quote issues via regex")
    return fixed

content = fix_fstring_quotes(content)

# Also fix arrow patterns in print statements
content = content.replace('â†\x92', '->').replace('â\x80\x93', '-').replace('â\x80\x94', '-')

open('core/supervisor.py', 'w', encoding='utf-8').write(content)

try:
    ast.parse(open('core/supervisor.py', encoding='utf-8').read())
    print("supervisor.py: SYNTAX OK")
except SyntaxError as e:
    print(f"BROKEN line {e.lineno}: {e.msg}")
    lines = open('core/supervisor.py', encoding='utf-8', errors='replace').readlines()
    for i, l in enumerate(lines[max(0,e.lineno-3):e.lineno+3], max(0,e.lineno-3)+1):
        print(f"  {i}: {repr(l[:100])}")
