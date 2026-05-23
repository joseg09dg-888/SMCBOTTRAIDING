"""Fix startup.py — remove TextIOWrapper, use reconfigure instead."""
import ast, os, re
os.chdir(r'C:\Users\jose-\projects\trading_agent')

content = open('startup.py', encoding='utf-8').read()
print(f"File: {len(content)} chars")

# Find and replace the encoding block
# Use regex to handle any corruption in the comment line
pattern = r'# Fix encoding[^\n]*\nimport sys, io, os\n.*?reconfigure\(encoding=.utf-8., errors=.replace.\) if hasattr\(sys\.stdout, .reconfigure.\) else None'

# Check what we actually have at the top
lines = content.splitlines()
for i, l in enumerate(lines[:20]):
    print(f"{i+1:2}: {repr(l[:80])}")

# Find the encoding block start/end
start_idx = None
end_idx = None
for i, l in enumerate(lines):
    if 'Fix encoding' in l and start_idx is None:
        start_idx = i
    if start_idx is not None and l.strip().startswith('import argparse') and end_idx is None:
        end_idx = i
        break

print(f"\nEncoding block: lines {start_idx+1} to {end_idx}")

if start_idx is not None and end_idx is not None:
    # Replace encoding block with clean version
    new_block = [
        '# Encoding fix: use environment variables (safe in all contexts)',
        'import sys, os',
        "os.environ['PYTHONIOENCODING'] = 'utf-8'",
        "os.environ['PYTHONUTF8']       = '1'",
        "if hasattr(sys.stdout, 'reconfigure'):",
        "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')",
        '',
    ]
    new_lines = lines[:start_idx] + new_block + lines[end_idx:]
    new_content = '\n'.join(new_lines)
    ast.parse(new_content)
    open('startup.py', 'w', encoding='utf-8').write(new_content)
    print('startup.py encoding fixed')
else:
    print('Encoding block not found — checking content')
