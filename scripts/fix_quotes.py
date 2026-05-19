"""Fix smart quotes and other encoding issues in supervisor.py."""
import ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

path = 'core/supervisor.py'
raw = open(path, 'rb').read()

# Replace smart quotes (U+201C left ", U+201D right ") with straight quotes
raw = raw.replace('“'.encode('utf-8'), b'"')
raw = raw.replace('”'.encode('utf-8'), b'"')
# Replace en-dash (U+2013) with plain dash
raw = raw.replace('–'.encode('utf-8'), b'-')
# Replace em-dash (U+2014) with plain dash
raw = raw.replace('—'.encode('utf-8'), b'-')

open(path, 'wb').write(raw)

# Verify
try:
    ast.parse(open(path, encoding='utf-8').read())
    print("supervisor.py: syntax OK after smart-quote fix")
except SyntaxError as e:
    print(f"STILL BROKEN: {e}")
