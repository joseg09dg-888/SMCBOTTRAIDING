"""Add score logging to supervisor.py _market_scan_loop using line-based patch."""
import ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

content = open('core/supervisor.py', encoding='utf-8').read()

# Add import if missing
if 'from core.score_db import save_score' not in content:
    content = content.replace(
        'logger = logging.getLogger(__name__)',
        'logger = logging.getLogger(__name__)\nfrom core.score_db import save_score'
    )
    print("Added save_score import")

# Insert save_score after each scan result block using line-based approach
lines = content.splitlines(keepends=True)
new_lines = []
i = 0
inserted = 0

while i < len(lines):
    line = lines[i]
    new_lines.append(line)

    # After the "ejecutando trade DEMO" block in crypto scan (indented 28 spaces)
    # Look for the pattern: "await self._execute_demo_trade(signal)" followed by nothing
    stripped = line.strip()

    # Detect the crypto scan execute block (indented with 28 spaces = 7 levels)
    if ('await self._execute_demo_trade(signal)' in line or
            'self._dispatch(signal)' in line):
        indent_size = len(line) - len(line.lstrip())
        # Add save_score after this line if next line doesn't already have it
        next_line = lines[i+1] if i+1 < len(lines) else ''
        if 'save_score' not in next_line and indent_size >= 28:
            # Build matching indent
            indent = ' ' * indent_size
            # Figure out signal direction
            save_block = (
                f"{indent}save_score(\n"
                f"{indent}    symbol=symbol, timeframe=tf if 'tf' in dir() else tf_label if 'tf_label' in dir() else '',\n"
                f"{indent}    score=score, direction=signal.signal_type.value,\n"
                f"{indent}    entry=signal.entry,\n"
                f"{indent}    sl=signal.stop_loss if signal.stop_loss else 0.0,\n"
                f"{indent}    tp=signal.take_profit,\n"
                f"{indent}    executed='execute_demo' in repr(self._execute_demo_trade),\n"
                f"{indent})\n"
            )
            # Only add once, skip if already there
            new_lines.append(save_block)
            inserted += 1
    i += 1

if inserted > 0:
    content = ''.join(new_lines)
    print(f"Added {inserted} save_score calls")
else:
    print("No insertions made via line scan — using simpler approach")

# Simpler: replace each "await self._execute_demo_trade(signal)" with save_score after
# Only in the scan loop context
import re

def add_save_after_execute(m):
    full = m.group(0)
    indent = m.group(1)
    return (full +
            f'\n{indent}save_score('
            f'symbol=symbol, timeframe=tf, score=score,'
            f' direction=signal.signal_type.value,'
            f' entry=signal.entry,'
            f' sl=signal.stop_loss if signal.stop_loss else 0.0,'
            f' tp=signal.take_profit, executed=True)')

# Actually use a targeted approach - add save_score at the end of each try block
# that contains a score calculation
# Simplest: find all places where "score = signal.decision_score" is set
# and add save_score after the if/elif/else block

# Find the crypto scan try block and add save_score at its end
old_crypto = (
    '                            if signal.signal_type == SignalType.WAIT or score < threshold:\n'
    '                                print(" â sin setup")\n'
    '                            elif self.demo_mode:\n'
    '                                print(f" â ejecutando trade DEMO")\n'
    '                                await self._execute_demo_trade(signal)\n'
    '                            else:\n'
    '                                print(f" â ejecutando trade")\n'
    '                                self._dispatch(signal)'
)

# Actually just read the exact bytes from the file and match
raw = open('core/supervisor.py', 'rb').read()
print(f"File bytes around line 509-514: {repr(raw[raw.find(b'sin setup'):raw.find(b'sin setup')+100])}")

# Find the exact text
idx = content.find('sin setup')
if idx > 0:
    ctx = content[max(0,idx-200):idx+400]
    print(f"\nContext: {repr(ctx[:300])}")

ast.parse(content)
print("Syntax OK")
open('core/supervisor.py', 'w', encoding='utf-8').write(content)
print("supervisor.py written")
