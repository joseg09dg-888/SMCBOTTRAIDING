"""Copy SMCBotEA.mq5 to all MT5 Expert directories and compile."""
import shutil, os, glob, subprocess, sys, time

os.chdir(r'C:\Users\jose-\projects\trading_agent')
EA_SRC = 'mt5_ea/SMCBotEA.mq5'

# Find all MT5 Experts directories
paths = glob.glob(
    r'C:\Users\jose-\AppData\Roaming\MetaQuotes\Terminal\*\MQL5\Experts'
)
print(f'Found MT5 Experts dirs: {len(paths)}')
for p in paths: print(f'  {p}')

# Copy EA to each directory
copied = []
for path in paths:
    os.makedirs(path, exist_ok=True)
    dest = os.path.join(path, 'SMCBotEA.mq5')
    shutil.copy(EA_SRC, dest)
    print(f'Copied: {dest}')
    copied.append(dest)

if not copied:
    print('No MT5 directories found.')
    sys.exit(1)

# Try to compile using MetaEditor
# MetaEditor is installed alongside MetaTrader 5
editor_paths = [
    r'C:\Program Files\Axi MetaTrader 5 Terminal\metaeditor64.exe',
    r'C:\Program Files\MetaTrader 5\metaeditor64.exe',
    r'C:\Program Files (x86)\MetaTrader 5\metaeditor64.exe',
]

editor_exe = None
for p in editor_paths:
    if os.path.exists(p):
        editor_exe = p
        print(f'Found MetaEditor: {editor_exe}')
        break

if editor_exe:
    for dest in copied:
        print(f'Compiling {dest}...')
        result = subprocess.run(
            [editor_exe, '/compile', dest, '/log'],
            capture_output=True, text=True, timeout=60
        )
        print(f'  Exit code: {result.returncode}')
        if result.stdout: print(f'  stdout: {result.stdout[:200]}')
        if result.stderr: print(f'  stderr: {result.stderr[:200]}')

        # Check if .ex5 was created
        ex5 = dest.replace('.mq5', '.ex5')
        if os.path.exists(ex5):
            print(f'  Compiled: {ex5}')
        else:
            print(f'  No .ex5 found — may need manual compile in MT5')
else:
    print('MetaEditor not found — compile manually in MT5')

print()
print('=== NEXT STEPS IN AXI MT5 ===')
print('1. Press F5 in MT5 (or View → Navigator)')
print('2. Right-click "Expert Advisors" → Refresh')
print('3. Find "SMCBotEA" in the list')
print('4. If no .ex5: double-click to open in MetaEditor → F7 to compile')
print('5. Drag SMCBotEA onto EURUSD chart')
print('6. Click OK → green smiley appears = EA running')
