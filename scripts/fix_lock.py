"""Fix startup.py lock mechanism — only kills Python processes."""
import ast, os, re
os.chdir(r'C:\Users\jose-\projects\trading_agent')

content = open('startup.py', encoding='utf-8').read()

old = (
    'def _acquire_lock():\n'
    '    if LOCK_FILE.exists():\n'
    '        try:\n'
    '            old_pid = int(LOCK_FILE.read_text().strip())\n'
    '            import psutil\n'
    '            psutil.Process(old_pid).kill()\n'
    '            print(f"[Bot] Instancia anterior (PID {old_pid}) terminada")\n'
    '        except Exception:\n'
    '            pass\n'
    '        LOCK_FILE.unlink(missing_ok=True)\n'
    '    LOCK_FILE.write_text(str(os.getpid()))\n'
    '    atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True) if LOCK_FILE.exists() else None)'
)

new = (
    'def _acquire_lock():\n'
    '    if LOCK_FILE.exists():\n'
    '        try:\n'
    '            old_pid = int(LOCK_FILE.read_text().strip())\n'
    '            if old_pid != os.getpid():\n'
    '                import psutil\n'
    '                p = psutil.Process(old_pid)\n'
    '                pname = (p.name() or "").lower()\n'
    '                # Only kill if it is actually a Python/bot process\n'
    '                if "python" in pname:\n'
    '                    p.kill()\n'
    '                    print(f"[Bot] Instancia anterior (PID {old_pid}) terminada")\n'
    '        except Exception:\n'
    '            pass\n'
    '        LOCK_FILE.unlink(missing_ok=True)\n'
    '    LOCK_FILE.write_text(str(os.getpid()))\n'
    '    atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True) if LOCK_FILE.exists() else None)'
)

if old in content:
    content = content.replace(old, new)
    ast.parse(content)
    open('startup.py', 'w', encoding='utf-8').write(content)
    print('Lock mechanism fixed')
else:
    # Show what we have
    idx = content.find('def _acquire_lock')
    if idx >= 0:
        print('Current _acquire_lock:')
        print(repr(content[idx:idx+500]))
    else:
        print('Function not found')
