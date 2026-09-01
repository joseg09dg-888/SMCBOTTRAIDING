module.exports = {
  apps: [{
    name: 'smc-bot',
    script: 'startup.py',
    interpreter: 'C:\\\\Users\\\\JOSÉ\\\\SMCBOTTRAIDING\\\\.venv\\\\Scripts\\\\python.exe',
    args: '--auto',
    cwd: 'C:\\Users\\JOSÉ\\SMCBOTTRAIDING',
    restart_delay: 10000,
    max_restarts: 99,
    watch: ['core', 'agents', 'connectors', 'strategies', 'smc', 'dashboard', 'execution', 'startup.py'],
    ignore_watch: ['memory', '__pycache__', '*.pyc', '.git', 'logs', '*.log', '*.db', '*.db-shm', '*.db-wal', 'tests', '.venv'],
    watch_options: {
      followSymlinks: false,
      usePolling: true,
      interval: 2000
    },
    env: {
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8',
      // BUG-PYCACHE-RESTART-STORM (2026-09-01, live audit): ignore_watch's
      // '__pycache__'/'*.pyc' entries do not match nested paths like
      // core\__pycache__\supervisor.cpython-314.pyc on Windows -- confirmed
      // via C:\Users\JOSÉ\.pm2\pm2.log ("Change detected on path
      // core\__pycache__\..." causing a second unnecessary restart right
      // after every real code-triggered restart). Disabling bytecode
      // writes removes the false-positive trigger at the root instead of
      // trying to perfect an already-proven-unreliable glob pattern.
      PYTHONDONTWRITEBYTECODE: '1'
    }
  }]
}

