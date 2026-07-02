module.exports = {
  apps: [{
    name: 'smc-bot',
    script: 'startup.py',
    interpreter: 'C:\\\\Users\\\\jose-\\\\projects\\\\trading_agent\\\\.venv\\\\Scripts\\\\python.exe',
    args: '--auto',
    cwd: 'C:\\Users\\jose-\\projects\\trading_agent',
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
      PYTHONIOENCODING: 'utf-8'
    }
  }]
}

