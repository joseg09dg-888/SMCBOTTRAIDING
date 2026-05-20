module.exports = {
  apps: [{
    name: 'smc-bot',
    script: 'startup.py',
    interpreter: 'C:\\\\Users\\\\jose-\\\\projects\\\\trading_agent\\\\.venv\\\\Scripts\\\\python.exe',
    args: '--auto --capital 1000',
    cwd: 'C:\\Users\\jose-\\projects\\trading_agent',
    restart_delay: 10000,
    max_restarts: 99,
    watch: false,
    env: {
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8'
    }
  }]
}

