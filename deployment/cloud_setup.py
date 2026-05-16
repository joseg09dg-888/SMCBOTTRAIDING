"""Cloud deployment utilities: Dockerfile, docker-compose, PM2, validation."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class DeploymentConfig:
    bot_name: str = "smc-trading-bot"
    python_version: str = "3.12"
    restart_policy: str = "always"
    capital: float = 1000.0
    log_dir: str = "logs"
    memory_dir: str = "memory"
    reports_dir: str = "reports"


REQUIRED_ENV_KEYS = [
    "ANTHROPIC_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
]

KEY_PACKAGES = ["anthropic", "python-telegram-bot", "python-binance", "pandas", "numpy"]


class CloudSetup:
    def __init__(self, project_root: str = None):
        self.root = Path(project_root or ".")
        self.config = DeploymentConfig()

    def generate_dockerfile(self) -> str:
        return (
            f"FROM python:{self.config.python_version}-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "ENV PYTHONUNBUFFERED=1\n"
            "ENV PYTHONIOENCODING=utf-8\n"
            'HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \\\n'
            '  CMD python -c "import os; exit(0 if os.path.exists(\'trading_bot.lock\') else 1)"\n'
            'CMD ["python", "-u", "startup.py", "--auto", "--capital", "1000"]\n'
        )

    def generate_docker_compose(self) -> str:
        return (
            "version: '3.8'\n"
            "services:\n"
            f"  {self.config.bot_name}:\n"
            "    build: .\n"
            f"    restart: {self.config.restart_policy}\n"
            "    env_file: .env\n"
            "    volumes:\n"
            f"      - ./{self.config.memory_dir}:/app/{self.config.memory_dir}\n"
            f"      - ./{self.config.log_dir}:/app/{self.config.log_dir}\n"
            f"      - ./{self.config.reports_dir}:/app/{self.config.reports_dir}\n"
            "    healthcheck:\n"
            '      test: ["CMD", "python", "-c", '
            '"import os; exit(0 if os.path.exists(\'trading_bot.lock\') else 1)"]\n'
            "      interval: 30s\n"
            "      timeout: 10s\n"
            "      retries: 3\n"
            "      start_period: 60s\n"
        )

    def save_dockerfile(self) -> str:
        path = self.root / "Dockerfile"
        path.write_text(self.generate_dockerfile(), encoding="utf-8")
        return str(path)

    def save_docker_compose(self) -> str:
        path = self.root / "docker-compose.yml"
        path.write_text(self.generate_docker_compose(), encoding="utf-8")
        return str(path)

    def validate_env_file(self) -> dict:
        env_path = self.root / ".env"
        if not env_path.exists():
            return {"valid": False, "missing": list(REQUIRED_ENV_KEYS), "present": []}
        try:
            text = env_path.read_text(encoding="utf-8")
            present = [k for k in REQUIRED_ENV_KEYS if f"{k}=" in text and f"{k}=PEGA" not in text]
            missing = [k for k in REQUIRED_ENV_KEYS if k not in present]
            return {"valid": len(missing) == 0, "missing": missing, "present": present}
        except Exception as e:
            return {"valid": False, "missing": list(REQUIRED_ENV_KEYS), "present": [], "error": str(e)}

    def validate_requirements(self) -> dict:
        req_path = self.root / "requirements.txt"
        if not req_path.exists():
            return {"valid": False, "missing": KEY_PACKAGES}
        try:
            text = req_path.read_text(encoding="utf-8").lower()
            missing = [p for p in KEY_PACKAGES if p.lower().split("-")[-1] not in text and p.lower() not in text]
            return {"valid": len(missing) == 0, "missing": missing}
        except Exception as e:
            return {"valid": False, "missing": KEY_PACKAGES, "error": str(e)}

    def generate_pm2_config(self) -> str:
        return (
            "module.exports = {\n"
            "  apps: [{\n"
            "    name: 'smc-bot',\n"
            "    script: 'startup.py',\n"
            "    interpreter: '.venv/Scripts/python',\n"
            "    args: '--auto --capital 1000',\n"
            "    restart_delay: 5000,\n"
            "    max_restarts: 99,\n"
            "    env: { PYTHONUNBUFFERED: '1' }\n"
            "  }]\n"
            "}\n"
        )

    def get_deployment_checklist(self) -> list:
        items = [
            {
                "name": ".env file",
                "status": (self.root / ".env").exists(),
                "description": "Required credentials file",
            },
            {
                "name": "requirements.txt",
                "status": (self.root / "requirements.txt").exists(),
                "description": "Python dependencies list",
            },
            {
                "name": "logs/ directory",
                "status": (self.root / "logs").exists(),
                "description": "Log output directory",
            },
            {
                "name": "memory/ directory",
                "status": (self.root / "memory").exists(),
                "description": "Persistent memory directory",
            },
            {
                "name": "startup.py",
                "status": (self.root / "startup.py").exists(),
                "description": "Bot entry point",
            },
            {
                "name": "trading_bot.lock",
                "status": True,  # Will be created at runtime
                "description": "Process lock (created at runtime)",
            },
        ]
        return items
