import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List

load_dotenv()

@dataclass
class Config:
    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Binance
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    binance_testnet: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    # MetaTrader
    mt5_login: int = int(os.getenv("MT5_LOGIN", "0"))
    mt5_password: str = os.getenv("MT5_PASSWORD", "")
    mt5_server: str = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    mt5_demo: bool = os.getenv("MT5_DEMO", "true").lower() == "true"

    # Glint
    glint_email: str = os.getenv("GLINT_EMAIL", "")
    glint_session_token: str = os.getenv("GLINT_SESSION_TOKEN", "")
    glint_ws_url: str = os.getenv("GLINT_WS_URL", "wss://glint.trade/ws")

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Riesgo
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.005"))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
    max_monthly_loss: float = float(os.getenv("MAX_MONTHLY_LOSS", "0.15"))
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

    # Modo operación: "auto" | "semi" | "alerts" | "hybrid"
    operation_mode: str = os.getenv("OPERATION_MODE", "hybrid")

    # Timeframes activos
    timeframes: List[str] = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_TIMEFRAMES", "1m,5m,15m,1h,4h,1d"
        ).split(",")
    )

    # Proyectos integrados
    music_project_api: str = os.getenv("MUSIC_PROJECT_API", "")
    marketing_platform_api: str = os.getenv("MARKETING_PLATFORM_API", "")


config = Config()
