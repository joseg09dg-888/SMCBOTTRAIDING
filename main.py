import asyncio
import sys
from core.supervisor import TradingSupervisor
from core.config import config


def validate_config():
    required = [
        ("ANTHROPIC_API_KEY",  config.anthropic_api_key),
        ("TELEGRAM_BOT_TOKEN", config.telegram_bot_token),
        ("TELEGRAM_CHAT_ID",   config.telegram_chat_id),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        print(f"Faltan variables de entorno: {', '.join(missing)}")
        print("Copia .env.example como .env y llena los valores.")
        sys.exit(1)

    if config.binance_testnet:
        print("MODO DEMO — Binance Testnet activo")
    if config.mt5_demo:
        print("MODO DEMO — MetaTrader Demo activo")


if __name__ == "__main__":
    print("=" * 50)
    print("  SMC TRADING BOT")
    print("=" * 50)

    validate_config()

    capital = float(input("Capital inicial (USD) [1000]: ") or "1000")
    mode_input = input("Modo [1=Auto / 2=Semi-auto / Enter=Hybrid]: ").strip()
    mode_map = {"1": "auto", "2": "semi", "": "hybrid"}
    config.operation_mode = mode_map.get(mode_input, "hybrid")

    supervisor = TradingSupervisor(capital=capital)
    try:
        asyncio.run(supervisor.run())
    except KeyboardInterrupt:
        print("\nBot detenido.")
