"""
SMC Bot — Startup Script (modo no-interactivo)
Lee capital y modo desde .env, envía mensaje de bienvenida por Telegram.
Uso: python startup.py
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import config
from core.supervisor import TradingSupervisor


WELCOME_MSG = """SMC Bot iniciado en modo {mode}
Capital: ${capital:,.0f} (DEMO)
Riesgo max: {risk}% por trade
Pares crypto:  BTC ETH BNB SOL XRP ADA DOGE AVAX
Pares forex:   EURUSD GBPUSD XAUUSD US30 NAS100 USOIL USDJPY GBPJPY
DecisionFilter: <60=NO | 60-74=25% | 75-89=100% | 90+=PREMIUM
Comandos: /auto /semi /pause /status /scores /risk /positions
Regla de oro: Si no hay setup claro, no se opera.
Listo para operar. La paciencia paga."""


async def send_welcome(supervisor: TradingSupervisor):
    mode = config.operation_mode.upper()
    capital = float(input("\nCapital inicial (USD) [1000]: ").strip() or "1000")
    supervisor.capital = capital
    supervisor.risk_manager.capital = capital

    msg = WELCOME_MSG.format(
        mode=mode,
        capital=capital,
        risk=config.max_risk_per_trade * 100,
    )
    print("\n" + "="*55)
    print(msg)
    print("="*55 + "\n")
    await supervisor.telegram.send_glint_alert(msg)


async def main():
    # Validate required credentials
    missing = []
    if not config.anthropic_api_key or "PEGA" in config.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if not config.telegram_bot_token or "PEGA" in config.telegram_bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not config.telegram_chat_id or "PEGA" in config.telegram_chat_id:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        print(f"\n[ERROR] Faltan credenciales en .env: {', '.join(missing)}")
        print("Sigue la guia en .env para configurarlas.\n")
        sys.exit(1)

    print("\n[OK] Credenciales validadas")
    print(f"[OK] Modo: {config.operation_mode.upper()}")
    print(f"[OK] Testnet Binance: {config.binance_testnet}")
    print(f"[OK] MT5 Demo: {config.mt5_demo}")

    supervisor = TradingSupervisor(capital=1000.0)
    await send_welcome(supervisor)
    await supervisor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot detenido.")
