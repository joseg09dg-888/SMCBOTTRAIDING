"""
SMC Bot — Startup Script
Uso manual:    python startup.py
Autoarranque:  python startup.py --auto --capital 1000 --reason auto_restart
"""
# Encoding fix: use environment variables (safe in all contexts)
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8']       = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import asyncio
import atexit
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── Process lock: garantiza que solo corra una instancia ─────────────────────
LOCK_FILE = Path(__file__).parent / "trading_bot.lock"

def _acquire_lock():
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if old_pid != os.getpid():
                import psutil
                p = psutil.Process(old_pid)
                pname = (p.name() or "").lower()
                # Only kill if it is actually a Python/bot process
                if "python" in pname:
                    p.kill()
                    print(f"[Bot] Instancia anterior (PID {old_pid}) terminada")
        except Exception:
            pass
        LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True) if LOCK_FILE.exists() else None)

try:
    _acquire_lock()
except Exception as e:
    print(f"[Bot] Lock warning: {e}")
# ─────────────────────────────────────────────────────────────────────────────

from core.config import config
from core.supervisor import TradingSupervisor
from core.wakeup_recovery import run_recovery


WELCOME_MSG = """<b>SMC Bot iniciado en modo {mode}</b>
Capital: ${capital:,.0f} (DEMO)
Riesgo max: {risk}% por trade
Pares crypto: BTC ETH BNB SOL XRP
Pares forex:  EURUSD GBPUSD XAUUSD USDJPY
Score: 35+=DEMO | 60+=REAL | 90+=PREMIUM
Comandos: /auto /semi /pause /status /scores
Regla: Si no hay setup claro, no se opera.
Listo para operar."""

RESTART_REASONS = {
    "auto_restart": "Reinicio automático de Windows",
    "watchdog":     "Reiniciado por watchdog (proceso caído)",
    "crash":        "Reiniciado tras error inesperado",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMC Trading Bot")
    parser.add_argument("--auto",    action="store_true",   help="Modo no-interactivo (autoarranque)")
    parser.add_argument("--capital", type=float, default=0.0, help="Capital inicial en USD (0=auto desde MT5)")
    parser.add_argument("--reason",  type=str,  default="",      help="Motivo del arranque para Telegram")
    return parser.parse_args()


async def send_welcome(supervisor: TradingSupervisor, capital: float, auto: bool, reason: str):
    mode = config.operation_mode.upper()

    if not auto:
        try:
            raw = input("\nCapital inicial (USD) [auto desde MT5]: ").strip()
            capital = float(raw) if raw else 0.0
        except (EOFError, ValueError):
            pass

    # Auto-detect capital from MT5 balance when 0
    if capital <= 0:
        try:
            mt5_info = supervisor.mt5.get_account_info()
            balance = mt5_info.get("balance", 0.0)
            if balance > 0:
                capital = balance
                print(f"[OK] Capital detectado desde MT5: ${capital:,.2f}", flush=True)
            else:
                capital = 97_000.0  # fallback conservador: balance real Axi (~$97K)
                print(f"[WARN] No se pudo leer balance MT5, usando ${capital:,.0f}", flush=True)
        except Exception as e:
            capital = 97_000.0  # fallback conservador: balance real Axi (~$97K)
            print(f"[WARN] Error leyendo MT5 balance: {e} — usando ${capital:,.0f}", flush=True)

    supervisor.capital = capital
    supervisor.risk_manager.capital = capital
    supervisor._edge.capital = capital

    if auto and reason:
        reason_label = RESTART_REASONS.get(reason, reason)
        restart_header = f"🔄 *Bot reiniciado automáticamente — todo OK*\nMotivo: {reason_label}\n\n"
    else:
        restart_header = ""

    msg = restart_header + WELCOME_MSG.format(
        mode=mode,
        capital=capital,
        risk=config.max_risk_per_trade * 100,
    )
    print("\n" + "="*55)
    print(msg)
    print("="*55 + "\n")
    await supervisor.telegram.send_glint_alert(msg)

    if auto and reason in ("auto_restart", "watchdog", "crash"):
        recovery_msg = await run_recovery(supervisor.telegram, capital, mt5_connector=supervisor.mt5)
        if recovery_msg:
            print(recovery_msg)
            await supervisor.telegram.send_glint_alert(recovery_msg)


async def main():
    args = _parse_args()

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
    if args.auto:
        print(f"[OK] Autoarranque: capital=${args.capital:,.0f}")

    supervisor = TradingSupervisor(capital=args.capital)
    await send_welcome(supervisor, args.capital, args.auto, args.reason)
    await supervisor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot detenido.")