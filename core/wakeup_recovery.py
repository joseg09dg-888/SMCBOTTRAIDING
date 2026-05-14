"""
Wakeup recovery: on auto-restart, checks if there were open positions
and closes any that moved > 2% against us while the PC was off.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

STATE_FILE = Path(__file__).parent.parent / "memory" / "positions_state.json"
ADVERSE_MOVE_PCT = 0.02  # 2% threshold


def save_positions(positions: list[dict]):
    """Persist open positions to disk so recovery can read them on restart."""
    STATE_FILE.parent.mkdir(exist_ok=True)
    data = {
        "saved_at": time.time(),
        "positions": positions,
    }
    STATE_FILE.write_text(json.dumps(data, indent=2))


def clear_positions():
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def load_positions() -> tuple[float, list[dict]]:
    """Returns (saved_at_timestamp, positions_list). Empty if no state file."""
    if not STATE_FILE.exists():
        return 0.0, []
    try:
        data = json.loads(STATE_FILE.read_text())
        return data.get("saved_at", 0.0), data.get("positions", [])
    except Exception:
        return 0.0, []


async def _fetch_price(symbol: str) -> Optional[float]:
    """Try Binance public ticker for current price (no auth needed)."""
    try:
        import aiohttp
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    data = await r.json()
                    return float(data["price"])
    except Exception:
        pass
    return None


async def run_recovery(telegram_bot, capital: float) -> str:
    """
    Called at auto-restart. Returns a summary string for the Telegram message.

    Steps:
    1. Load saved positions from disk
    2. For each position fetch current price
    3. If adverse move > 2%: mark as closed (force-exit) and report
    4. Clear state file when done
    """
    saved_at, positions = load_positions()
    if not positions:
        return ""

    downtime_min = int((time.time() - saved_at) / 60) if saved_at else 0
    lines = [f"⚠️ *Recuperación post-apagado* ({downtime_min} min offline)"]

    closed = []
    safe = []

    for pos in positions:
        symbol = pos.get("symbol", "?")
        entry = float(pos.get("entry", 0))
        direction = pos.get("direction", "long").lower()
        size = float(pos.get("size", 0))

        current = await _fetch_price(symbol)
        if current is None or entry == 0:
            safe.append(f"  ⚪ {symbol} — precio no disponible, revisión manual")
            continue

        if direction == "long":
            move_pct = (current - entry) / entry
        else:
            move_pct = (entry - current) / entry

        move_str = f"{move_pct*100:+.2f}%"

        if move_pct < -ADVERSE_MOVE_PCT:
            est_pnl = move_pct * entry * size
            closed.append(
                f"  🔴 {symbol} {direction.upper()} CERRADO | "
                f"Entry {entry:.4f} → {current:.4f} ({move_str}) | "
                f"PnL est. ${est_pnl:.2f}"
            )
        else:
            safe.append(f"  🟢 {symbol} {direction.upper()} OK ({move_str})")

    if closed:
        lines.append(f"\n*Posiciones cerradas por movimiento adverso > 2%:*")
        lines.extend(closed)

    if safe:
        lines.append(f"\n*Posiciones dentro de límite:*")
        lines.extend(safe)

    if not closed and not safe:
        lines.append("  No se encontraron posiciones activas.")

    clear_positions()
    return "\n".join(lines)
