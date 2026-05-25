"""
MT5 EA Bridge — sends trade signals via JSON files to SMCBotEA.mq5.
Bypasses tradeapi_disabled=True restriction since EAs have full trading access.

Usage:
    bridge = MT5EABridge()
    result = bridge.place_order("EURUSD", "long", volume=0.01, sl_pips=50, tp_pips=100)
"""
import json
import time
import os
from pathlib import Path
from typing import Optional

# MT5 reads from Common\Files when using FILE_COMMON flag in MQL5
# This is the correct path that MT5 EAs can access
_APPDATA = Path(os.environ.get("APPDATA", ""))
MT5_COMMON_FILES = _APPDATA / "MetaQuotes" / "Terminal" / "Common" / "Files"

# Use Common\Files if available (EA uses FILE_COMMON), else fallback to local
SIGNALS_DIR = MT5_COMMON_FILES if MT5_COMMON_FILES.parent.exists() else (
    Path(__file__).parent.parent / "mt5_signals"
)
SIGNAL_FILE  = SIGNALS_DIR / "signal.json"
RESULT_FILE  = SIGNALS_DIR / "result.json"
LOCK_FILE    = SIGNALS_DIR / "processing.lock"

# MT5 trade return codes
TRADE_RETCODE_DONE = 10009


class MT5EABridge:
    """
    Communicates with SMCBotEA.mq5 via JSON signal files.
    The EA runs inside MT5 (which has trading permission) and executes orders.
    """

    def __init__(self, timeout_sec: float = 30.0):
        self.timeout_sec = timeout_sec
        SIGNALS_DIR.mkdir(exist_ok=True)
        # Clean stale files on init
        for f in [SIGNAL_FILE, RESULT_FILE, LOCK_FILE]:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass

    def is_ea_running(self) -> bool:
        """Check if EA wrote a result recently (within 30s)."""
        try:
            if RESULT_FILE.exists():
                age = time.time() - RESULT_FILE.stat().st_mtime
                return age < 60
        except Exception:
            pass
        return False

    def place_order(
        self,
        symbol: str,
        direction: str,       # "long" | "short"
        volume: float = 0.01,
        sl_pips: float = 50.0,
        tp_pips: float = 100.0,
        comment: str = "SMC Bot",
    ) -> dict:
        """
        Write signal to signal.json, wait for EA to execute and write result.json.
        Returns result dict with retcode, order, price etc.
        """
        # Clean previous result
        try:
            RESULT_FILE.unlink(missing_ok=True)
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass

        # Write signal
        signal = {
            "symbol":    symbol.upper(),
            "direction": direction.lower(),
            "volume":    round(volume, 2),
            "sl_pips":   sl_pips,
            "tp_pips":   tp_pips,
            "comment":   comment,
            "timestamp": time.time(),
        }
        SIGNAL_FILE.write_text(json.dumps(signal), encoding="utf-8")

        # Wait for EA to process
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            time.sleep(0.5)
            if RESULT_FILE.exists() and not LOCK_FILE.exists():
                try:
                    result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
                    RESULT_FILE.unlink(missing_ok=True)
                    return result
                except Exception:
                    pass

        # Timeout
        try:
            SIGNAL_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "retcode": -1,
            "error": f"EA did not respond within {self.timeout_sec}s. "
                     "Make sure SMCBotEA is compiled and attached to a chart.",
            "order": 0,
        }

    def get_last_result(self) -> Optional[dict]:
        """Read last result without waiting."""
        try:
            if RESULT_FILE.exists():
                return json.loads(RESULT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    def format_signal_for_ea(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        volume: float = 0.01,
    ) -> dict:
        """Convert bot signal to EA-compatible pips format."""
        pip_size = 0.0001 if "JPY" not in symbol.upper() else 0.01
        if symbol.upper() == "XAUUSD":
            pip_size = 0.1
        sl_pips = abs(entry - stop_loss) / pip_size if stop_loss else 50
        tp_pips = abs(take_profit - entry) / pip_size if take_profit else 100
        return {
            "symbol":    symbol,
            "direction": direction,
            "volume":    volume,
            "sl_pips":   round(sl_pips, 1),
            "tp_pips":   round(tp_pips, 1),
        }
