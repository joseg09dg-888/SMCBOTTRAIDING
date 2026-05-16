# connectors/metatrader_connector.py
import logging
from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    mt5 = None

TIMEFRAME_MAP = {
    "1m":  1,   "5m":  5,   "15m": 15,  "30m": 30,
    "1h":  16385, "4h": 16388, "1d": 16408,
    "H1":  16385, "H4": 16388, "D1": 16408,
}


class MT5Connector:
    SYMBOLS = [
        "EURUSD", "GBPUSD", "XAUUSD", "US30",
        "NAS100", "USOIL", "USDJPY", "GBPJPY",
    ]

    def __init__(self, login: int = 0, password: str = "", server: str = "MetaQuotes-Demo"):
        self.login    = login
        self.password = password
        self.server   = server
        self._connected = False

    def connect(self) -> bool:
        if not HAS_MT5:
            return False
        try:
            mt5.shutdown()  # reset any stale state
        except Exception:
            pass
        for attempt in [
            # 1. Direct credentials
            lambda: mt5.initialize(login=self.login, password=self.password, server=self.server),
            # 2. Path + credentials
            lambda: mt5.initialize(
                path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
                login=self.login, password=self.password, server=self.server),
            # 3. Active session (no credentials)
            lambda: mt5.initialize(),
        ]:
            try:
                if attempt():
                    self._connected = True
                    return True
                mt5.shutdown()
            except Exception:
                pass
        self._connected = False
        return False

    def last_error_msg(self) -> str:
        """Human-readable last error. Returns empty string if no error."""
        if not HAS_MT5:
            return "MetaTrader5 package not installed"
        try:
            code, msg = mt5.last_error()
            if code == -6:
                return (
                    "MT5: Algo Trading desactivado. "
                    "En MT5 activa el boton 'Algo Trading' (rayo verde) "
                    "en la barra superior."
                )
            return f"MT5 error {code}: {msg}"
        except Exception:
            return "MT5 desconocido"

    def get_ohlcv(self, symbol: str, timeframe: str = "H1", count: int = 200) -> pd.DataFrame:
        if not HAS_MT5:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        try:
            tf = TIMEFRAME_MAP.get(timeframe, 16385)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = pd.DataFrame(rates)
            df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"MT5 get_ohlcv error: {e}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def get_account_info(self) -> dict:
        if not HAS_MT5:
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0}
        try:
            info = mt5.account_info()
            if info is None:
                return {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0}
            return {
                "balance":     info.balance,
                "equity":      info.equity,
                "margin":      info.margin,
                "free_margin": info.margin_free,
            }
        except Exception as e:
            logger.error(f"MT5 account_info error: {e}")
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0}

    def place_order(self, symbol: str, order_type: str, volume: float,
                    sl: float = 0.0, tp: float = 0.0) -> dict:
        if not HAS_MT5:
            return {"error": "MT5 not installed"}
        try:
            action = mt5.TRADE_ACTION_DEAL
            ot = mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL
            request = {
                "action": action, "symbol": symbol, "volume": volume,
                "type": ot, "sl": sl, "tp": tp,
                "comment": "SMC Bot", "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"error": f"Order failed: {result}"}
            return {"ticket": result.order, "status": "filled"}
        except Exception as e:
            logger.error(f"MT5 place_order error: {e}")
            return {"error": str(e)}

    def get_positions(self) -> List[dict]:
        if not HAS_MT5:
            return []
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            return [{"ticket": p.ticket, "symbol": p.symbol,
                     "type": "BUY" if p.type == 0 else "SELL",
                     "volume": p.volume, "profit": p.profit} for p in positions]
        except Exception as e:
            logger.error(f"MT5 get_positions error: {e}")
            return []

    def close_position(self, ticket: int) -> bool:
        if not HAS_MT5:
            return False
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                return False
            p = pos[0]
            close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
                "volume": p.volume, "type": close_type,
                "position": ticket, "comment": "SMC Bot Close",
            }
            result = mt5.order_send(request)
            return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"MT5 close_position error: {e}")
            return False
