# connectors/metatrader_connector.py
import logging
import os
import re
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

# Axi session config
AXI_CFG = os.path.expandvars(
    r"%APPDATA%\MetaQuotes\Terminal"
    r"\6FBEE76C719DC78AB2AE839B5A0C7442\config\common.ini"
)
AXI_EXE = r"C:\Program Files\Axi MetaTrader 5 Terminal\terminal64.exe"
STD_EXE = r"C:\Program Files\MetaTrader 5\terminal64.exe"


def _write_axi_config(login: int, server: str):
    """Write Axi MT5 config with port 443 variants — UTF-8, no BOM."""
    cfg_dir = os.path.dirname(AXI_CFG)
    os.makedirs(cfg_dir, exist_ok=True)
    content = (
        "[Common]\n"
        f"Login={login}\n"
        f"Server={server}\n"
        "ProxyEnable=0\n"
        "ProxyType=0\n"
        "ProxyAddress=\n"
        "[Experts]\n"
        "AllowDllImport=1\n"
        "Enabled=1\n"
        "Account=1\n"
        "Profile=1\n"
        "Chart=0\n"
        "Api=1\n"
        "WebRequest=1\n"
    )
    with open(AXI_CFG, "w", encoding="utf-8") as f:
        f.write(content)


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

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not HAS_MT5:
            return False
        try:
            mt5.shutdown()
        except Exception:
            pass

        axi_path = AXI_EXE if os.path.exists(AXI_EXE) else STD_EXE

        # Port 443 server variants (always open on ISP, never blocked)
        server_443 = f"{self.server}:443" if ":" not in self.server else self.server

        attempts = [
            # 1. Server with port 443 (bypasses ISP port blocks)
            lambda: mt5.initialize(login=self.login, password=self.password,
                                   server=server_443, timeout=5000),
            # 2. Normal server name
            lambda: mt5.initialize(login=self.login, password=self.password,
                                   server=self.server, timeout=5000),
            # 3. Axi terminal path + port 443
            lambda: mt5.initialize(path=axi_path, login=self.login,
                                   password=self.password, server=server_443,
                                   timeout=5000),
            # 4. Attach to active session
            lambda: mt5.initialize(timeout=5000),
        ]

        for attempt in attempts:
            try:
                if attempt():
                    self._connected = True
                    return True
                mt5.shutdown()
            except Exception:
                pass

        self._connected = False
        return False

    def is_connected(self) -> bool:
        """Fast check if MT5 connection is still alive."""
        if not HAS_MT5 or not self._connected:
            return False
        try:
            info = mt5.account_info()
            return info is not None
        except Exception:
            self._connected = False
            return False

    def reconnect(self) -> bool:
        """Attempt to reconnect. Returns True if successful."""
        self._connected = False
        try:
            mt5.shutdown()
        except Exception:
            pass
        return self.connect()

    def ensure_port_443_config(self):
        """
        Update Axi config to force port 443 connection.
        This allows connection even on ISPs that block default MT5 ports.
        """
        if not os.path.exists(AXI_CFG):
            return
        try:
            content = open(AXI_CFG, encoding="utf-8").read()
            # If server doesn't already specify port 443, add it
            def add_port(m):
                srv = m.group(1)
                if ":" not in srv and srv:
                    return f"Server={srv}:443"
                return m.group(0)
            new_content = re.sub(r"^Server=(.+)$", add_port, content, flags=re.M)
            if new_content != content:
                with open(AXI_CFG, "w", encoding="utf-8") as f:
                    f.write(new_content)
                logger.info("Axi config updated: port 443 added to server")
        except Exception as e:
            logger.debug(f"ensure_port_443_config: {e}")

    # ── Error reporting ───────────────────────────────────────────────────

    def last_error_msg(self) -> str:
        if not HAS_MT5:
            return "MetaTrader5 no instalado"
        try:
            code, msg = mt5.last_error()
            if code in (-6, 1):
                return (
                    "MT5 no autorizado -- activa hotspot del celular o "
                    "habilita 'Algo Trading' en MT5 (boton verde en toolbar)"
                )
            if code == -10005:
                return (
                    "MT5: IPC timeout -- servidor no responde. "
                    "Soluciones: 1) Activa hotspot celular "
                    "2) Reinicia Axi MT5 terminal"
                )
            if code == 0:
                return "MT5: conexion OK"
            return f"MT5 error {code}: {msg}"
        except Exception:
            return "MT5: error desconocido"

    def disconnect_alert_msg(self) -> str:
        """Message to send to Telegram when MT5 disconnects."""
        return (
            "<b>MT5 AXI DESCONECTADO</b>\n"
            "Para reconectar:\n"
            "1. Activa hotspot del celular\n"
            "2. Abre Axi MetaTrader 5 Terminal\n"
            "3. El bot reconectara automaticamente\n"
            "Server: Axi-US50-Demo | Login: 10042896"
        )

    # ── Data ──────────────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str, timeframe: str = "H1", count: int = 200) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if not HAS_MT5:
            return empty
        try:
            tf = TIMEFRAME_MAP.get(timeframe, 16385)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None:
                return empty
            df = pd.DataFrame(rates)
            df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"MT5 get_ohlcv error: {e}")
            return empty

    def get_account_info(self) -> dict:
        empty = {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0}
        if not HAS_MT5:
            return empty
        try:
            info = mt5.account_info()
            if info is None:
                return empty
            return {
                "balance":     info.balance,
                "equity":      info.equity,
                "margin":      info.margin,
                "free_margin": info.margin_free,
            }
        except Exception as e:
            logger.error(f"MT5 account_info error: {e}")
            return empty

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
