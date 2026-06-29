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
        import time
        if not HAS_MT5:
            return False
        try:
            mt5.shutdown()
        except Exception:
            pass
        time.sleep(2)  # allow MT5 terminal to reset IPC state

        axi_path = AXI_EXE if os.path.exists(AXI_EXE) else STD_EXE

        # Port 443 server variants (always open on ISP, never blocked)
        server_443 = f"{self.server}:443" if ":" not in self.server else self.server

        attempts = [
            # 1. Normal server name (most reliable on Axi)
            lambda: mt5.initialize(login=self.login, password=self.password,
                                   server=self.server, timeout=8000),
            # 2. Server with port 443 (bypasses ISP port blocks)
            lambda: mt5.initialize(login=self.login, password=self.password,
                                   server=server_443, timeout=8000),
            # 3. Axi terminal path explicit
            lambda: mt5.initialize(path=axi_path, login=self.login,
                                   password=self.password, server=self.server,
                                   timeout=8000),
            # 4. Attach to active session
            lambda: mt5.initialize(timeout=8000),
        ]

        for attempt in attempts:
            try:
                if attempt():
                    self._connected = True
                    return True
                mt5.shutdown()
                time.sleep(1)
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
        if not self.is_connected():
            logger.warning(f"MT5 get_ohlcv: no conectado para {symbol}")
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
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"No tick for {symbol}: {mt5.last_error()}"}
            info = mt5.symbol_info(symbol)
            ot    = mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if ot == mt5.ORDER_TYPE_BUY else tick.bid
            if price <= 0:
                return {"error": f"Precio invalido ({price}) para {symbol} — mercado cerrado?"}

            # Enforce minimum stop distance required by the broker
            if info is not None:
                point     = info.point
                min_dist  = info.trade_stops_level * point
                # Add a safety buffer: 3× the minimum stop distance
                safe_dist = max(min_dist * 3, point * 50)
                if sl != 0.0:
                    if ot == mt5.ORDER_TYPE_BUY and sl >= price - safe_dist:
                        sl = round(price - safe_dist, info.digits)
                    elif ot == mt5.ORDER_TYPE_SELL and sl <= price + safe_dist:
                        sl = round(price + safe_dist, info.digits)
                if tp != 0.0:
                    if ot == mt5.ORDER_TYPE_BUY and tp <= price + safe_dist:
                        tp = round(price + safe_dist, info.digits)
                    elif ot == mt5.ORDER_TYPE_SELL and tp >= price - safe_dist:
                        tp = round(price - safe_dist, info.digits)

            # Detect supported filling mode — NAS100/indices on some brokers only support FOK or RETURN
            _sym_fill = mt5.symbol_info(symbol)
            _fill_mode_order = mt5.ORDER_FILLING_IOC
            if _sym_fill:
                _fm = _sym_fill.filling_mode
                if _fm & 0x01:
                    _fill_mode_order = mt5.ORDER_FILLING_FOK
                elif _fm & 0x02:
                    _fill_mode_order = mt5.ORDER_FILLING_IOC
                elif _fm & 0x04:
                    _fill_mode_order = mt5.ORDER_FILLING_RETURN
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       volume,
                "type":         ot,
                "price":        price,
                "sl":           sl,
                "tp":           tp,
                "deviation":    50,
                "magic":        234000,
                "comment":      "SMC Bot",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": _fill_mode_order,
            }
            result = mt5.order_send(request)
            if result is None:
                return {"error": f"order_send None: {mt5.last_error()}"}
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"error": f"Retcode {result.retcode}: {result.comment}"}
            logger.info(f"MT5 order filled: {symbol} {order_type} #{result.order} @{result.price}")

            # Some brokers (Axi indices) strip SL/TP from the fill request.
            # Apply SL/TP as a separate SLTP modification with retries + delay.
            if (sl != 0.0 or tp != 0.0) and result.order:
                import time as _time
                # Wait for the position to appear in the terminal before modifying
                _time.sleep(1.5)
                # Find the actual position ticket (may differ from order ticket on some brokers)
                _pos_ticket = result.order
                _positions = mt5.positions_get(symbol=symbol)
                if _positions:
                    # Use the most recently opened position for this symbol
                    _matching = [p for p in _positions if p.magic == 234000]
                    if _matching:
                        _pos_ticket = sorted(_matching, key=lambda p: p.time, reverse=True)[0].ticket
                sl_req = {
                    "action":   mt5.TRADE_ACTION_SLTP,
                    "position": _pos_ticket,
                    "symbol":   symbol,
                    "sl":       sl,
                    "tp":       tp,
                }
                sl_ok = False
                for _attempt in range(6):  # up to 6 tries with longer delays
                    _time.sleep(1.0 * (_attempt + 1))
                    sl_result = mt5.order_send(sl_req)
                    rc = sl_result.retcode if sl_result else None
                    # 10009=DONE, 10025=NO_CHANGES (SL/TP already set from fill request)
                    if rc in (mt5.TRADE_RETCODE_DONE, 10025):
                        sl_ok = True
                        logger.info(
                            f"MT5 SL={sl:.5f} TP={tp:.5f} confirmed on "
                            f"#{_pos_ticket} retcode={rc} (attempt {_attempt+1})"
                        )
                        break
                    logger.warning(
                        f"MT5 SL/TP attempt {_attempt+1}/6 retcode={rc} — retrying..."
                    )
                if not sl_ok:
                    logger.error(
                        f"MT5 SL/TP DEFINITIVELY FAILED after 4 attempts — "
                        f"#{result.order} {symbol} CLOSING POSITION IMMEDIATELY to avoid no-SL exposure."
                    )
                    # Close the position immediately — never leave a position open without SL
                    _close_type = mt5.ORDER_TYPE_SELL if ot == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    _close_tick = mt5.symbol_info_tick(symbol)
                    _close_price = _close_tick.bid if ot == mt5.ORDER_TYPE_BUY else _close_tick.ask
                    # Detect supported filling mode for this symbol to avoid retcode=10013
                    _sym_fill_info = mt5.symbol_info(symbol)
                    _fill_mode = mt5.ORDER_FILLING_IOC
                    if _sym_fill_info:
                        _fm = _sym_fill_info.filling_mode
                        if _fm & 0x01:
                            _fill_mode = mt5.ORDER_FILLING_FOK
                        elif _fm & 0x02:
                            _fill_mode = mt5.ORDER_FILLING_IOC
                        elif _fm & 0x04:
                            _fill_mode = mt5.ORDER_FILLING_RETURN
                    _close_req = {
                        "action":       mt5.TRADE_ACTION_DEAL,
                        "symbol":       symbol,
                        "volume":       volume,
                        "type":         _close_type,
                        "position":     _pos_ticket,
                        "price":        _close_price,
                        "deviation":    100,
                        "magic":        234000,
                        "comment":      "SMC Bot NoSL Close",
                        "type_time":    mt5.ORDER_TIME_GTC,
                        "type_filling": _fill_mode,
                    }
                    _cr = mt5.order_send(_close_req)
                    if _cr and _cr.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.error(f"MT5 NoSL position #{_pos_ticket} CLOSED immediately.")
                        return {"error": f"SL/TP failed — position closed immediately to avoid exposure"}
                    else:
                        logger.error(f"MT5 NoSL close FAILED retcode={_cr.retcode if _cr else 'None'} — MANUAL INTERVENTION REQUIRED")

            return {"ticket": result.order, "status": "filled", "price": result.price}
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
                     "volume": p.volume, "profit": p.profit,
                     "price_open": p.price_open,
                     "price_current": p.price_current,
                     "sl": p.sl, "tp": p.tp,
                     "time": p.time} for p in positions]
        except Exception as e:
            logger.error(f"MT5 get_positions error: {e}")
            return []

    def get_pnl_report(self, initial_balance: float = 100_000.0) -> dict:
        """Real P&L report: balance vs initial, monthly deals, per-trade stats."""
        empty = {"error": "MT5 not available"}
        if not HAS_MT5:
            return empty
        try:
            from datetime import datetime, timezone
            # Ensure initialized
            if not mt5.terminal_info():
                mt5.initialize()
            acc = mt5.account_info()
            if acc is None:
                return empty

            balance    = acc.balance
            equity     = acc.equity
            profit     = acc.profit        # unrealized (open positions)
            currency   = acc.currency

            # Month-to-date deals
            now        = datetime.now(timezone.utc)
            month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
            deals       = mt5.history_deals_get(month_start, now) or []

            # Closing deals (entry=1) carry realized P&L
            closing    = [d for d in deals if d.entry == 1 and d.symbol != ""]
            realized   = sum(d.profit + d.swap + d.commission for d in closing)
            trade_deals = [d for d in deals if d.symbol != ""]
            n_trades   = len(trade_deals)

            # Net change vs initial deposit
            net_change = balance - initial_balance

            # Last 5 trades for display
            recent = []
            for d in trade_deals[-5:]:
                dt = datetime.fromtimestamp(d.time, tz=timezone.utc).strftime("%m-%d %H:%M")
                direction = "BUY" if d.type == 0 else "SELL"
                notional  = round(d.volume * 100_000 / max(d.price, 1), 2) if d.price > 0 else 0.0
                recent.append({
                    "dt": dt, "symbol": d.symbol, "direction": direction,
                    "volume": d.volume, "price": d.price,
                    "notional_usd": notional,
                    "profit": d.profit + d.swap,
                    "entry": d.entry,
                })

            return {
                "initial_balance": initial_balance,
                "balance":         balance,
                "equity":          equity,
                "profit_open":     profit,
                "net_change":      net_change,
                "realized_pnl":    realized,
                "currency":        currency,
                "n_trades":        n_trades,
                "recent_trades":   recent,
            }
        except Exception as e:
            logger.error(f"MT5 get_pnl_report error: {e}")
            return {"error": str(e)}

    def get_closing_deal(self, position_ticket: int) -> dict:
        """Return the closing deal for a position (entry=1 in MT5 history)."""
        if not HAS_MT5:
            return {}
        try:
            from datetime import datetime, timezone, timedelta
            now  = datetime.now(timezone.utc)
            from_dt = now - timedelta(days=7)
            deals = mt5.history_deals_get(from_dt, now) or []
            for d in deals:
                if d.position_id == position_ticket and d.entry == 1:
                    return {
                        "profit":  round(d.profit + d.swap + d.commission, 2),
                        "price":   d.price,
                        "symbol":  d.symbol,
                        "volume":  d.volume,
                        "time":    d.time,
                    }
        except Exception as e:
            logger.error(f"MT5 get_closing_deal error: {e}")
        return {}

    def get_daily_pnl(self) -> float:
        """Realized P&L for today (UTC)."""
        if not HAS_MT5:
            return 0.0
        try:
            from datetime import datetime, timezone
            now   = datetime.now(timezone.utc)
            today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            deals = mt5.history_deals_get(today, now) or []
            return round(sum(
                d.profit + d.swap + d.commission
                for d in deals if d.entry == 1 and d.symbol != ""
            ), 2)
        except Exception:
            return 0.0

    def get_scalp_daily_pnl(self) -> float:
        """Realized P&L from scalp trades today (volume <= 0.1L). Syncs from MT5."""
        if not HAS_MT5:
            return 0.0
        try:
            from datetime import datetime, timezone
            now   = datetime.now(timezone.utc)
            today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            deals = mt5.history_deals_get(today, now) or []
            return round(sum(
                d.profit + d.swap + d.commission
                for d in deals if d.entry == 1 and d.symbol != "" and d.volume <= 0.11
            ), 2)
        except Exception:
            return 0.0

    def modify_position_sl_tp(self, ticket: int, sl: float, tp: float = 0.0) -> bool:
        """Set or update SL/TP on an existing open position."""
        if not HAS_MT5:
            return False
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                return False
            p = pos[0]
            request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol":   p.symbol,
                "sl":       sl,
                "tp":       tp if tp != 0.0 else p.tp,
            }
            result = mt5.order_send(request)
            ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
            if not ok and result is not None:
                logger.warning(f"modify_sl_tp retcode {result.retcode}: {result.comment}")
            return ok
        except Exception as e:
            logger.error(f"MT5 modify_position_sl_tp error: {e}")
            return False

    def close_position(self, ticket: int) -> bool:
        if not HAS_MT5:
            return False
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                return False
            p = pos[0]
            close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(p.symbol)
            if tick is None:
                logger.error(f"close_position: no tick para {p.symbol}, abortando #{ticket}")
                return False
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            if price <= 0:
                logger.error(f"close_position: precio invalido ({price}) para {p.symbol}")
                return False
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
                "volume": p.volume, "type": close_type,
                "position": ticket, "price": price,
                "deviation": 20, "comment": "SMC Bot Close",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"MT5 close_position error: {e}")
            return False

    def partial_close_position(self, ticket: int, close_volume: float) -> bool:
        """Close *close_volume* lots of an open position (partial close).
        Used to lock in 50% profit at 1:1 RR milestone."""
        if not HAS_MT5:
            return False
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                return False
            p = pos[0]
            sym_info = mt5.symbol_info(p.symbol)
            if sym_info is None:
                return False
            min_vol  = sym_info.volume_min
            vol_step = sym_info.volume_step
            # Round close_volume to broker step, floor to min_vol
            vol = max(min_vol, round(round(close_volume / vol_step) * vol_step, 8))
            if vol >= p.volume:
                return self.close_position(ticket)  # full close
            close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(p.symbol)
            if tick is None:
                return False
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
                "volume": vol, "type": close_type,
                "position": ticket, "price": price,
                "deviation": 20, "comment": "SMC PartialClose 50%",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
            if ok:
                logger.info(f"MT5 partial close: #{ticket} {p.symbol} -{vol}L @ {price}")
            else:
                rc = result.retcode if result else "None"
                logger.warning(f"MT5 partial close failed: #{ticket} retcode={rc}")
            return ok
        except Exception as e:
            logger.error(f"MT5 partial_close error: {e}")
            return False
