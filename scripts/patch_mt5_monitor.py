"""Add MT5 reconnect monitor to supervisor.py."""
import ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

content = open('core/supervisor.py', encoding='utf-8').read()
ast.parse(content)
print(f"Base: {len(content)} chars, parses OK")

# 1. Update the MT5 startup check in run() to also ensure port 443 config
old_mt5_check = (
    '        # MT5 startup check\n'
    '        loop = asyncio.get_event_loop()\n'
    '        mt5_ok = await loop.run_in_executor(None, self.mt5.connect)\n'
    '        if mt5_ok:\n'
    '            self._mt5_available = True\n'
    '            info = await loop.run_in_executor(None, self.mt5.get_account_info)\n'
    '            bal  = info.get("balance", 0)\n'
    '            print(f"  MT5:           CONECTADO -- Balance ${bal:,.2f}")\n'
    '            print(f"  Forex:         {\', \'.join(MT5_SYMBOLS)}")\n'
    '        else:\n'
    '            self._mt5_available = False\n'
    '            msg = self.mt5.last_error_msg()\n'
    '            print(f"  MT5:           {msg}")\n'
    '            try:\n'
    '                await self.telegram.send_glint_alert(f"MT5 no disponible -- {msg}")\n'
    '            except Exception:\n'
    '                pass\n'
    '        print()'
)
new_mt5_check = (
    '        # MT5 startup check — try port 443 first for ISP compatibility\n'
    '        loop = asyncio.get_event_loop()\n'
    '        await loop.run_in_executor(None, self.mt5.ensure_port_443_config)\n'
    '        mt5_ok = await loop.run_in_executor(None, self.mt5.connect)\n'
    '        if mt5_ok:\n'
    '            self._mt5_available = True\n'
    '            info = await loop.run_in_executor(None, self.mt5.get_account_info)\n'
    '            bal  = info.get("balance", 0)\n'
    '            print(f"  MT5:           CONECTADO -- Balance ${bal:,.2f}")\n'
    '            print(f"  Forex:         {\', \'.join(MT5_SYMBOLS)}")\n'
    '            try:\n'
    '                await self.telegram.send_glint_alert(\n'
    '                    f"<b>MT5 AXI CONECTADO</b>\\nBalance: ${bal:,.2f} USD\\nServer: {self.mt5.server}"\n'
    '                )\n'
    '            except Exception:\n'
    '                pass\n'
    '        else:\n'
    '            self._mt5_available = False\n'
    '            msg = self.mt5.last_error_msg()\n'
    '            print(f"  MT5:           {msg}")\n'
    '            try:\n'
    '                await self.telegram.send_glint_alert(f"MT5 no disponible -- {msg}")\n'
    '            except Exception:\n'
    '                pass\n'
    '        print()'
)
if old_mt5_check in content:
    content = content.replace(old_mt5_check, new_mt5_check)
    print("MT5 startup check updated with port 443 and connect notification")
else:
    print("WARNING: MT5 startup check block not found exactly")

# 2. Add MT5 reconnect monitor to the scan loop
# Find the forex scan block and add MT5 disconnect detection before the sleep
old_sleep = '            await asyncio.sleep(SCAN_INTERVAL_SEC)  # next full scan'
new_sleep = (
    '            # MT5 reconnect monitor — check every scan cycle\n'
    '            if self._mt5_available and not self.mt5.is_connected():\n'
    '                self._mt5_available = False\n'
    '                print("[MT5] Desconectado — reintentando en 60s...")\n'
    '                try:\n'
    '                    await self.telegram.send_glint_alert(\n'
    '                        self.mt5.disconnect_alert_msg()\n'
    '                    )\n'
    '                except Exception:\n'
    '                    pass\n'
    '            elif not self._mt5_available:\n'
    '                # Silent reconnect attempt every scan cycle\n'
    '                loop2 = asyncio.get_event_loop()\n'
    '                mt5_ok = await loop2.run_in_executor(None, self.mt5.reconnect)\n'
    '                if mt5_ok:\n'
    '                    self._mt5_available = True\n'
    '                    info = await loop2.run_in_executor(None, self.mt5.get_account_info)\n'
    '                    bal = info.get("balance", 0)\n'
    '                    print(f"[MT5] Reconectado! Balance ${bal:,.2f}")\n'
    '                    try:\n'
    '                        await self.telegram.send_glint_alert(\n'
    '                            f"<b>MT5 AXI RECONECTADO</b>\\nBalance: ${bal:,.2f} USD"\n'
    '                        )\n'
    '                    except Exception:\n'
    '                        pass\n'
    '\n'
    '            await asyncio.sleep(SCAN_INTERVAL_SEC)  # next full scan'
)
if old_sleep in content:
    content = content.replace(old_sleep, new_sleep)
    print("MT5 reconnect monitor added to scan loop")
else:
    print("WARNING: sleep line not found")

ast.parse(content)
open('core/supervisor.py', 'w', encoding='utf-8').write(content)
print("supervisor.py updated")
