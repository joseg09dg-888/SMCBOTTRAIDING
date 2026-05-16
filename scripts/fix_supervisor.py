"""Rebuild supervisor.py cleanly from git history + MT5 additions."""
import subprocess, sys, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

# 1. Get clean base from last known-good commit
result = subprocess.run(
    ['git', 'show', '43561ac:core/supervisor.py'],
    capture_output=True
)
content = result.stdout.decode('utf-8-sig')  # strip BOM if present
print(f"Base from git: {len(content)} chars")

# 2. Verify it parses
import ast
ast.parse(content)
print("Base parses OK")

# 3. Apply MT5 additions

# 3a. Add MT5Connector import
content = content.replace(
    'from connectors.binance_connector import BinanceConnector\n'
    'from connectors.glint_connector import GlintSignal',
    'from connectors.binance_connector import BinanceConnector\n'
    'from connectors.metatrader_connector import MT5Connector\n'
    'from connectors.glint_connector import GlintSignal'
)

# 3b. Add MT5 constants after SCAN_TIMEFRAMES
content = content.replace(
    'SCAN_TIMEFRAMES  = ["1h", "4h"]\n',
    'SCAN_TIMEFRAMES  = ["1h", "4h"]\n\n'
    '# MT5 forex/indices symbols\n'
    'MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]\n'
    'MT5_TIMEFRAMES   = ["H1", "H4"]\n'
    'MT5_MIN_VOLUME   = 0.01\n'
)

# 3c. Add MT5Connector to __init__ after signal_agent
content = content.replace(
    '        self.signal_agent   = SignalAgent(min_confidence=0.55)\n'
    '        self.binance        = BinanceConnector(',
    '        self.signal_agent   = SignalAgent(min_confidence=0.55)\n'
    '        self.mt5            = MT5Connector(\n'
    '            login    = config.mt5_login,\n'
    '            password = config.mt5_password,\n'
    '            server   = config.mt5_server,\n'
    '        )\n'
    '        self._mt5_available = False\n'
    '        self.binance        = BinanceConnector('
)

# 3d. Add MT5 startup check in run() after demo mode print
old_run = (
    '        if self.demo_mode:\n'
    '            print(f"  DEMO MODE:     threshold={DEMO_SCORE_THRESHOLD} | max_trades={DEMO_MAX_POSITIONS}")\n'
    '            print(f"  Pares:         {\', \'.join(SCAN_SYMBOLS)}")\n'
    '        print()\n\n'
    '        await asyncio.gather(\n'
    '            self.commander.start_polling(),\n'
    '            self.glint.connect(),\n'
    '            self._market_scan_loop(),\n'
    '        )'
)
new_run = (
    '        if self.demo_mode:\n'
    '            print(f"  DEMO MODE:     threshold={DEMO_SCORE_THRESHOLD} | max_trades={DEMO_MAX_POSITIONS}")\n'
    '            print(f"  Crypto:        {\', \'.join(SCAN_SYMBOLS)}")\n'
    '        print()\n\n'
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
    '        print()\n\n'
    '        await asyncio.gather(\n'
    '            self.commander.start_polling(),\n'
    '            self.glint.connect(),\n'
    '            self._market_scan_loop(),\n'
    '        )'
)
if old_run in content:
    content = content.replace(old_run, new_run)
    print("run() MT5 check inserted")
else:
    print("WARNING: run() target not found — skipping")

# 3e. Add _scan_mt5_symbol before _execute_demo_trade
mt5_scan_method = '''
    async def _scan_mt5_symbol(self, symbol: str, timeframe: str):
        """Fetch MT5 OHLCV, run SMC lite, return signal or None."""
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, lambda: self.mt5.get_ohlcv(symbol, timeframe, 200)
        )
        if df is None or df.empty or len(df) < 50:
            return None
        smc = self._run_smc_lite(df)
        current_price = float(df["close"].iloc[-1])
        signal = self.signal_agent.evaluate(
            analysis_text=smc["analysis_text"], symbol=symbol,
            timeframe=timeframe, current_price=current_price,
            poi_zones=smc["poi_zones"], glint_context=self._last_glint_text,
        )
        if signal.signal_type == SignalType.WAIT:
            return signal
        signal = self.route_signal(signal, df)
        return signal

'''
# Insert before _execute_demo_trade
content = content.replace(
    '    async def _execute_demo_trade(self, signal: TradeSignal):',
    mt5_scan_method + '    async def _execute_demo_trade(self, signal: TradeSignal):'
)

# 3f. Fix demo trade message to clean HTML (replace the whole body)
old_msg = (
    '        direction = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"\n'
    '        dir_icon  = "\\U0001f7e2" if signal.signal_type == SignalType.LONG else "\\U0001f534"\n'
)
# Actually just replace what's there
import re

# Find and replace the _execute_demo_trade body
old_body_pattern = r'(    async def _execute_demo_trade\(self, signal: TradeSignal\):.*?"""Record a simulated trade, send Telegram notification.""".*?        if len\(self\._demo_trades\) >= DEMO_MAX_POSITIONS:.*?return.*?)(.*?)(        print\(f"\[DEMO TRADE\].*?\).*?try:.*?await self\.telegram\.send_glint_alert\(msg\).*?except Exception:.*?pass)'

new_trade_body = '''    async def _execute_demo_trade(self, signal: TradeSignal):
        """Record a simulated trade, send Telegram notification."""
        if len(self._demo_trades) >= DEMO_MAX_POSITIONS:
            return

        demo = DemoTrade(signal, signal.decision_score)
        self._demo_trades.append(demo)

        direction = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"
        dir_icon  = "\\U0001f7e2" if signal.signal_type == SignalType.LONG else "\\U0001f534"
        market    = "MT5" if signal.symbol in MT5_SYMBOLS else "Binance"

        msg = (
            f"<b>🚀 TRADE DEMO ABIERTO</b>\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"📊 Par: <b>{signal.symbol}</b> | {signal.timeframe} | {market}\\n"
            f"{dir_icon} <b>{direction}</b>\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"📍 Entrada:    <code>{signal.entry:,.5f}</code>\\n"
            f"🛑 Stop Loss:  <code>{signal.stop_loss:,.5f}</code>\\n"
            f"🎯 Take Profit:<code>{signal.take_profit:,.5f}</code>\\n"
            f"📊 R:R: <code>1:{signal.risk_reward:.1f}</code>\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"⚡ Score: <b>{signal.decision_score}/100</b>\\n"
            f"Trigger: {signal.trigger}\\n"
            f"Activos: {len(self._demo_trades)}/{DEMO_MAX_POSITIONS}\\n"
            f"💡 DEMO - sin dinero real"
        )
        print(f"[DEMO TRADE] {signal.symbol} {direction} "
              f"entry={signal.entry:.4f} score={signal.decision_score}")
        try:
            await self.telegram.send_glint_alert(msg)
        except Exception:
            pass
'''

# Find and replace the method
idx_start = content.find('    async def _execute_demo_trade(self, signal: TradeSignal):')
idx_end = content.find('\n    # ── Market scan loop', idx_start)
if idx_start > 0 and idx_end > 0:
    content = content[:idx_start] + new_trade_body + '\n' + content[idx_end:]
    print("_execute_demo_trade replaced")
else:
    print("WARNING: _execute_demo_trade not found cleanly")

# 3g. Add MT5 scan block to _market_scan_loop
old_rate_limit = '                        await asyncio.sleep(1)  # rate limit between symbols\n\n            except asyncio.CancelledError:'
new_rate_limit = (
    '                        await asyncio.sleep(1)  # rate limit between symbols\n\n'
    '                # MT5 forex/indices scan\n'
    '                if self._mt5_available:\n'
    '                    for symbol in MT5_SYMBOLS:\n'
    '                        for tf in MT5_TIMEFRAMES:\n'
    '                            if not self._running: break\n'
    '                            try:\n'
    '                                signal = await self._scan_mt5_symbol(symbol, tf)\n'
    '                                if signal is None:\n'
    '                                    print(f"[MT5][{symbol}][{tf}] Sin datos")\n'
    '                                    continue\n'
    '                                score = signal.decision_score\n'
    '                                bias  = signal.signal_type.value.upper()\n'
    '                                print(f"[MT5][{symbol}][{tf}] Score: {score} | {bias}", end="")\n'
    '                                if signal.signal_type == SignalType.WAIT or score < threshold:\n'
    '                                    print(" -- sin setup")\n'
    '                                elif self.demo_mode:\n'
    '                                    print(f" -- trade DEMO MT5")\n'
    '                                    await self._execute_demo_trade(signal)\n'
    '                                else:\n'
    '                                    self._dispatch(signal)\n'
    '                            except Exception as exc:\n'
    '                                print(f"[MT5][{symbol}][{tf}] Error: {exc.__class__.__name__}")\n'
    '                            await asyncio.sleep(0.5)\n\n'
    '            except asyncio.CancelledError:'
)
if old_rate_limit in content:
    content = content.replace(old_rate_limit, new_rate_limit)
    print("MT5 scan block inserted")
else:
    print("WARNING: rate limit target not found")

# 4. Verify final content parses
try:
    ast.parse(content)
    print("Final content parses OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

# 5. Write without BOM
utf8_no_bom = open('core/supervisor.py', 'w', encoding='utf-8')
utf8_no_bom.write(content)
utf8_no_bom.close()
print("supervisor.py written cleanly")
