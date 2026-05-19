"""Deep fix: restore clean supervisor.py from git and apply all changes via Python."""
import subprocess, ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

# 1. Get clean base from last commit that had supervisor.py working (43561ac)
result = subprocess.run(['git', 'show', '43561ac:core/supervisor.py'], capture_output=True)
raw = result.stdout
# Decode stripping BOM
content = raw.decode('utf-8-sig')
print(f"Base from git: {len(content)} chars, {len(content.splitlines())} lines")

# Verify it parses
ast.parse(content)
print("Base parses OK")

# 2. Apply all required changes as string operations

# 2a. Add MT5Connector import
content = content.replace(
    'from connectors.binance_connector import BinanceConnector\n'
    'from connectors.glint_connector import GlintSignal',
    'from connectors.binance_connector import BinanceConnector\n'
    'from connectors.metatrader_connector import MT5Connector\n'
    'from connectors.glint_connector import GlintSignal'
)

# 2b. Replace constants block
old_constants = (
    '# Demo mode: lower score threshold so the bot actually trades while learning\n'
    'DEMO_SCORE_THRESHOLD = 35   # instead of 60 -- generates more trades for training\n'
    'DEMO_MAX_POSITIONS   = 5    # maximum simultaneous demo trades\n'
    '\n'
    '# Symbols and timeframes to scan\n'
    'SCAN_SYMBOLS     = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]\n'
    'SCAN_TIMEFRAMES  = ["1h", "4h"]\n'
)
new_constants = (
    '# Demo mode: lower score threshold so the bot actually trades while learning\n'
    'DEMO_SCORE_THRESHOLD = 30   # aggressive demo -- learn from more trades\n'
    'DEMO_MAX_POSITIONS   = 5\n'
    'SCAN_INTERVAL_SEC    = 30   # scan every 30s instead of 60s\n'
    '\n'
    '# Symbols and timeframes to scan\n'
    'SCAN_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]\n'
    'SCAN_TIMEFRAMES = ["5m", "15m", "1h", "4h"]\n'
    '\n'
    '# MT5 forex/indices symbols\n'
    'MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]\n'
    'MT5_TIMEFRAMES   = ["H1", "H4"]\n'
    'MT5_MIN_VOLUME   = 0.01\n'
)
if old_constants in content:
    content = content.replace(old_constants, new_constants)
    print("Constants updated")
else:
    print("WARNING: constants block not matched exactly -- patching line by line")
    content = content.replace(
        'DEMO_SCORE_THRESHOLD = 35',
        'DEMO_SCORE_THRESHOLD = 30   # aggressive demo\nSCAN_INTERVAL_SEC    = 30'
    )
    content = content.replace(
        'SCAN_SYMBOLS     = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]',
        'SCAN_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]'
    )
    content = content.replace(
        'SCAN_TIMEFRAMES  = ["1h", "4h"]',
        'SCAN_TIMEFRAMES = ["5m", "15m", "1h", "4h"]\n\n'
        '# MT5 forex/indices symbols\n'
        'MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]\n'
        'MT5_TIMEFRAMES   = ["H1", "H4"]\n'
        'MT5_MIN_VOLUME   = 0.01'
    )

# 2c. Add MT5Connector and _mt5_available to __init__
old_init = '        self.signal_agent   = SignalAgent(min_confidence=0.55)\n        self.binance        = BinanceConnector('
new_init = (
    '        self.signal_agent   = SignalAgent(min_confidence=0.55)\n'
    '        self.mt5            = MT5Connector(\n'
    '            login    = config.mt5_login,\n'
    '            password = config.mt5_password,\n'
    '            server   = config.mt5_server,\n'
    '        )\n'
    '        self._mt5_available = False\n'
    '        self.binance        = BinanceConnector('
)
if old_init in content:
    content = content.replace(old_init, new_init)
    print("MT5Connector added to __init__")

# 2d. Fix demo trade message -- replace old execute method
old_demo_start = '    async def _execute_demo_trade(self, signal: TradeSignal):\n'
old_demo_end   = '\n    async def _market_scan_loop(self):'
idx_s = content.find(old_demo_start)
idx_e = content.find(old_demo_end, idx_s)
if idx_s > 0 and idx_e > 0:
    new_demo = '''    async def _execute_demo_trade(self, signal: TradeSignal):
        """Record a simulated demo trade and notify via Telegram."""
        if len(self._demo_trades) >= DEMO_MAX_POSITIONS:
            return

        demo = DemoTrade(signal, signal.decision_score)
        self._demo_trades.append(demo)

        direction = "long" if signal.signal_type == SignalType.LONG else "short"
        market    = "MT5" if signal.symbol in MT5_SYMBOLS else "Binance"

        print(f"[DEMO TRADE] {signal.symbol} {direction.upper()} "
              f"entry={signal.entry:.4f} score={signal.decision_score} "
              f"({market}) [{len(self._demo_trades)}/{DEMO_MAX_POSITIONS}]")

        try:
            await self.telegram.send_signal_demo(
                symbol    = signal.symbol,
                direction = direction,
                entry     = signal.entry,
                sl        = signal.stop_loss if signal.stop_loss else signal.entry * 0.995,
                tp        = signal.take_profit,
                score     = signal.decision_score,
                timeframe = signal.timeframe,
                market    = market,
            )
        except Exception:
            pass
'''
    content = content[:idx_s] + new_demo + content[idx_e:]
    print("_execute_demo_trade replaced cleanly")

# 2e. Add _scan_mt5_symbol method before _execute_demo_trade
mt5_method = '''
    async def _scan_mt5_symbol(self, symbol: str, timeframe: str):
        """Fetch MT5 OHLCV, run SMC lite, return signal or None."""
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, lambda: self.mt5.get_ohlcv(symbol, timeframe, 200))
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
target = '    async def _execute_demo_trade(self, signal: TradeSignal):'
if mt5_method.strip() not in content:
    content = content.replace(target, mt5_method + target)
    print("_scan_mt5_symbol added")

# 2f. Add MT5 startup check and MT5 scan block
# Update run() to check MT5
old_run_print = (
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
new_run_print = (
    '        if self.demo_mode:\n'
    '            print(f"  DEMO MODE:     threshold={DEMO_SCORE_THRESHOLD} | max_trades={DEMO_MAX_POSITIONS}")\n'
    '            print(f"  Crypto:        {\', \'.join(SCAN_SYMBOLS)}")\n'
    '            print(f"  Timeframes:    {\', \'.join(SCAN_TIMEFRAMES)}")\n'
    '            print(f"  Scan interval: {SCAN_INTERVAL_SEC}s")\n'
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
if old_run_print in content:
    content = content.replace(old_run_print, new_run_print)
    print("run() MT5 check added")

# 2g. Update scan loop sleep
content = content.replace(
    'await asyncio.sleep(60)  # next full scan in 60s',
    'await asyncio.sleep(SCAN_INTERVAL_SEC)  # next full scan'
)
content = content.replace(
    "print('[Scan] Escaneando mercados...')",
    "print(f'[Scan] Escaneando {len(SCAN_SYMBOLS)} pares x {len(SCAN_TIMEFRAMES)} TF...')"
)

# 2h. Add MT5 scan block to scan loop (if not present)
old_rate_limit = '                        await asyncio.sleep(1)  # rate limit between symbols\n\n            except asyncio.CancelledError:'
if '# MT5 forex/indices scan' not in content:
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
        print("MT5 scan block added")
else:
    print("MT5 scan block already present")

# 3. Verify and write
try:
    ast.parse(content)
    print("Final content parses OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
    lines = content.splitlines()
    if e.lineno:
        start = max(0, e.lineno - 3)
        for i, l in enumerate(lines[start:e.lineno+2], start+1):
            print(f"  {i}: {repr(l)}")
    exit(1)

open('core/supervisor.py', 'w', encoding='utf-8').write(content)
print("supervisor.py written cleanly (UTF-8 no BOM)")
