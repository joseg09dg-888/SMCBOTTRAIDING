"""Clean patch: add save_score to supervisor.py via deep_fix approach."""
import subprocess, ast, os, re
os.chdir(r'C:\Users\jose-\projects\trading_agent')

# Get clean base
result = subprocess.run(['git', 'show', '43561ac:core/supervisor.py'], capture_output=True)
content = result.stdout.decode('utf-8-sig')
ast.parse(content)
print(f"Clean base: {len(content)} chars")

# Apply all previous patches (same as deep_fix_supervisor.py but updated)

# 1. Add imports
content = content.replace(
    'logger = logging.getLogger(__name__)',
    'logger = logging.getLogger(__name__)\nfrom core.score_db import save_score'
)
content = content.replace(
    'from connectors.binance_connector import BinanceConnector\nfrom connectors.glint_connector import GlintSignal',
    'from connectors.binance_connector import BinanceConnector\nfrom connectors.metatrader_connector import MT5Connector\nfrom connectors.glint_connector import GlintSignal'
)

# 2. Constants
content = content.replace(
    '# Demo mode: lower score threshold so the bot actually trades while learning\n'
    'DEMO_SCORE_THRESHOLD = 35   # instead of 60 -- generates more trades for training\n'
    'DEMO_MAX_POSITIONS   = 5    # maximum simultaneous demo trades\n'
    '\n'
    '# Symbols and timeframes to scan\n'
    'SCAN_SYMBOLS     = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]\n'
    'SCAN_TIMEFRAMES  = ["1h", "4h"]\n',
    '# Demo mode: lower score threshold so the bot actually trades while learning\n'
    'DEMO_SCORE_THRESHOLD = 30   # aggressive demo\n'
    'DEMO_MAX_POSITIONS   = 5\n'
    'SCAN_INTERVAL_SEC    = 30\n'
    '\n'
    '# Symbols and timeframes to scan\n'
    'SCAN_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]\n'
    'SCAN_TIMEFRAMES = ["5m", "15m", "1h", "4h"]\n'
    '\n'
    '# MT5 forex/indices symbols\n'
    'MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]\n'
    'MT5_TIMEFRAMES   = ["H1", "H4"]\n'
    'MT5_MIN_VOLUME   = 0.01\n'
    '\n'
    '# yfinance forex symbols (fallback when MT5 unavailable)\n'
    'YFINANCE_FOREX   = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",\n'
    '                    "USDJPY": "USDJPY=X", "GBPJPY": "GBPJPY=X"}\n'
    'YFINANCE_TF      = ["1h", "4h"]\n'
)

# 3. Add MT5 + signal_agent to __init__
content = content.replace(
    '        self.signal_agent   = SignalAgent(min_confidence=0.55)\n        self.binance        = BinanceConnector(',
    '        self.signal_agent   = SignalAgent(min_confidence=0.55)\n'
    '        self.mt5            = MT5Connector(\n'
    '            login    = config.mt5_login,\n'
    '            password = config.mt5_password,\n'
    '            server   = config.mt5_server,\n'
    '        )\n'
    '        self._mt5_available = False\n'
    '        self.binance        = BinanceConnector('
)

# 4. Add _scan_mt5_symbol + _scan_forex_yfinance methods before _execute_demo_trade
mt5_methods = '''
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

    async def _scan_forex_yfinance(self, symbol: str, tf_yf: str, tf_label: str):
        """Scan forex pair using yfinance data (MT5 fallback)."""
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()
            ticker_sym = YFINANCE_FOREX.get(symbol, symbol + "=X")
            def _fetch():
                period_map = {"1h": "5d", "4h": "30d"}
                period = period_map.get(tf_yf, "5d")
                t = yf.Ticker(ticker_sym)
                df = t.history(period=period, interval=tf_yf)
                if df.empty:
                    return None
                df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                return df[["open","high","low","close","volume"]].copy()
            df = await loop.run_in_executor(None, _fetch)
            if df is None or len(df) < 50:
                return None
            smc = self._run_smc_lite(df)
            current_price = float(df["close"].iloc[-1])
            signal = self.signal_agent.evaluate(
                analysis_text=smc["analysis_text"], symbol=symbol,
                timeframe=tf_label, current_price=current_price,
                poi_zones=smc["poi_zones"], glint_context=self._last_glint_text,
            )
            if signal.signal_type == SignalType.WAIT:
                return signal
            signal = self.route_signal(signal, df)
            return signal
        except Exception as exc:
            logger.debug(f"yfinance scan {symbol} error: {exc}")
            return None

'''

target = '    async def _execute_demo_trade(self, signal: TradeSignal):'
if '_scan_mt5_symbol' not in content:
    content = content.replace(target, mt5_methods + target)

# 5. Replace _execute_demo_trade with clean version + save_score
old_demo_start = '    async def _execute_demo_trade(self, signal: TradeSignal):\n'
old_demo_end   = '\n    async def _market_scan_loop(self):'
idx_s = content.find(old_demo_start)
idx_e = content.find(old_demo_end, idx_s)
if idx_s > 0 and idx_e > 0:
    new_demo = '''    async def _execute_demo_trade(self, signal: TradeSignal):
        """Record a simulated demo trade, notify via Telegram, log to SQLite."""
        if len(self._demo_trades) >= DEMO_MAX_POSITIONS:
            return

        demo = DemoTrade(signal, signal.decision_score)
        self._demo_trades.append(demo)

        direction = "long" if signal.signal_type == SignalType.LONG else "short"
        market    = "MT5" if signal.symbol in MT5_SYMBOLS else "Binance"

        # Log to SQLite for /scores and /criterios commands
        save_score(
            symbol    = signal.symbol,
            timeframe = signal.timeframe,
            score     = signal.decision_score,
            direction = direction,
            entry     = signal.entry,
            sl        = signal.stop_loss if signal.stop_loss else 0.0,
            tp        = signal.take_profit,
            executed  = True,
        )

        print(f"[DEMO TRADE] {signal.symbol} {direction.upper()} "
              f"entry={signal.entry:.4f} score={signal.decision_score} "
              f"({market}) [{len(self._demo_trades)}/{DEMO_MAX_POSITIONS}]")

        msg = (
            f"<b>TRADE DEMO ABIERTO</b>\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"Par: <b>{signal.symbol}</b> | {signal.timeframe} | {market}\\n"
            f"{'LONG' if direction=='long' else 'SHORT'}\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"Entrada:    <code>{signal.entry:,.5f}</code>\\n"
            f"Stop Loss:  <code>{signal.stop_loss if signal.stop_loss else 0.0:,.5f}</code>\\n"
            f"Take Profit:<code>{signal.take_profit:,.5f}</code>\\n"
            f"R:R: <code>1:{signal.risk_reward:.1f}</code>\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"Score: <b>{signal.decision_score}/100</b>\\n"
            f"Activos: {len(self._demo_trades)}/{DEMO_MAX_POSITIONS}\\n"
            f"DEMO - sin dinero real"
        )
        try:
            await self.telegram.send_signal_demo(
                symbol=signal.symbol, direction=direction,
                entry=signal.entry,
                sl=signal.stop_loss if signal.stop_loss else signal.entry*0.995,
                tp=signal.take_profit, score=signal.decision_score,
                timeframe=signal.timeframe, market=market,
            )
        except Exception:
            pass
'''
    content = content[:idx_s] + new_demo + content[idx_e:]
    print("_execute_demo_trade replaced with save_score")

# 6. Add MT5 startup check + forex scan to _market_scan_loop
# Update run() method
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
if old_run in content:
    content = content.replace(old_run, new_run)
    print("run() MT5 check added")

# 7. Update scan loop sleep + add forex + MT5 blocks
# Find and update sleep
content = content.replace(
    'await asyncio.sleep(60)  # next full scan in 60s',
    'await asyncio.sleep(SCAN_INTERVAL_SEC)  # next full scan'
)

# Add scan_forex block + MT5 block if not present
old_rate_limit = '                        await asyncio.sleep(1)  # rate limit between symbols\n\n            except asyncio.CancelledError:'
if 'yfinance forex scan' not in content and old_rate_limit in content:
    forex_mt5_block = (
        '                        await asyncio.sleep(1)  # rate limit between symbols\n\n'
        '                # yfinance forex scan (always runs)\n'
        '                for symbol, tf_yf, tf_label in [\n'
        '                    ("EURUSD","1h","1H"), ("GBPUSD","1h","1H"),\n'
        '                    ("USDJPY","1h","1H"), ("GBPJPY","1h","1H"),\n'
        '                ]:\n'
        '                    if not self._running: break\n'
        '                    try:\n'
        '                        signal = await self._scan_forex_yfinance(symbol, tf_yf, tf_label)\n'
        '                        if signal is None:\n'
        '                            continue\n'
        '                        score = signal.decision_score\n'
        '                        bias  = signal.signal_type.value.upper()\n'
        '                        print(f"[FOREX][{symbol}][{tf_label}] Score: {score} | {bias}", end="")\n'
        '                        if signal.signal_type == SignalType.WAIT or score < threshold:\n'
        '                            print(" -- sin setup")\n'
        '                        elif self.demo_mode:\n'
        '                            print(f" -- DEMO FOREX")\n'
        '                            await self._execute_demo_trade(signal)\n'
        '                        else:\n'
        '                            self._dispatch(signal)\n'
        '                    except Exception as exc:\n'
        '                        print(f"[FOREX][{symbol}] Error: {exc.__class__.__name__}")\n'
        '                    await asyncio.sleep(0.5)\n\n'
        '            except asyncio.CancelledError:'
    )
    content = content.replace(old_rate_limit, forex_mt5_block)
    print("forex scan block added")

ast.parse(content)
print("Final content parses OK")
open('core/supervisor.py', 'w', encoding='utf-8').write(content)
print("supervisor.py written cleanly")
