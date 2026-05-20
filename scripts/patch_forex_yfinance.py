"""Add yfinance forex scan to supervisor.py as fallback when MT5 not available."""
import ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

content = open('core/supervisor.py', encoding='utf-8').read()

# 1. Add yfinance forex symbols
old_mt5_syms = '# MT5 forex/indices symbols\nMT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]\nMT5_TIMEFRAMES   = ["H1", "H4"]\nMT5_MIN_VOLUME   = 0.01'

new_mt5_syms = '''# MT5 forex/indices symbols
MT5_SYMBOLS      = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]
MT5_TIMEFRAMES   = ["H1", "H4"]
MT5_MIN_VOLUME   = 0.01

# yfinance forex symbols (fallback when MT5 unavailable)
YFINANCE_FOREX   = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
                    "USDJPY": "USDJPY=X", "GBPJPY": "GBPJPY=X"}
YFINANCE_TF      = ["1h", "4h"]'''

if old_mt5_syms in content:
    content = content.replace(old_mt5_syms, new_mt5_syms)
    print("Added YFINANCE_FOREX constant")
else:
    print("WARNING: MT5_SYMBOLS block not found - adding manually")
    content = content.replace(
        'MT5_MIN_VOLUME   = 0.01',
        'MT5_MIN_VOLUME   = 0.01\n\n# yfinance forex symbols (fallback when MT5 unavailable)\nYFINANCE_FOREX   = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "GBPJPY": "GBPJPY=X"}\nYFINANCE_TF      = ["1h", "4h"]'
    )

# 2. Add _scan_forex_yfinance method before _execute_demo_trade
yf_method = '''
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
                df = df.rename(columns={"Open":"open","High":"high",
                                        "Low":"low","Close":"close","Volume":"volume"})
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
if '# yfinance forex symbols' in content and '_scan_forex_yfinance' not in content:
    content = content.replace(target, yf_method + target)
    print("_scan_forex_yfinance method added")
elif '_scan_forex_yfinance' in content:
    print("_scan_forex_yfinance already present")

# 3. Add yfinance scan block to _market_scan_loop
# Find the MT5 scan block and add yfinance before/after it
yf_scan_block = '''
                # yfinance forex scan (always runs — no MT5 required)
                for symbol, tf_yf, tf_label in [
                    ("EURUSD","1h","1H"), ("GBPUSD","1h","1H"),
                    ("USDJPY","1h","1H"), ("GBPJPY","1h","1H"),
                ]:
                    if not self._running: break
                    try:
                        signal = await self._scan_forex_yfinance(symbol, tf_yf, tf_label)
                        if signal is None:
                            continue
                        score = signal.decision_score
                        bias  = signal.signal_type.value.upper()
                        print(f"[FOREX][{symbol}][{tf_label}] Score: {score} | {bias}", end="")
                        if signal.signal_type == SignalType.WAIT or score < threshold:
                            print(" -- sin setup")
                        elif self.demo_mode:
                            print(f" -- DEMO FOREX")
                            await self._execute_demo_trade(signal)
                        else:
                            self._dispatch(signal)
                    except Exception as exc:
                        print(f"[FOREX][{symbol}] Error: {exc.__class__.__name__}")
                    await asyncio.sleep(0.5)

'''

# Insert before the MT5 scan block or before the except CancelledError
insert_before = '                # MT5 forex/indices scan'
if insert_before in content and 'yfinance forex scan' not in content:
    content = content.replace(insert_before, yf_scan_block + insert_before)
    print("yfinance scan block inserted before MT5 block")
elif 'yfinance forex scan' in content:
    print("yfinance scan block already present")
else:
    # Fallback: insert before except CancelledError
    insert_before2 = '            except asyncio.CancelledError:'
    if insert_before2 in content and 'yfinance forex scan' not in content:
        content = content.replace(insert_before2, yf_scan_block + insert_before2)
        print("yfinance scan block inserted before CancelledError")

# Verify
ast.parse(content)
print("supervisor.py parses OK")
open('core/supervisor.py', 'w', encoding='utf-8').write(content)
print("supervisor.py written")
