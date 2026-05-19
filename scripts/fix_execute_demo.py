"""Fix _execute_demo_trade — replace corrupted body with clean version."""
import os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

path = 'core/supervisor.py'
content = open(path, encoding='utf-8').read()

# Find start/end of the corrupted method body (lines 407-430 range)
start_marker = '    async def _execute_demo_trade(self, signal: TradeSignal):\n'
end_marker   = '\n    # '  # next section comment

idx_start = content.find(start_marker)
if idx_start < 0:
    print("ERROR: _execute_demo_trade not found")
    exit(1)

idx_end = content.find(end_marker, idx_start + 200)
if idx_end < 0:
    print("ERROR: end marker not found")
    exit(1)

new_method = '''    async def _execute_demo_trade(self, signal: TradeSignal):
        """Record a simulated trade and send Telegram notification."""
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

patched = content[:idx_start] + new_method + content[idx_end:]
open(path, 'w', encoding='utf-8').write(patched)

# Verify
import ast
ast.parse(open(path, encoding='utf-8').read())
print("supervisor.py fixed and parses OK")
print(f"New method around idx {idx_start}")
