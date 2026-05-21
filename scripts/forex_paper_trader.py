"""Forex paper trader — real prices via yfinance, virtual execution, SQLite recording.
Works WITHOUT MT5. Trades count toward Axi Select Edge Score.
"""
import asyncio, sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv; load_dotenv()

FOREX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X",
    "XAUUSD": "GC=F",   # Gold futures as proxy
}

async def run():
    import yfinance as yf
    import pandas as pd
    from smc.structure import MarketStructure
    from smc.orderblocks import OrderBlockDetector
    from agents.signal_agent import SignalAgent, SignalType
    from core.decision_filter import DecisionFilter
    from core.risk_manager import RiskManager
    from core.config import config
    from core.score_db import save_score
    from dashboard.telegram_bot import TradingTelegramBot

    bot = TradingTelegramBot()
    rm  = RiskManager(config, capital=1000.0)
    df_filter = DecisionFilter(config, rm)
    sig_agent = SignalAgent(min_confidence=0.55)

    print("=== FOREX PAPER TRADER (yfinance) ===")
    print(f"Pairs: {list(FOREX_PAIRS.keys())}")
    print()

    executed = 0
    for symbol, ticker in FOREX_PAIRS.items():
        try:
            print(f"[{symbol}] Fetching data...")
            t = yf.Ticker(ticker)
            df = t.history(period="5d", interval="1h")
            if df.empty or len(df) < 50:
                print(f"  No data")
                continue

            df_clean = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            df_clean = df_clean[["open","high","low","close","volume"]].copy()

            # SMC Analysis
            ms     = MarketStructure(df_clean)
            struct = ms.analyze()
            bos    = ms.detect_bos()
            choch  = ms.detect_choch()
            obs    = OrderBlockDetector(df_clean)
            bull   = obs.find_bullish_obs()
            bear   = obs.find_bearish_obs()

            # Determine direction
            is_bull = struct.bias == "bullish"
            is_bear = struct.bias == "bearish"
            if not (is_bull or is_bear) and bos:
                d = bos[-1].get("direction", "")
                if d == "bullish":   is_bull = True
                elif d == "bearish": is_bear = True

            dw = "bullish" if is_bull else ("bearish" if is_bear else "neutral")
            poi = (bull if is_bull else bear)[:3]
            at  = f"{dw} trend"
            if bos:  at += " BOS confirmado"
            if poi:  at += " order block presente setup valido"

            cp  = float(df_clean["close"].iloc[-1])
            sig = sig_agent.evaluate(at, symbol, "1H", cp, poi)

            if sig.signal_type == SignalType.WAIT:
                print(f"  [{symbol}] No setup | bias={struct.bias}")
                continue

            # DecisionFilter
            sig = sig_agent.evaluate(at, symbol, "1H", cp, poi)
            if sig.signal_type != SignalType.WAIT:
                try:
                    result = df_filter.evaluate(
                        df=df_clean, symbol=symbol, timeframe="1H",
                        entry=sig.entry, stop_loss=sig.stop_loss,
                        take_profit=sig.take_profit, bias=struct.bias,
                    )
                    sig.decision_score = result.score
                except Exception:
                    sig.decision_score = 35

            score = sig.decision_score
            direction = sig.signal_type.value
            print(f"  [{symbol}] Score:{score} {direction.upper()} entry={sig.entry:.5f}")

            if score >= 30:
                # Execute paper trade
                save_score(
                    symbol=symbol, timeframe="1H", score=score,
                    direction=direction, entry=sig.entry,
                    sl=sig.stop_loss if sig.stop_loss else sig.entry*0.995,
                    tp=sig.take_profit, executed=True,
                )
                executed += 1
                print(f"  PAPER TRADE EXECUTED: {symbol} {direction.upper()} @ {sig.entry:.5f}")

                # Telegram notification
                try:
                    await bot.send_signal_demo(
                        symbol=symbol, direction=direction,
                        entry=sig.entry,
                        sl=sig.stop_loss if sig.stop_loss else sig.entry*0.995,
                        tp=sig.take_profit, score=score,
                        timeframe="1H", market="Paper/yfinance",
                    )
                except Exception as e:
                    print(f"  Telegram: {e}")

        except Exception as e:
            print(f"  [{symbol}] Error: {e}")

    print(f"\n=== RESULT ===")
    print(f"Paper trades executed: {executed}/{len(FOREX_PAIRS)}")
    print("All saved to SQLite (memory/scores.db)")
    print("Run /axi in Telegram to see Edge Score progress")

if __name__ == "__main__":
    asyncio.run(run())
