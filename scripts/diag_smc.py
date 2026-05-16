"""Diagnostic: check SMC signal generation for BTCUSDT 1h."""
import sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.chdir(r'C:\Users\jose-\projects\trading_agent')
from dotenv import load_dotenv
load_dotenv()
from connectors.binance_connector import BinanceConnector
from core.config import config
from smc.structure import MarketStructure
from smc.orderblocks import OrderBlockDetector, FVGDetector
from agents.signal_agent import SignalAgent

b = BinanceConnector(config.binance_api_key, config.binance_api_secret, True)
df = b.get_ohlcv('BTCUSDT', '1h', 200)
ms   = MarketStructure(df)
struct = ms.analyze()
bos  = ms.detect_bos()
choch = ms.detect_choch()
bull_obs = OrderBlockDetector(df).find_bullish_obs()
bear_obs = OrderBlockDetector(df).find_bearish_obs()

print(f"bias={struct.bias}  structure={struct.structure_type.value}")
print(f"BOS={len(bos)}  CHoCH={len(choch)}")
if bos:
    print(f"Last BOS direction: {bos[-1].get('direction')}")
if choch:
    print(f"Last CHoCH direction: {choch[-1].get('direction')}")

is_bull = struct.bias == "bullish"
is_bear = struct.bias == "bearish"

if not (is_bull or is_bear) and bos:
    d = bos[-1].get("direction", "")
    if d == "bullish":   is_bull = True
    elif d == "bearish": is_bear = True

if not (is_bull or is_bear) and choch:
    d = choch[-1].get("direction", "")
    if d == "bullish":   is_bull = True
    elif d == "bearish": is_bear = True

dw  = "bullish" if is_bull else ("bearish" if is_bear else "neutral")
obs = (bull_obs if is_bull else bear_obs)
poi = obs[:3]
at  = f"{dw} trend"
if bos:  at += " BOS confirmado"
if poi:  at += " order block presente setup valido"

print(f"\nAfter BOS fix => is_bull={is_bull} is_bear={is_bear}")
print(f"POI zones: {len(poi)} | analysis_text: '{at}'")

cp  = float(df["close"].iloc[-1])
sig = SignalAgent(0.55).evaluate(at, "BTCUSDT", "1h", cp, poi)
print(f"\nsignal type:  {sig.signal_type.value}")
print(f"entry:        {sig.entry:.2f}")
print(f"confidence:   {sig.confidence}")
print(f"is_valid:     {sig.is_valid()}")
