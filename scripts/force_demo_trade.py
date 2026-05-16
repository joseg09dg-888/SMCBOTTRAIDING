"""
force_demo_trade.py — Ejecuta un ciclo de análisis completo ahora mismo.
Uso: .venv/Scripts\python scripts/force_demo_trade.py [SYMBOL] [TIMEFRAME]

Ejemplo:
  .venv/Scripts\python scripts/force_demo_trade.py BTCUSDT 1h
  .venv/Scripts\python scripts/force_demo_trade.py ETHUSDT 4h
"""
import sys
import asyncio
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

DEMO_THRESHOLD = 35  # demo mode threshold


MT5_SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "NAS100", "US30"]


async def run(symbol: str = "BTCUSDT", timeframe: str = "1h", market: str = ""):
    use_mt5 = market.upper() == "MT5" or symbol.upper() in MT5_SYMBOLS
    market_label = "MT5" if use_mt5 else "Binance Testnet"

    print(f"\n{'='*55}")
    print(f"  FORCE DEMO TRADE — {symbol} | {timeframe} | {market_label}")
    print(f"{'='*55}\n")

    from core.config import config

    # --- 1. Fetch real data ---
    if use_mt5:
        from connectors.metatrader_connector import MT5Connector
        print(f"[1/5] Conectando a MT5...")
        mt5c = MT5Connector(config.mt5_login, config.mt5_password, config.mt5_server)
        if not mt5c.connect():
            print(f"  FAIL: {mt5c.last_error_msg()}")
            print(f"\n  SOLUCION: En MT5, activa el boton 'Algo Trading' (rayo verde).")
            return
        # Map timeframe to MT5 format
        tf_mt5 = timeframe.upper().replace("1H", "H1").replace("4H", "H4").replace("1D", "D1")
        if "1h" in timeframe.lower(): tf_mt5 = "H1"
        if "4h" in timeframe.lower(): tf_mt5 = "H4"
        print(f"  MT5 conectado. Descargando {symbol} {tf_mt5}...")
        df = mt5c.get_ohlcv(symbol, tf_mt5, 200)
    else:
        from connectors.binance_connector import BinanceConnector
        binance = BinanceConnector(
            api_key    = config.binance_api_key,
            api_secret = config.binance_api_secret,
            testnet    = config.binance_testnet,
        )
        print(f"[1/5] Descargando datos {symbol} {timeframe} desde {market_label}...")
        df = binance.get_ohlcv(symbol, timeframe, limit=200)

    if df is None or df.empty:
        print(f"ERROR: No se obtuvieron datos.")
        return

    current_price = float(df["close"].iloc[-1])
    print(f"      OK — {len(df)} velas | precio actual: {current_price:,.4f}")

    # --- 2. SMC analysis ---
    print(f"\n[2/5] Análisis SMC técnico...")
    from smc.structure import MarketStructure
    from smc.orderblocks import OrderBlockDetector, FVGDetector

    ms       = MarketStructure(df)
    struct   = ms.analyze()
    ob_det   = OrderBlockDetector(df)
    fvg_det  = FVGDetector(df)
    bull_obs  = ob_det.find_bullish_obs()
    bear_obs  = ob_det.find_bearish_obs()
    bull_fvgs = fvg_det.find_bullish_fvg()
    bear_fvgs = fvg_det.find_bearish_fvg()
    bos_list  = ms.detect_bos()
    choch_list = ms.detect_choch()

    print(f"      Estructura: {struct.structure_type.value}")
    print(f"      Bias: {struct.bias.upper()}")
    print(f"      Order Blocks: {len(bull_obs)} alcistas | {len(bear_obs)} bajistas")
    print(f"      FVGs: {len(bull_fvgs)} alcistas | {len(bear_fvgs)} bajistas")
    print(f"      BOS: {len(bos_list)} | CHoCH: {len(choch_list)}")

    # --- 3. Generate signal ---
    print(f"\n[3/5] Generando señal...")
    from agents.signal_agent import SignalAgent, SignalType

    is_bullish = struct.bias == "bullish"
    is_bearish = struct.bias == "bearish"
    # For neutral bias, derive direction from last BOS/CHoCH
    if not (is_bullish or is_bearish) and bos_list:
        last_dir = bos_list[-1].get("direction", "")
        if last_dir == "bullish": is_bullish = True
        elif last_dir == "bearish": is_bearish = True
    if not (is_bullish or is_bearish) and choch_list:
        last_choch = choch_list[-1].get("direction", "")
        if last_choch == "bullish": is_bullish = True
        elif last_choch == "bearish": is_bearish = True

    poi_zones = []
    for ob in ((bull_obs if is_bullish else bear_obs) or bull_obs + bear_obs)[:3]:
        poi_zones.append(ob)

    direction_word = "bullish" if is_bullish else ("bearish" if is_bearish else "neutral")
    analysis_text = f"{direction_word} trend {struct.structure_type.value}"
    if bos_list:   analysis_text += " BOS confirmado ✅"
    if choch_list: analysis_text += " CHoCH detectado ✅"
    if poi_zones:  analysis_text += " order block presente ✅ setup válido"

    agent = SignalAgent(min_confidence=0.55)
    signal = agent.evaluate(
        analysis_text = analysis_text,
        symbol        = symbol,
        timeframe     = timeframe,
        current_price = current_price,
        poi_zones     = poi_zones,
    )

    if signal.signal_type == SignalType.WAIT:
        print(f"      Resultado: SIN SETUP — no hay señal técnica clara")
        print(f"      Motivo: {signal.trigger}")
    else:
        direction = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"
        print(f"      Señal: {direction}")
        print(f"      Entry: {signal.entry:,.5f}")
        print(f"      SL:    {signal.stop_loss:,.5f}")
        print(f"      TP:    {signal.take_profit:,.5f}")
        print(f"      R:R:   1:{signal.risk_reward:.1f}")

    # --- 4. DecisionFilter ---
    print(f"\n[4/5] DecisionFilter...")
    from core.risk_manager import RiskManager
    from core.decision_filter import DecisionFilter, TradeGrade
    from core.config import config as cfg

    rm = RiskManager(cfg, capital=1000.0)
    df_filter = DecisionFilter(cfg, rm)

    if signal.signal_type != SignalType.WAIT and signal.stop_loss:
        result = df_filter.evaluate(
            df         = df,
            symbol     = symbol,
            timeframe  = timeframe,
            entry      = signal.entry,
            stop_loss  = signal.stop_loss,
            take_profit = signal.take_profit,
            bias       = struct.bias,
        )
        signal.decision_score = result.score
        signal.decision_grade = result.grade.value

        print(f"      Score total: {result.score}/100")
        print(f"      Grade: {result.grade.value.upper()}")
        print(f"      Breakdown: {result.breakdown}")
        print(f"      Motivo: {result.reason}")
    else:
        result = None
        print(f"      No aplica (sin señal válida)")

    # --- 5. Decision ---
    print(f"\n[5/5] Decisión demo (threshold={DEMO_THRESHOLD})...")
    score = signal.decision_score if result else 0

    if signal.signal_type == SignalType.WAIT:
        print(f"\n  ⛔ SIN TRADE — No hay setup técnico ({struct.bias} bias sin confluencias)")
    elif score < DEMO_THRESHOLD:
        print(f"\n  ⚠️  SCORE INSUFICIENTE — {score}/100 < {DEMO_THRESHOLD} demo threshold")
        print(f"     No se ejecuta. Espera mejor setup.")
    else:
        direction = "🟢 LONG" if signal.signal_type == SignalType.LONG else "🔴 SHORT"
        grade_emoji = {"premium":"🔥","full":"✅","reduced":"⚡"}.get(signal.decision_grade, "🟡")

        print(f"\n  {grade_emoji} TRADE DEMO EJECUTADO")
        print(f"  {'='*50}")
        print(f"  Par: {symbol} | TF: {timeframe}")
        print(f"  {direction}")
        print(f"  Entry:   {signal.entry:,.5f}")
        print(f"  SL:      {signal.stop_loss:,.5f}")
        print(f"  TP:      {signal.take_profit:,.5f}")
        print(f"  R:R:     1:{signal.risk_reward:.1f}")
        print(f"  Score:   {score}/100 | {signal.decision_grade.upper()}")
        print(f"  {'='*50}")

        # Send Telegram
        from dashboard.telegram_bot import TradingTelegramBot
        bot = TradingTelegramBot()
        msg = (
            f"🚀 *TRADE DEMO — {symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{direction} | {timeframe}\n"
            f"Entry: {signal.entry:,.5f}\n"
            f"SL: {signal.stop_loss:,.5f}\n"
            f"TP: {signal.take_profit:,.5f}\n"
            f"R:R: 1:{signal.risk_reward:.1f}\n"
            f"Score: {score}/100 | Trigger: {signal.trigger}\n"
            f"SMC: {'✅ OB' if poi_zones else '❌'} | "
            f"BOS: {'✅' if bos_list else '❌'} | "
            f"FVG: {'✅' if bull_fvgs or bear_fvgs else '❌'}\n"
            f"💡 Modo DEMO — sin dinero real"
        )
        try:
            await bot.send_glint_alert(msg)
            print(f"\n  Telegram: mensaje enviado")
        except Exception as e:
            print(f"\n  Telegram: {e}")


if __name__ == "__main__":
    symbol    = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSDT"
    timeframe = sys.argv[2].lower() if len(sys.argv) > 2 else "1h"
    market    = sys.argv[3].upper() if len(sys.argv) > 3 else ""
    asyncio.run(run(symbol, timeframe, market))



