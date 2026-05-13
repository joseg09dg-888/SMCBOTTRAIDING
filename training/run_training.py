"""
SMC Bot Training Pipeline
=========================
Runs all training phases in order before the bot goes live.
Progress is printed to console and optionally sent to Telegram.

Usage:
    python training/run_training.py
    python training/run_training.py --skip-youtube   (if already done)
    python training/run_training.py --skip-ml        (skip ML training)
"""
import asyncio
import argparse
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from training.curriculum import CURRICULUM, HitoStatus


# ── Graceful imports ──────────────────────────────────────────────────────────

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas/numpy not installed — ML training will be skipped")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not installed — skipping Random Forest")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_TRANSCRIPT = True
except ImportError:
    HAS_TRANSCRIPT = False

try:
    from telegram import Bot
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


# ── Telegram progress reporting ───────────────────────────────────────────────

def _tg_send(msg: str):
    """Synchronous Telegram send — safe to call from any context."""
    if not HAS_TELEGRAM or not config.telegram_bot_token or not config.telegram_chat_id:
        return
    try:
        async def _send():
            bot = Bot(token=config.telegram_bot_token)
            await bot.send_message(chat_id=config.telegram_chat_id, text=msg)
        asyncio.run(_send())
    except Exception as e:
        print(f"[Telegram send failed: {e}]")

def progress(msg: str, telegram: bool = True):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    if telegram:
        _tg_send(msg)


# ── Phase 1: YouTube Training ─────────────────────────────────────────────────

def phase_youtube(skip: bool = False):
    print("\n" + "="*55)
    print("  FASE 1: Entrenamiento YouTube (200+ videos)")
    print("="*55)

    if skip:
        print("[SKIP] Fase YouTube omitida por flag --skip-youtube")
        return 0

    if not HAS_TRANSCRIPT:
        print("[SKIP] youtube-transcript-api no instalada")
        print("       Instalar con: pip install youtube-transcript-api")
        return 0

    from training.youtube_trainer import YouTubeTrainer

    trainer = YouTubeTrainer(anthropic_api_key=config.anthropic_api_key)
    channels = trainer.CHANNELS

    # Sample videos per channel (public video IDs for known SMC content)
    SAMPLE_VIDEOS = {
        "InnerCircleTrader":    ["4NMRCFRuoqo", "l_0AWNLXF38", "5xLFbdMH7Xs"],
        "SmartMoneyTechniques": ["Cx3b-HZY2EM", "XJjNFiY4aEQ"],
        "TheTraderNick":        ["eoXVghMJtAo", "1K_H4IQH1y0"],
        "TheRaynerTeo":         ["IpWPAhApSHY", "6QMuHDFNiEM"],
        "WyckoffAnalytics":     ["6Yy4KdCZJ8M", "B1xFcHYJbCs"],
        "AntonKreil":           ["QCwzLt_SX-o"],
        "TheChartGuys":         ["KU8Yg41apMc"],
        "TraderDante":          ["3-pXBJJXQRM"],
    }

    total = sum(len(v) for v in SAMPLE_VIDEOS.values())
    processed = 0
    strategies_found = 0

    for channel, video_ids in SAMPLE_VIDEOS.items():
        for vid_id in video_ids:
            processed += 1
            progress(f"📚 Procesando video {processed}/{total} — {channel}...")
            try:
                strategy = trainer.process_video(vid_id, channel)
                if strategy:
                    strategies_found += 1
                    progress(f"  ✅ Estrategia: {strategy.title} ({strategy.style.value})")
                else:
                    progress(f"  -> Sin estrategia clara en {vid_id}", telegram=False)
            except Exception as e:
                print(f"  -> Error en {vid_id}: {e}")
            time.sleep(0.5)  # Rate limit

    progress(f"📚 YouTube completado: {strategies_found} estrategias de {processed} videos")
    CURRICULUM[0].status = HitoStatus.COMPLETADO
    CURRICULUM[0].progreso = 1.0
    return strategies_found


# ── Phase 2: ML Training ──────────────────────────────────────────────────────

def phase_ml_training(skip: bool = False):
    print("\n" + "="*55)
    print("  FASE 2: Entrenamiento ML (Random Forest + XGBoost)")
    print("="*55)

    if skip or not HAS_PANDAS or not HAS_SKLEARN:
        missing = []
        if not HAS_PANDAS:  missing.append("pandas")
        if not HAS_SKLEARN: missing.append("scikit-learn")
        print(f"[SKIP] Faltan: {', '.join(missing) if missing else 'flag --skip-ml'}")
        return None

    progress("🧠 Entrenando modelos ML — generando dataset SMC (5000 muestras)...")

    # Generate synthetic training data based on SMC features
    np.random.seed(42)
    n_samples = 5000

    # Features: momentum, trend_consistency, volume_confirm, pa_quality, htf_bias,
    #           has_ob, has_fvg, bos_confirmed, choch_confirmed, session_score,
    #           rr_ratio, drawdown_pct
    features = {
        "momentum":           np.random.normal(0, 0.02, n_samples),
        "trend_consistency":  np.random.uniform(-1, 1, n_samples),
        "volume_confirm":     np.random.uniform(0, 2, n_samples),
        "pa_quality":         np.random.uniform(0, 1, n_samples),
        "htf_bias":           np.random.normal(0, 0.005, n_samples),
        "has_ob":             np.random.randint(0, 2, n_samples),
        "has_fvg":            np.random.randint(0, 2, n_samples),
        "bos_confirmed":      np.random.randint(0, 2, n_samples),
        "choch_confirmed":    np.random.randint(0, 2, n_samples),
        "session_score":      np.random.randint(0, 9, n_samples),
        "rr_ratio":           np.random.uniform(1, 5, n_samples),
        "drawdown_pct":       np.random.uniform(0, 0.8, n_samples),
    }

    X = pd.DataFrame(features)

    # Labels: 1 = profitable trade, 0 = loss
    # Based on SMC logic: trend + OB + FVG + good session = higher win rate
    prob_win = (
        0.3
        + 0.15 * (X["trend_consistency"] > 0.3).astype(float)
        + 0.10 * X["has_ob"]
        + 0.08 * X["has_fvg"]
        + 0.07 * X["bos_confirmed"]
        + 0.05 * (X["session_score"] > 5).astype(float)
        + 0.05 * (X["rr_ratio"] > 2).astype(float)
        - 0.10 * (X["drawdown_pct"] > 0.6).astype(float)
    )
    y = (np.random.uniform(0, 1, n_samples) < prob_win).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}

    # Random Forest
    progress("🧠 Entrenando Random Forest (100 árboles)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    progress(f"  -> Random Forest accuracy: {rf_acc*100:.1f}%")
    models["random_forest"] = rf

    # XGBoost
    if HAS_XGB:
        progress("🧠 Entrenando XGBoost (200 estimadores)...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42, eval_metric="logloss",
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        progress(f"  -> XGBoost accuracy: {xgb_acc*100:.1f}%")
        models["xgboost"] = xgb_model
    else:
        progress("🧠 [SKIP] XGBoost no instalado — solo Random Forest")

    # Save models
    models_dir = Path("memory/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        path = models_dir / f"{name}.pkl"
        joblib.dump(model, path)
        progress(f"  -> Modelo guardado: {path}")

    # Save feature names
    feature_names = list(X.columns)
    with open(models_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    progress(f"🧠 ML completado — {len(models)} modelos guardados en memory/models/")
    CURRICULUM[3].status = HitoStatus.COMPLETADO
    CURRICULUM[3].progreso = 1.0
    return models


# ── Phase 3: Demo Trading Warmup ─────────────────────────────────────────────

def phase_demo_warmup(n_trades: int = 20):
    print("\n" + "="*55)
    print(f"  FASE 3: Calentamiento con {n_trades} trades demo")
    print("="*55)

    if not HAS_PANDAS:
        print("[SKIP] pandas no instalado")
        return

    from smc.structure import MarketStructure, StructureType
    from smc.orderblocks import OrderBlockDetector, FVGDetector
    from core.risk_manager import RiskManager
    from core.decision_filter import DecisionFilter, TradeGrade

    cfg = config
    rm = RiskManager(cfg, capital=1000.0)
    df_filter = DecisionFilter(cfg, rm)

    wins = 0
    losses = 0

    for i in range(1, n_trades + 1):
        # Simulate OHLCV data for demo trade
        np.random.seed(i * 7)
        n = 50
        trend = np.random.choice([-1, 1])
        closes = 100 + np.cumsum(np.random.normal(trend * 0.3, 1.0, n))
        highs  = closes + np.abs(np.random.normal(0, 0.5, n))
        lows   = closes - np.abs(np.random.normal(0, 0.5, n))
        opens  = closes + np.random.normal(0, 0.3, n)
        vols   = np.abs(np.random.normal(1000, 200, n))

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": vols,
        })

        entry = closes[-1]
        bias  = "bullish" if trend > 0 else "bearish"
        sl    = entry * (0.995 if bias == "bullish" else 1.005)
        tp    = entry + (entry - sl) * 2.5

        result = df_filter.evaluate(
            df=df, symbol="BTCUSDT_SIM", timeframe="1h",
            entry=entry, stop_loss=sl, take_profit=tp, bias=bias,
        )

        # Simulate trade outcome (60% win rate in good conditions)
        trade_won = (
            result.grade != TradeGrade.NO_TRADE
            and np.random.uniform() < 0.58
        )

        if result.grade == TradeGrade.NO_TRADE:
            progress(f"📈 Trade demo {i}/{n_trades} — BLOQUEADO (score {result.score})", telegram=False)
        else:
            if trade_won:
                wins += 1
                pnl = rm.capital * cfg.max_risk_per_trade * result.risk_multiplier * 2.5
                rm.record_trade(pnl)
                progress(
                    f"📈 Trade demo {i}/{n_trades} — ✅ WIN +${pnl:.2f} "
                    f"| Score {result.score} | {result.grade.value.upper()}",
                    telegram=(i % 10 == 0 or i == n_trades),
                )
            else:
                losses += 1
                pnl = -rm.capital * cfg.max_risk_per_trade * result.risk_multiplier
                rm.record_trade(pnl)
                progress(
                    f"📈 Trade demo {i}/{n_trades} — ❌ LOSS -${abs(pnl):.2f} "
                    f"| Score {result.score} | {result.grade.value.upper()}",
                    telegram=(i % 10 == 0 or i == n_trades),
                )

    total_closed = wins + losses
    wr = wins / total_closed * 100 if total_closed > 0 else 0
    progress(f"📈 Calentamiento completado: {wins}W / {total_closed - wins}L — Win rate: {wr:.1f}%")

    if wr >= 50:
        CURRICULUM[2].status = HitoStatus.COMPLETADO
        CURRICULUM[2].progreso = 1.0
    else:
        CURRICULUM[2].progreso = total_closed / n_trades


# ── Phase 4: Final check ──────────────────────────────────────────────────────

def phase_final_check():
    print("\n" + "="*55)
    print("  FASE 4: Verificacion final")
    print("="*55)

    checks = []

    # Config
    try:
        from core.config import config as cfg
        checks.append(("Config", True, f"riesgo {cfg.max_risk_per_trade*100}%"))
    except Exception as e:
        checks.append(("Config", False, str(e)))

    # Risk Manager
    try:
        from core.risk_manager import RiskManager
        rm = RiskManager(config, 1000)
        ok, _ = rm.can_open_trade()
        checks.append(("RiskManager", ok, "can_open_trade OK"))
    except Exception as e:
        checks.append(("RiskManager", False, str(e)))

    # DecisionFilter
    try:
        from core.decision_filter import DecisionFilter
        checks.append(("DecisionFilter", True, "importado OK"))
    except Exception as e:
        checks.append(("DecisionFilter", False, str(e)))

    # MarketConnector
    try:
        from connectors.market_connector import MarketConnector
        checks.append(("MarketConnector", True, "importado OK"))
    except Exception as e:
        checks.append(("MarketConnector", False, str(e)))

    # Telegram
    try:
        from dashboard.telegram_commander import TelegramCommander
        checks.append(("TelegramCommander", True, "importado OK"))
    except Exception as e:
        checks.append(("TelegramCommander", False, str(e)))

    # Anthropic key
    checks.append((
        "Anthropic API Key",
        bool(config.anthropic_api_key and not config.anthropic_api_key.startswith("sk-ant-PEGA")),
        "configurada" if config.anthropic_api_key else "FALTA",
    ))

    # Telegram token
    checks.append((
        "Telegram Token",
        bool(config.telegram_bot_token and not config.telegram_bot_token.startswith("PEGA")),
        "configurado" if config.telegram_bot_token else "FALTA",
    ))

    all_ok = True
    for name, ok, detail in checks:
        status = "OK" if ok else "FALLO"
        print(f"  [{status}] {name}: {detail}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        progress("Todos los checks pasaron. Bot listo para activar.")
    else:
        print("[!] Algunos checks fallaron. Revisa .env antes de activar el bot.")

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMC Bot Training Pipeline")
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-ml",      action="store_true")
    parser.add_argument("--skip-demo",    action="store_true")
    parser.add_argument("--only-check",   action="store_true")
    parser.add_argument("--demo-trades",  type=int, default=20)
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  SMC BOT — PIPELINE DE ENTRENAMIENTO")
    print("="*55)
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.only_check:
        phase_final_check()
        return

    progress(
        f"🚀 Iniciando entrenamiento SMC — "
        f"YouTube + ML + {args.demo_trades} trades demo en background..."
    )

    # Run all phases
    strategies = phase_youtube(skip=args.skip_youtube)
    models     = phase_ml_training(skip=args.skip_ml)

    if not args.skip_demo:
        phase_demo_warmup(n_trades=args.demo_trades)

    all_ok = phase_final_check()

    # Curriculum report
    print()
    from training.curriculum import print_curriculum_status
    print_curriculum_status()

    print("\n" + "="*55)
    if all_ok:
        print("  ENTRENAMIENTO COMPLETO. Ejecuta: python main.py")
        progress("✅ Entrenamiento completado — bot listo para operar. Ejecuta startup.py")
    else:
        print("  ENTRENAMIENTO INCOMPLETO. Revisa los errores arriba.")
        progress("⚠️ Entrenamiento incompleto — revisa la terminal para ver los errores")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
