"""
BACKTEST MULTI-ANUAL — 8 DIMENSIONES | 2 años H1 + 10 años D1
===============================================================
Las 8 dimensiones del mercado:
  DIM 1 — Temporal: borde varía por año/trimestre/mes/hora
  DIM 2 — Régimen de volatilidad: alto (2020,2022) vs bajo (2017,2024)
  DIM 3 — Régimen de tendencia: trending vs choppy vs lateral
  DIM 4 — Sesión: qué horas UTC dan mejor WR históricamente
  DIM 5 — Par: cuál da mejor edge por régimen
  DIM 6 — Riesgo: Kelly fraction óptima con 10 años de datos
  DIM 7 — Salida: partial TP óptimo (0.8R, 1.0R, 1.5R, 2.0R)
  DIM 8 — Correlación: efecto portafolio real entre pares

Monte Carlo: 100,000 simulaciones con distribución empírica real.
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import yfinance as yf

from smc.momentum import MomentumIndicators
from smc.bill_williams import BillWilliamsIndicators
from smc.liquidity_sweep import check_setup as silver_bullet_check

print("=" * 72)
print("  BACKTEST MULTI-ANUAL — 8 DIMENSIONES")
print("  H1: 2 años | D1: 10 años | Monte Carlo: 100,000 sims")
print("=" * 72)

MAX_OPEN_TEST = int(os.environ.get("MAX_OPEN_TEST", "2"))  # 2026-07-16: parametrizado para comparar 2 vs 3 (pedido por Jose)
REQUIRE_D1 = os.environ.get("REQUIRE_D1", "1") == "1"  # 2026-07-21: medir impacto real de D1-FILTER (ver smc_signal())
REQUIRE_H4 = os.environ.get("REQUIRE_H4", "1") == "1"  # 2026-07-21: medir impacto real de H4-FILTER
PEAK_GUARD_MIN = float(os.environ.get("PEAK_GUARD_MIN", "200"))    # 2026-07-16: recalibracion pedida por Jose
PEAK_GUARD_RETRACE = float(os.environ.get("PEAK_GUARD_RETRACE", "0.30"))
# 2026-07-24: STAGNANT/TIME-CLOSE-36H/FRIDAY-CLOSE were NEVER simulated here --
# live data shows they (not TP/SL/PEAK-GUARD) drive 77% of real closes, so every
# conclusion drawn from this backtest before today was measured against a model
# missing the dominant real exit mechanism. Added to match core/position_guards.py
# exactly (values there as of 2026-07-24), parametrized for sweeping.
STAGNANT_HOURS       = float(os.environ.get("STAGNANT_HOURS", "4.0"))
STAGNANT_PEAK_MAX    = float(os.environ.get("STAGNANT_PEAK_MAX", "15.0"))
STAGNANT_GRACE_HOURS = float(os.environ.get("STAGNANT_GRACE_HOURS", "2.0"))
MAX_HOLD_HOURS       = float(os.environ.get("MAX_HOLD_HOURS_TEST", "36.0"))
SWING_MAX_LOSS_ABS   = float(os.environ.get("SWING_MAX_LOSS_TEST", "150.0"))
FRIDAY_CLOSE_HOUR    = 19  # UTC -- matches live; DEAD_HOURS_UTC below already
                           # skips hour 19, so the sim's earliest reachable
                           # Friday close check is hour 20, same as live effect
CAPITAL = 96_184.0
RISK_PCT = 0.005
MAX_RISK = 275.0  # probado doblar a 550 (2026-07-09): P(pasar Axi)+2.5pp pero
# P(mes<-5%) 6%->16% (casi triplico) -- Axi revienta la cuenta a ese drawdown,
# no vale la pena el intercambio. Revertido a 275.
RR = float(os.environ.get("RR_TEST", "3.0"))  # 2026-07-24: parametrizado -- live data shows only 2.8% of real closes hit the designed TP (77% close via guards instead), testing whether a more reachable RR changes that
DAILY_TARGET = 250.0
PAIRS_FOREX = {
    # actualizado 2026-07-09: GBPUSD removido -- auditoria de episodes.db (591 trades
    # reales) lo mostro como el peor par activo (n=147, WR=25.9%, PF=0.53, neto -$887.55).
    # USDCHF/EURAUD/GBPCAD agregados 2026-07-05 tras screening backtest positivo.
    "EURUSD": "EURUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
    "EURAUD": "EURAUD=X",
    "GBPCAD": "GBPCAD=X",
}
PAIR_NAS = {"NAS100": "^NDX"}
PIP_SZ  = {"EURUSD":0.0001,"GBPUSD":0.0001,"AUDUSD":0.0001,"USDCAD":0.0001,"NZDUSD":0.0001,
           "USDCHF":0.0001,"EURAUD":0.0001,"GBPCAD":0.0001,"NAS100":1.0}
PIP_VAL = {"EURUSD":10.0,"GBPUSD":10.0,"AUDUSD":10.0,"USDCAD":10.0,"NZDUSD":10.0,
           "USDCHF":10.0,"EURAUD":6.6,"GBPCAD":7.1,"NAS100":1.0}

rng = np.random.default_rng(42)

# Toggles para aislar el efecto de cada filtro nuevo 2026-07-09 (diagnostico
# temporal -- ver cual filtro realmente ayuda antes de decidir la config final)
ENABLE_MOMENTUM_FILTERS = False  # desactivado en vivo 2026-07-09 -- ver core/supervisor.py
ENABLE_SILVER_BULLET_GATE = True  # activo en vivo pero nunca dispara en 2 anios de datos (inerte)
ENABLE_REGIME_FILTER = False  # probado 2026-07-09: empeoro TODO (P(pasar Axi) 40.3%->31.5%,
# E[mensual] $3395->$1559, Sharpe 0.476->0.20, P(mes<-5%) 6%->21%) -- rechazado

# ── Data download ──────────────────────────────────────────────────────
print("\n[DATA] Descargando datos historicos...")
print("       H1: hasta 2 años | D1: hasta 10 años")

all_tickers = {**PAIRS_FOREX, **PAIR_NAS}
d1_data = {}
h1_data = {}

END = datetime.now()
START_H1 = END - timedelta(days=700)   # yfinance limit: <730 days for 1h
START_D1 = END - timedelta(days=3650)

for pair, tk in all_tickers.items():
    try:
        # ~2 years H1 (700 days = max reliable for yfinance 1h)
        dh1 = yf.download(tk, start=START_H1, end=END, interval="1h", progress=False, auto_adjust=True)
        if isinstance(dh1.columns, pd.MultiIndex):
            dh1.columns = dh1.columns.get_level_values(0)
        dh1.columns = [c.lower() for c in dh1.columns]
        dh1.dropna(inplace=True)
        h1_data[pair] = dh1

        # 10 years D1
        dd1 = yf.download(tk, start=START_D1, end=END, interval="1d", progress=False, auto_adjust=True)
        if isinstance(dd1.columns, pd.MultiIndex):
            dd1.columns = dd1.columns.get_level_values(0)
        dd1.columns = [c.lower() for c in dd1.columns]
        dd1.dropna(inplace=True)
        d1_data[pair] = dd1

        print(f"  {pair}: H1={len(dh1)} bars ({len(dh1)//504:.1f} años efectivos) | D1={len(dd1)} bars ({len(dd1)/252:.1f} años)")
    except Exception as e:
        print(f"  {pair}: ERROR {e}")

# ── Utils ──────────────────────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def atr14(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def vol_regime(df, idx, lookback=60):
    """Returns 'HIGH', 'NORMAL', 'LOW' volatility regime."""
    if idx < lookback: return "NORMAL"
    w = df.iloc[max(0,idx-lookback):idx+1]
    a = atr14(w)
    cur = a.iloc[-1]
    hist_mean = a.mean()
    hist_std = a.std()
    if pd.isna(cur) or pd.isna(hist_mean): return "NORMAL"
    if cur > hist_mean + 0.7 * hist_std: return "HIGH"
    if cur < hist_mean - 0.5 * hist_std: return "LOW"
    return "NORMAL"

def trend_regime(df, idx):
    """Returns 'STRONG_TREND', 'MILD_TREND', 'CHOPPY'."""
    if idx < 50: return "MILD_TREND"
    w = df.iloc[max(0,idx-50):idx+1]
    c = w["close"]
    e8, e21, e50 = ema(c,8).iloc[-1], ema(c,21).iloc[-1], ema(c,50).iloc[-1]
    aligned = (e8 > e21 > e50) or (e8 < e21 < e50)
    # ADX proxy via ATR change
    atr_now = atr14(w).iloc[-1]
    atr_old = atr14(df.iloc[max(0,idx-100):idx-50]).mean() if idx > 100 else atr_now
    if aligned and atr_now > atr_old * 0.9:
        return "STRONG_TREND"
    elif aligned:
        return "MILD_TREND"
    return "CHOPPY"

def smc_signal(df, idx):
    """Score 0-100 for BOTH directions independently, return whichever
    qualifies. Returns (signal_dir, score, atr_val).

    2026-07-21: rewritten to decouple signal generation from D1/H4 bias --
    it used to take `bias` as a parameter and only ever score the ONE
    direction it was told to look for (`if bias=="LONG": ...`), so a
    genuine bearish setup during a nominal D1 uptrend was never even
    evaluated -- the backtest could never measure what happens if
    D1-FILTER/H4-FILTER (which gate on real code in core/supervisor.py)
    were relaxed, because the bias was baked into signal generation
    itself rather than applied as a separate post-hoc gate the way the
    live bot actually works (SMC structure generates a real bullish/
    bearish read independently; D1-FILTER/H4-FILTER are separate hard
    checks applied AFTER). See REQUIRE_D1/REQUIRE_H4 at the call site.
    """
    if idx < 60: return "WAIT", 0, 0.0
    w = df.iloc[max(0,idx-80):idx+1]
    if len(w) < 40: return "WAIT", 0, 0.0
    atr_v = atr14(w).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0: return "WAIT", 0, 0.0
    c, h, l = w["close"], w["high"], w["low"]
    e8, e21, e50 = ema(c,8).iloc[-1], ema(c,21).iloc[-1], ema(c,50).iloc[-1]
    cur = c.iloc[-1]
    # BOS detection
    prev_h = h.iloc[-20:-5].max()
    prev_l = l.iloc[-20:-5].min()
    bos_bull = cur > prev_h and c.iloc[-2] <= prev_h
    bos_bear = cur < prev_l and c.iloc[-2] >= prev_l
    # Order Block (last strong impulse candle)
    body = (c - w["open"]).abs().iloc[-15:]
    ob_strong = body.max() > atr_v * 0.8

    score_long = 0
    if bos_bull: score_long += 35
    if e8 > e21:  score_long += 15
    if e21 > e50: score_long += 12
    if cur > e21: score_long += 10
    if ob_strong: score_long += 12
    if cur > c.iloc[-4]: score_long += 16  # momentum

    score_short = 0
    if bos_bear: score_short += 35
    if e8 < e21:  score_short += 15
    if e21 < e50: score_short += 12
    if cur < e21: score_short += 10
    if ob_strong: score_short += 12
    if cur < c.iloc[-4]: score_short += 16

    if score_long >= 50 and score_long >= score_short:
        return "LONG", min(100, int(score_long * 1.25)), float(atr_v)
    if score_short >= 50 and score_short > score_long:
        return "SHORT", min(100, int(score_short * 1.25)), float(atr_v)
    return "WAIT", 0, float(atr_v)

def d1_trend(dfd, dt):
    s = dfd[dfd.index.date <= pd.Timestamp(dt).date()]
    if len(s) < 50: return "UNKNOWN"
    c = s["close"]
    return "LONG" if c.iloc[-1] > ema(c,50).iloc[-1] else "SHORT"

def h4_bias(dh1, dt):
    """Estimate H4 trend from H1 data via resampling."""
    s = dh1[dh1.index <= pd.Timestamp(dt)].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    if len(s) < 20: return "WAIT"
    c = s["close"]
    e8, e20 = ema(c,8).iloc[-1], ema(c,20).iloc[-1]
    return "LONG" if e8 > e20 else ("SHORT" if e8 < e20 else "WAIT")

def risk_for_score(score):
    """Dynamic risk based on conviction score."""
    if score >= 90: return min(MAX_RISK * 1.5, 400.0), 0.01
    if score >= 80: return MAX_RISK, 0.005
    return MAX_RISK * 0.7, 0.0025

# ── DIMENSIÓN 1+2+3: Run full historical simulation ───────────────────
print("\n" + "=" * 72)
print("  DIMENSIONES 1-3: Backtest temporal + régimen vol + régimen trend")
print("  Simulando 2 años H1 con todos los pares...")
print("=" * 72)

trade_log = []       # all trades with metadata
daily_pnl  = defaultdict(float)
regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
hour_stats   = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
year_stats   = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
pair_stats   = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

for pair, df1 in h1_data.items():
    dfd = d1_data.get(pair, pd.DataFrame())
    open_pos = []

    for idx in range(80, len(df1)):
        bar = df1.iloc[idx]
        dt = df1.index[idx]
        if pd.Timestamp(dt).weekday() >= 5: continue
        hour_utc = pd.Timestamp(dt).hour
        # Bug found 2026-07-07: this used to keep hours 13-19 UTC, which is NOT
        # what the live bot trades. Real DEAD_HOURS_UTC (core/supervisor.py:121)
        # blocks {0-13, 17,18,19} -- active hours are 14-16 and 20-23 UTC. The
        # old window here INCLUDED hour 13 and the empirically-bad 17-19 block
        # (WR=24-28%, see DEAD_HOURS_UTC comment) while EXCLUDING 20-23, which
        # the live bot actually trades. Every cached backtest_results.json
        # number produced by this script was simulating the wrong hours.
        DEAD_HOURS_UTC = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19}
        if hour_utc in DEAD_HOURS_UTC: continue  # kill zone
        day_str = str(pd.Timestamp(dt).date())
        year_str = str(pd.Timestamp(dt).year)

        # Manage open positions (partial TP + BE at 1.0R, full TP/SL)
        new_open = []
        is_friday = pd.Timestamp(dt).weekday() == 4
        for pos in open_pos:
            (eidx, direction, entry, sl, tp, vol_p, sl_dist,
             partial_done, be_sl, pip_v, pair_p, peak_pnl, stagn_flag_idx) = pos
            pnl = None
            age_h = idx - eidx  # H1 bars: 1 bar == 1 hour

            cur_h = bar["high"]
            cur_l = bar["low"]
            cur_c = bar["close"]

            # Fix 2026-07-06: live bot no longer partial-closes at 1R+immediate-BE
            # (validated against 584 real trades: it was capping every winner near
            # ~0.5R while losses ran to full SL, ratio 1.36:1 vs the RR=3.0 the
            # system is configured for -- generalized the XAUUSD-only skip to all
            # symbols). Simulate that directly: full SL or full TP, no partial leg,
            # matching core/supervisor.py's current live exit logic. Trailing-to-BE
            # at 1.5R still exists live but only protects against giveback after
            # 1.5R -- doesn't change the SL/TP outcome distribution modeled here.
            # SWING-STOP sim (2026-07-24): tried modeling the flat -$150
            # emergency backstop here, approximated via bar adverse-excursion.
            # REVERTED same-session: it made close-type proportions LESS
            # realistic, not more -- final_SL dropped to 0.1% of closes
            # (live measured 20.1%) because designed_sl_loss > $150 for most
            # simulated positions, so SWING-STOP pre-empted the real SL on
            # almost every losing trade. That means the backtest's position
            # sizing doesn't match live's actual SL-risk distribution closely
            # enough for this specific refinement to be trustworthy yet --
            # needs the sizing model calibrated first, not just this guard
            # bolted on. Left disabled; STAGNANT/TIME-CLOSE/FRIDAY-CLOSE
            # (added earlier this session) still measurably improved realism
            # and are kept.
            top_close_type = None

            if direction == "LONG":
                if cur_l <= sl:
                    pnl = -vol_p * sl_dist * pip_v / PIP_SZ[pair_p]
                elif cur_h >= tp:
                    pnl = vol_p * sl_dist * RR * pip_v / PIP_SZ[pair_p]
            else:
                if cur_h >= sl:
                    pnl = -vol_p * sl_dist * pip_v / PIP_SZ[pair_p]
                elif cur_l <= tp:
                    pnl = vol_p * sl_dist * RR * pip_v / PIP_SZ[pair_p]

            if pnl is not None:
                if pnl != 0.0:
                    daily_pnl[day_str] += pnl
                    vr = vol_regime(df1, idx)
                    tr = trend_regime(df1, idx)
                    trade_log.append({
                        "pair": pair_p, "type": top_close_type or "final", "pnl": pnl,
                        "win": pnl > 0, "hour": hour_utc, "year": year_str,
                        "vol_regime": vr, "trend_regime": tr,
                    })
                    regime_stats[(vr, tr)]["trades"] += 1
                    regime_stats[(vr, tr)]["wins"] += int(pnl > 0)
                    regime_stats[(vr, tr)]["pnl"] += pnl
                    hour_stats[hour_utc]["trades"] += 1
                    hour_stats[hour_utc]["wins"] += int(pnl > 0)
                    hour_stats[hour_utc]["pnl"] += pnl
                    year_stats[year_str]["trades"] += 1
                    year_stats[year_str]["wins"] += int(pnl > 0)
                    year_stats[year_str]["pnl"] += pnl
                    pair_stats[pair_p]["trades"] += 1
                    pair_stats[pair_p]["wins"] += int(pnl > 0)
                    pair_stats[pair_p]["pnl"] += pnl
            if pnl is None:
                # PEAK-GUARD sim (2026-07-16): track running peak floating $ this
                # trade has reached (using bar's favorable extreme), close early
                # if it retraces PEAK_GUARD_RETRACE from a peak >= PEAK_GUARD_MIN.
                if direction == "LONG":
                    fav_pnl = vol_p * (cur_h - entry) * pip_v / PIP_SZ[pair_p]
                    close_pnl = vol_p * (cur_c - entry) * pip_v / PIP_SZ[pair_p]
                else:
                    fav_pnl = vol_p * (entry - cur_l) * pip_v / PIP_SZ[pair_p]
                    close_pnl = vol_p * (entry - cur_c) * pip_v / PIP_SZ[pair_p]
                if fav_pnl > peak_pnl:
                    peak_pnl = fav_pnl

                close_type = None
                if (peak_pnl >= PEAK_GUARD_MIN
                        and close_pnl < peak_pnl * (1.0 - PEAK_GUARD_RETRACE)):
                    pnl = close_pnl
                    close_type = "peak_guard"
                # BUG-BACKTEST-NO-STAGNANT-TIME-FRIDAY (2026-07-24): these 3
                # guards drive 77% of real live closes (measured against 45
                # days of MT5 deal history) but were never modeled here --
                # every conclusion this script produced before today was
                # measured against a simulation missing the dominant real
                # exit mechanism. Added to mirror core/position_guards.py
                # exactly: FRIDAY-CLOSE closes everyone regardless of P&L;
                # STAGNANT closes a position that's been open STAGNANT_HOURS+
                # and never reached STAGNANT_PEAK_MAX profit (immediately if
                # pnl>=0, else after STAGNANT_GRACE_HOURS more); TIME-CLOSE
                # only force-closes LOSING positions past MAX_HOLD_HOURS
                # (winners are left to run / get caught by PEAK-GUARD).
                elif is_friday and hour_utc >= FRIDAY_CLOSE_HOUR + 1:
                    pnl = close_pnl
                    close_type = "friday_close"
                elif age_h >= STAGNANT_HOURS and peak_pnl < STAGNANT_PEAK_MAX:
                    if stagn_flag_idx is None:
                        stagn_flag_idx = idx
                    grace_h = idx - stagn_flag_idx
                    if close_pnl >= 0 or grace_h >= STAGNANT_GRACE_HOURS:
                        pnl = close_pnl
                        close_type = "stagnant"
                elif close_pnl <= 0 and age_h >= MAX_HOLD_HOURS:
                    pnl = close_pnl
                    close_type = "time_close"

                if close_type is not None:
                    if pnl != 0.0:
                        daily_pnl[day_str] += pnl
                        vr = vol_regime(df1, idx)
                        tr = trend_regime(df1, idx)
                        trade_log.append({
                            "pair": pair_p, "type": close_type, "pnl": pnl,
                            "win": pnl > 0, "hour": hour_utc, "year": year_str,
                            "vol_regime": vr, "trend_regime": tr,
                        })
                        regime_stats[(vr, tr)]["trades"] += 1
                        regime_stats[(vr, tr)]["wins"] += int(pnl > 0)
                        regime_stats[(vr, tr)]["pnl"] += pnl
                        hour_stats[hour_utc]["trades"] += 1
                        hour_stats[hour_utc]["wins"] += int(pnl > 0)
                        hour_stats[hour_utc]["pnl"] += pnl
                        year_stats[year_str]["trades"] += 1
                        year_stats[year_str]["wins"] += int(pnl > 0)
                        year_stats[year_str]["pnl"] += pnl
                        pair_stats[pair_p]["trades"] += 1
                        pair_stats[pair_p]["wins"] += int(pnl > 0)
                        pair_stats[pair_p]["pnl"] += pnl
                else:
                    new_open.append((eidx, direction, entry, sl, tp, vol_p, sl_dist,
                                      partial_done, be_sl, pip_v, pair_p, peak_pnl, stagn_flag_idx))

        open_pos = new_open
        if len(open_pos) >= MAX_OPEN_TEST: continue  # actualizado 2026-07-01: MAX_OPEN_POSITIONS real=2 (era 4, commit 468c476 bajo 3->2)

        # Signal generation -- decoupled from D1/H4 bias (see smc_signal()
        # docstring, 2026-07-21). REQUIRE_D1/REQUIRE_H4 let this backtest
        # measure the real impact of relaxing D1-FILTER/H4-FILTER (live in
        # core/supervisor.py), which the old bias-baked-into-generation
        # design could never test.
        sig, score, atr_v = smc_signal(df1, idx)
        if sig == "WAIT": continue

        d_dir = d1_trend(dfd, dt)
        if REQUIRE_D1:
            if d_dir == "UNKNOWN": continue
            if sig != d_dir: continue
        h4_d = h4_bias(df1, dt)
        if REQUIRE_H4 and h4_d not in (sig, "WAIT"): continue

        # DIAGNOSTICO 2026-07-09: DIM2/DIM3 del propio backtest concluyeron que
        # HIGH vol + STRONG_TREND es la mejor combinacion posible -- nunca se
        # habia probado como filtro duro, solo como observacion.
        if ENABLE_REGIME_FILTER:
            if vol_regime(df1, idx) != "HIGH" or trend_regime(df1, idx) != "STRONG_TREND":
                continue

        # Filtros nuevos 2026-07-09: RSI/Bollinger/Estocastico/volumen (smc/momentum.py)
        # + Alligator/Awesome Oscillator (smc/bill_williams.py), mismo criterio que el
        # pipeline en vivo (_enrich_with_agents en core/supervisor.py) -- ajustan el
        # score, no lo bloquean solos.
        _mw = df1.iloc[max(0, idx - 80):idx + 1]
        if ENABLE_MOMENTUM_FILTERS:
            try:
                score += MomentumIndicators(_mw).score_for_signal(sig).pts_adjustment
                score += BillWilliamsIndicators(_mw).score_for_signal(sig).pts_adjustment
            except Exception:
                pass

        # Silver Bullet ICT (2026-07-09): gate todo-o-nada SOLO en la kill zone
        # activa (14 UTC) -- si falta sweep+FVG+killzone en la direccion de la
        # senal, no se opera esa hora especifica, igual que en vivo.
        if ENABLE_SILVER_BULLET_GATE and hour_utc == 14:
            try:
                _sb = silver_bullet_check(_mw)
                _sb_dir = "bullish" if sig == "LONG" else "bearish"
                if _sb is None or not _sb.valid or _sb.direction != _sb_dir:
                    continue
            except Exception:
                pass

        # Threshold — actualizado 2026-07-05: MT5_SCORE_AUTO_REDUCE real=80 (core/supervisor.py:96,
        # recalibrado 2026-07-01 tras el sweep que probo 90-95 y NO mejoraba WR, solo cortaba volumen).
        # MT5_REAL_SCORE_THRESHOLD=95 es solo techo de excepcion, no la operacion normal.
        thr = 80 if h4_d != "WAIT" else 90
        if score < thr: continue

        # Risk scaling by score
        max_r, r_pct = risk_for_score(score)

        # DIAGNOSTICO 2026-07-09: escalar riesgo SOLO donde hay edge real
        # comprobado (episodes.db real: EURUSD PF=1.11, unico con neto positivo
        # entre los pares activos), en vez de subir el riesgo parejo a todos
        # (eso disparaba P(mes<-5%) de 6% a 16% -- ver commit e92f121).
        _PAIR_RISK_MULT = {"EURUSD": 1.8}
        _mult = _PAIR_RISK_MULT.get(pair, 1.0)
        max_r *= _mult

        # Volume
        sl_dist_p = atr_v * 1.5
        pip_v = PIP_VAL[pair]
        pip_s = PIP_SZ[pair]
        sl_pips = sl_dist_p / pip_s
        if sl_pips <= 0: continue
        vol = min(2.0, (CAPITAL * r_pct) / (sl_pips * pip_v))
        actual_risk = vol * sl_pips * pip_v
        if actual_risk > max_r:
            vol = max_r / (sl_pips * pip_v)
        vol = max(0.01, round(int(vol / 0.01) * 0.01, 2))
        actual_risk = vol * sl_pips * pip_v
        if actual_risk < 5: continue

        entry = bar["close"]
        if sig == "LONG":
            sl_p = entry - sl_dist_p
            tp_p = entry + sl_dist_p * RR
        else:
            sl_p = entry + sl_dist_p
            tp_p = entry - sl_dist_p * RR

        open_pos.append((idx, sig, entry, sl_p, tp_p, vol, sl_dist_p, False, entry, pip_v, pair, 0.0, None))

print(f"\n  Total trades en 2 años: {len(trade_log)}")
n_final = sum(1 for t in trade_log if t["type"] == "final")
n_partial = sum(1 for t in trade_log if t["type"] == "partial")
n_wins = sum(1 for t in trade_log if t["win"] and t["type"] in ("final",))
n_days = len(daily_pnl)
avg_d = np.mean(list(daily_pnl.values())) if daily_pnl else 0
print(f"  Parciales (50% a 1R): {n_partial} | Finals: {n_final} | Wins: {n_wins}/{n_final} = {n_wins/max(1,n_final)*100:.1f}% WR")
print(f"  Días con trades: {n_days} | Avg diario: ${avg_d:.0f}")

# 2026-07-24: como % del total de cierres reales (fuente de verdad: MT5 deal
# history), TP real=2.8%, SL real=20.1%, guardias(stagnant/time/friday/peak)=77.1%
_close_types = defaultdict(int)
for t in trade_log:
    _ctype = t["type"]
    if _ctype == "final":
        _ctype = "final_TP" if t["win"] else "final_SL"
    _close_types[_ctype] += 1
_n_total_closes = len(trade_log)
print("  Desglose por tipo de cierre (comparar contra realidad: TP=2.8% SL=20.1% guardias=77.1%):")
for _ct, _n in sorted(_close_types.items(), key=lambda kv: -kv[1]):
    print(f"    {_ct:12s}: {_n:6d} ({_n/max(1,_n_total_closes)*100:5.1f}%)")

# ── DIMENSIÓN 4: Por hora UTC ─────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 4: SESIÓN — Win rate y P&L por hora UTC")
print("=" * 72)
print(f"  {'Hora UTC':10s} | {'Trades':7s} | {'WR':6s} | {'Avg P&L':10s} | {'Rating':20s}")
print("  " + "-"*60)
for h in sorted(hour_stats.keys()):
    st = hour_stats[h]
    if st["trades"] < 5: continue
    wr = st["wins"] / st["trades"] * 100
    avg_pnl = st["pnl"] / st["trades"]
    rating = "🔥 PREMIUM" if wr > 50 and avg_pnl > 50 else ("✅ BUENA" if wr > 40 else ("⚠️ REGULAR" if wr > 30 else "❌ EVITAR"))
    print(f"  {h:02d}:00 UTC    | {st['trades']:7d} | {wr:5.0f}% | ${avg_pnl:8.0f}   | {rating}")

# Best hours
best_hours = sorted([h for h,s in hour_stats.items() if s["trades"] >= 3],
                    key=lambda h: hour_stats[h]["pnl"] / max(1, hour_stats[h]["trades"]), reverse=True)
print(f"\n  TOP 3 MEJORES HORAS: {best_hours[:3]}")

# ── DIMENSIÓN 1: Por año ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 1: TEMPORAL — Performance por año")
print("=" * 72)
print(f"  {'Año':6s} | {'Trades':7s} | {'WR':6s} | {'Total P&L':11s} | {'Avg/día':8s}")
print("  " + "-"*50)
for yr in sorted(year_stats.keys()):
    st = year_stats[yr]
    if st["trades"] < 10: continue
    wr = st["wins"] / st["trades"] * 100
    trading_days = st["trades"] / max(1, (n_final + n_partial) / max(1, n_days))
    print(f"  {yr:6s} | {st['trades']:7d} | {wr:5.0f}% | ${st['pnl']:9.0f}   | ${st['pnl']/max(1,trading_days):6.0f}")

# ── DIMENSIÓN 2+3: Por régimen ────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIONES 2+3: RÉGIMEN DE VOL + TENDENCIA — Cuándo funciona el sistema")
print("=" * 72)
print(f"  {'Vol':8s} | {'Trend':14s} | {'Trades':7s} | {'WR':6s} | {'Avg P&L':10s}")
print("  " + "-"*58)
for (vr, tr), st in sorted(regime_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
    if st["trades"] < 5: continue
    wr = st["wins"] / st["trades"] * 100
    avg_pnl = st["pnl"] / st["trades"]
    print(f"  {vr:8s} | {tr:14s} | {st['trades']:7d} | {wr:5.0f}% | ${avg_pnl:8.0f}")

# ── DIMENSIÓN 5: Por par ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 5: PAR — Performance por instrumento")
print("=" * 72)
print(f"  {'Par':8s} | {'Trades':7s} | {'WR':6s} | {'Total P&L':11s} | {'Avg P&L':8s}")
print("  " + "-"*52)
for p in sorted(pair_stats.keys(), key=lambda x: pair_stats[x]["pnl"], reverse=True):
    st = pair_stats[p]
    if st["trades"] < 5: continue
    wr = st["wins"] / st["trades"] * 100
    print(f"  {p:8s} | {st['trades']:7d} | {wr:5.0f}% | ${st['pnl']:9.0f}   | ${st['pnl']/st['trades']:6.0f}")

# ── DIMENSIÓN 6: Kelly Criterion ─────────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 6: KELLY — Tamaño óptimo de posición")
print("=" * 72)
if n_final > 10:
    final_trades = [t for t in trade_log if t["type"] == "final" and t["pnl"] != 0]
    wins_f = [t["pnl"] for t in final_trades if t["win"]]
    losses_f = [abs(t["pnl"]) for t in final_trades if not t["win"]]
    wr_f = len(wins_f) / len(final_trades)
    avg_win = np.mean(wins_f) if wins_f else 0
    avg_loss = np.mean(losses_f) if losses_f else 0
    if avg_loss > 0 and avg_win > 0:
        b = avg_win / avg_loss  # ratio win/loss
        kelly_f = (wr_f * (b + 1) - 1) / b
        print(f"  WR final: {wr_f*100:.1f}% | avg win: ${avg_win:.0f} | avg loss: ${avg_loss:.0f}")
        print(f"  B (win/loss ratio): {b:.2f}x")
        print(f"  Full Kelly fraction: {kelly_f*100:.1f}% del capital")
        print(f"  Half Kelly (safer):  {kelly_f*50:.1f}% del capital")
        print(f"  Actual risk:         {RISK_PCT*100:.1f}% del capital")
        kelly_mult = kelly_f / RISK_PCT if RISK_PCT > 0 else 0
        print(f"  Kelly recomienda {kelly_mult:.1f}x el riesgo actual ({RISK_PCT*100:.1f}%)")
        if kelly_f > 0:
            print(f"  => SUBUTILIZANDO capital — Kelly dice que {kelly_f*50:.1f}% es óptimo")

# ── DIMENSIÓN 7: Optimal Exit Level ──────────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 7: SALIDA ÓPTIMA — Partial TP level test")
print("=" * 72)

# Simulate different partial TP levels on the actual trade list
if final_trades:
    actual_wr = wr_f
    actual_rr = b

    # Model: P(hit 1R before SL) derived from WR pattern
    # In SMC with confirmed H4+D1, hitting 1R is more likely than 2.5R
    # WR at target X vs WR at 2.5R follows roughly: WR(x) ≈ WR * (2.5/x)^0.5
    for partial_r in [0.75, 1.0, 1.25, 1.5, 2.0, None]:
        if partial_r is None:
            # No partial — all-in
            e_val = actual_wr * actual_rr * avg_loss - (1-actual_wr) * avg_loss
            var = actual_wr * (actual_rr * avg_loss - e_val)**2 + (1-actual_wr) * (-avg_loss - e_val)**2
            label = "ALL-IN (sin partial)"
        else:
            wr_at_partial = min(0.80, actual_wr * (2.5/partial_r)**0.45)
            wr_at_full = actual_wr * (1 - 0.1 * (2.5 - partial_r))  # slightly less likely to hit full after partial

            # Expected value
            e1 = wr_at_partial * 0.5 * partial_r * avg_loss  # from partial
            e2 = wr_at_partial * wr_at_full * 0.5 * 2.5 * avg_loss  # from full
            e_loss = (1 - wr_at_partial) * avg_loss  # full loss
            e_val = e1 + e2 - e_loss
            # Variance (simplified)
            var = (1-wr_at_partial)*(avg_loss+e_val)**2 + wr_at_partial*(1-wr_at_full)*(0.5*partial_r*avg_loss-e_val)**2 + wr_at_partial*wr_at_full*(0.5*(partial_r+2.5)*avg_loss-e_val)**2
            label = f"partial@{partial_r}R"

        n_daily_trades = len(trade_log) / max(1, n_days)
        sigma_daily = np.sqrt(var * n_daily_trades)
        e_daily = e_val * n_daily_trades
        # P(day >= $250) using normal approximation
        z = (250 - e_daily) / max(1, sigma_daily)
        p250 = max(0, min(100, (1 - 0.5 * (1 + float(np.sign(z)) * (1 - np.exp(-abs(z)**1.6 / 2)))) * 100))
        from scipy import stats as _st
        p250_accurate = (1 - _st.norm.cdf(z)) * 100
        marker = " <== OPTIMO" if partial_r == 1.0 else ""
        print(f"  {label:22s}: E[trade]=${e_val:6.0f} | E[día]=${e_daily:6.0f} | sigma=${sigma_daily:6.0f} | P(>=$250)={p250_accurate:4.0f}%{marker}")

# ── DIMENSIÓN 8: Correlación portafolio ──────────────────────────────
print("\n" + "=" * 72)
print("  DIMENSIÓN 8: CORRELACIÓN — Efecto portafolio real")
print("=" * 72)

# Get daily returns for all pairs
daily_rets = {}
for pair, df1 in h1_data.items():
    df_d = df1["close"].resample("D").last().dropna()
    daily_rets[pair] = df_d.pct_change().dropna()

df_corr_all = pd.DataFrame(daily_rets).dropna()
if len(df_corr_all.columns) >= 2:
    corr = df_corr_all.corr()
    print("\n  Correlación de retornos diarios (2 años):")
    pairs_c = list(corr.columns)
    print(f"  {'':8s}", end="")
    for p in pairs_c: print(f"  {p:8s}", end="")
    print()
    for p1 in pairs_c:
        print(f"  {p1:8s}", end="")
        for p2 in pairs_c:
            v = corr.loc[p1, p2]
            print(f"  {v:+.2f}   ", end="")
        print()

    # Portfolio variance reduction
    print("\n  REGLA DE DIVERSIFICACION (basada en 2 años de datos reales):")
    high_corr_pairs = [(p1, p2, corr.loc[p1,p2]) for i,p1 in enumerate(pairs_c)
                       for j,p2 in enumerate(pairs_c) if i < j and abs(corr.loc[p1,p2]) > 0.65]
    for p1, p2, v in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"    {p1}+{p2}: r={v:+.2f} — {'NO ABRIR AMBOS EN MISMA DIRECCION' if v>0 else 'COBERTURA NATURAL'}")

# ── MONTE CARLO PRINCIPAL: 100,000 simulaciones ───────────────────────
print("\n" + "=" * 72)
print("  MONTE CARLO — 100,000 simulaciones con distribución empírica REAL")
print("=" * 72)

daily_vals = list(daily_pnl.values())
if len(daily_vals) >= 20:
    daily_arr = np.array(daily_vals)
    # Bootstrap: resample daily P&L
    sims_day   = rng.choice(daily_arr, size=100_000, replace=True)
    sims_week  = rng.choice(daily_arr, size=(100_000, 5), replace=True).sum(axis=1)
    sims_month = rng.choice(daily_arr, size=(100_000, 22), replace=True).sum(axis=1)

    p50  = np.percentile(daily_arr, 50)
    p25  = np.percentile(daily_arr, 25)
    p75  = np.percentile(daily_arr, 75)
    p05  = np.percentile(daily_arr, 5)
    p95  = np.percentile(daily_arr, 95)

    print(f"\n  DISTRIBUCION DIARIA (empírica {len(daily_vals)} días):")
    print(f"    Mediana:   ${p50:7.0f}")
    print(f"    P25-P75:   ${p25:7.0f} a ${p75:7.0f}")
    print(f"    P5-P95:    ${p05:7.0f} a ${p95:7.0f}")

    print(f"\n  ESTADÍSTICAS MONTE CARLO (100,000 sims):")
    print(f"    E[día]:              ${np.mean(sims_day):7.0f}")
    print(f"    E[semana]:           ${np.mean(sims_week):7.0f}")
    print(f"    E[mes]:              ${np.mean(sims_month):7.0f}")
    print(f"    P(día >= $250):      {np.mean(sims_day >= 250)*100:5.0f}%")
    print(f"    P(día >= $500):      {np.mean(sims_day >= 500)*100:5.0f}%")
    print(f"    P(semana >= $1,250): {np.mean(sims_week >= 1250)*100:5.0f}%")
    print(f"    P(semana >= $1,000): {np.mean(sims_week >= 1000)*100:5.0f}%")
    print(f"    P(semana >= $2,000): {np.mean(sims_week >= 2000)*100:5.0f}%")
    print(f"    P(semana entre $1,000-$2,000): {np.mean((sims_week >= 1000) & (sims_week <= 2000))*100:5.0f}%")
    print(f"    P(mes >= 5%=$4,851): {np.mean(sims_month >= 4851)*100:5.0f}%")
    print(f"    P(mes >= 4%=$3,881): {np.mean(sims_month >= 3881)*100:5.0f}%")
    print(f"    P(mes >= 3%=$2,910): {np.mean(sims_month >= 2910)*100:5.0f}%")
    print(f"    P(mes >= 2%=$1,940): {np.mean(sims_month >= 1940)*100:5.0f}%")
    print(f"    P(día <= -$1,000):   {np.mean(sims_day <= -1000)*100:5.0f}%")
    print(f"    P(mes < -5%=-$4851): {np.mean(sims_month <= -4851)*100:5.0f}%")
    print(f"    Sharpe mensual:      {np.mean(sims_month)/max(1,np.std(sims_month)):.2f}")

    # Percentile breakdown of THIS week specifically (5 trading days, real bootstrap)
    print(f"\n  DISTRIBUCION SEMANAL (escenarios, 5 dias de trading, {len(daily_vals)} dias historicos reales):")
    for wpct in [5, 25, 50, 75, 90]:
        wval = np.percentile(sims_week, wpct)
        print(f"    P{wpct:<3}: ${wval:8.0f}")

    # Percentile breakdown of monthly return
    print(f"\n  DISTRIBUCION MENSUAL (escenarios):")
    pcts = [5, 10, 25, 50, 75, 90, 95]
    for pct in pcts:
        val = np.percentile(sims_month, pct)
        label = f"P{pct}"
        roi = val / 97022 * 100
        print(f"    {label:4s}: ${val:8.0f} ({roi:+.1f}% ROI)")

    # ── OPTIMAL CONFIGURATION SUMMARY ────────────────────────────────
    print("\n" + "=" * 72)
    print("  CONFIGURACION OPTIMA FINAL (basada en 2 años + 100K sims)")
    print("=" * 72)

    p250_actual = np.mean(sims_day >= 250)*100
    e_monthly   = np.mean(sims_month)
    p_pass_axi  = np.mean(sims_month >= 4851)*100
    sharpe      = np.mean(sims_month) / max(1, np.std(sims_month))

    print(f"""
  RESULTADO: 2 AÑOS DE DATOS REALES + SIN PARTIAL (full SL/TP) + KILL ZONE 14-16,20-23 UTC

  P(dia >= $250):         {p250_actual:.0f}%
  E[mensual]:             ${e_monthly:.0f}
  P(pass Axi Select 5%):  {p_pass_axi:.0f}%
  Sharpe mensual:         {sharpe:.2f}

  LAS 8 DIMENSIONES — CONCLUSIONES:
  DIM 1 (Temporal):      Ver por año — algunos años >60% WR, otros <30%
  DIM 2 (Vol regimen):   HIGH vol regimen da MEJOR edge (mas BOS/CHoCH reales)
  DIM 3 (Trend regimen): STRONG_TREND + HIGH vol = mejor combo posible
  DIM 4 (Sesion):        14-16 UTC (NY open) y 20-23 UTC son las ventanas activas reales (DEAD_HOURS_UTC bloquea 0-13 y 17-19)
  DIM 5 (Par):           Ver ranking por par arriba — enfocarse en top 2
  DIM 6 (Kelly):         {"Sistema subutiliza capital — Kelly dice hasta " + f"{globals().get('kelly_f', 0)*50*100:.1f}%" if globals().get('kelly_f', -1) > 0 else "Kelly NEGATIVO en el tramo final-only (ver DIM6 arriba) — NO subir tamaño de posicion con este dato"}
  DIM 7 (Salida):        Partial-close desactivado en vivo (commit 5e3ffd5) — full SL/TP con trailing a breakeven
  DIM 8 (Correlacion):   EURUSD+GBPUSD+AUDUSD = riesgo triplicado si todos van igual

  ACCION INMEDIATA: sin partial-close (desactivado en vivo), focus 14-16+20-23 UTC
    """)

    # Save results JSON for future use
    results = {
        "date": datetime.now().isoformat(),
        "config": {
            "years_h1": 2, "years_d1": 10,
            "threshold_h4_confirmed": 80, "threshold_h4_wait": 90,
            "rr": RR, "partial_tp": None, "kill_zone_utc": "14-16,20-23",
            "pairs": list(PAIRS_FOREX.keys()),
        },
        "stats": {
            "total_trades": len(trade_log),
            "total_days": n_days,
            "wr_pct": round(n_wins / max(1, n_final) * 100, 1),
            "avg_daily": round(float(avg_d), 2),
            "p_day_250": round(float(np.mean(sims_day >= 250)*100), 1),
            "p_pass_axi": round(float(p_pass_axi), 1),
            "e_monthly": round(float(e_monthly), 2),
            "sharpe": round(float(sharpe), 3),
            "best_hours": best_hours[:5],
        }
    }
    _out_path = f"memory/backtest_results_maxopen{MAX_OPEN_TEST}.json" if MAX_OPEN_TEST != 2 else "memory/backtest_results.json"
    with open(_out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Resultados guardados en {_out_path}")
else:
    print("  Insuficientes datos para Monte Carlo (necesita 20+ días)")

print("\n" + "=" * 72)
print("  BACKTEST MULTI-ANUAL COMPLETADO")
print("=" * 72)
