"""
Screening backtest for CANDIDATE pairs not currently in MT5_SYMBOLS.
===============================================================
Same engine/rules as scripts/backtest_multiyear.py (thr=80/90, RR=3.0,
partial+BE at 1.0R, kill zone 13-20 UTC) applied to pairs the live bot
has never traded, to see if any show genuine 2-year historical edge
worth adding (same bar NZDUSD/GBPUSD had to clear before being added
on 2026-07-01).

Non-JPY crosses only (keeps pip math == 0.0001/$10 per lot, consistent
with the approximation already used for USDCAD elsewhere in this repo).
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import yfinance as yf

RISK_PCT = 0.005
MAX_RISK = 275.0
RR = 3.0

CANDIDATES = {
    "USDCHF": "USDCHF=X",
    "EURGBP": "EURGBP=X",
    "AUDCAD": "AUDCAD=X",
    "EURCAD": "EURCAD=X",
    "GBPCAD": "GBPCAD=X",
    "EURAUD": "EURAUD=X",
}
PIP_SZ  = {p: 0.0001 for p in CANDIDATES}
PIP_VAL = {p: 10.0 for p in CANDIDATES}

END = datetime.now()
START_H1 = END - timedelta(days=700)
START_D1 = END - timedelta(days=3650)

print("=" * 72)
print("  SCREENING DE PARES CANDIDATOS (no en MT5_SYMBOLS)")
print("  Mismas reglas que el bot en vivo: thr=80/90, RR=3.0, partial+BE@1R")
print("=" * 72)

h1_data, d1_data = {}, {}
for pair, tk in CANDIDATES.items():
    try:
        dh1 = yf.download(tk, start=START_H1, end=END, interval="1h", progress=False, auto_adjust=True)
        if isinstance(dh1.columns, pd.MultiIndex):
            dh1.columns = dh1.columns.get_level_values(0)
        dh1.columns = [c.lower() for c in dh1.columns]
        dh1.dropna(inplace=True)
        h1_data[pair] = dh1

        dd1 = yf.download(tk, start=START_D1, end=END, interval="1d", progress=False, auto_adjust=True)
        if isinstance(dd1.columns, pd.MultiIndex):
            dd1.columns = dd1.columns.get_level_values(0)
        dd1.columns = [c.lower() for c in dd1.columns]
        dd1.dropna(inplace=True)
        d1_data[pair] = dd1
        print(f"  {pair}: H1={len(dh1)} bars | D1={len(dd1)} bars")
    except Exception as e:
        print(f"  {pair}: ERROR {e}")


def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr14(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def smc_signal(df, idx, bias):
    if idx < 60: return "WAIT", 0, 0.0
    w = df.iloc[max(0,idx-80):idx+1]
    if len(w) < 40: return "WAIT", 0, 0.0
    atr_v = atr14(w).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0: return "WAIT", 0, 0.0
    c, h, l = w["close"], w["high"], w["low"]
    e8, e21, e50 = ema(c,8).iloc[-1], ema(c,21).iloc[-1], ema(c,50).iloc[-1]
    cur = c.iloc[-1]
    prev_h = h.iloc[-20:-5].max()
    prev_l = l.iloc[-20:-5].min()
    bos_bull = cur > prev_h and c.iloc[-2] <= prev_h
    bos_bear = cur < prev_l and c.iloc[-2] >= prev_l
    body = (c - w["open"]).abs().iloc[-15:]
    ob_strong = body.max() > atr_v * 0.8
    score = 0; sig = "WAIT"
    if bias == "LONG":
        if bos_bull: score += 35
        if e8 > e21:  score += 15
        if e21 > e50: score += 12
        if cur > e21: score += 10
        if ob_strong: score += 12
        if cur > c.iloc[-4]: score += 16
        sig = "LONG" if score >= 50 else "WAIT"
    elif bias == "SHORT":
        if bos_bear: score += 35
        if e8 < e21:  score += 15
        if e21 < e50: score += 12
        if cur < e21: score += 10
        if ob_strong: score += 12
        if cur < c.iloc[-4]: score += 16
        sig = "SHORT" if score >= 50 else "WAIT"
    return sig, min(100, int(score * 1.25)), float(atr_v)

def d1_trend(dfd, dt):
    s = dfd[dfd.index.date <= pd.Timestamp(dt).date()]
    if len(s) < 50: return "UNKNOWN"
    c = s["close"]
    return "LONG" if c.iloc[-1] > ema(c,50).iloc[-1] else "SHORT"

def h4_bias(dh1, dt):
    s = dh1[dh1.index <= pd.Timestamp(dt)].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    if len(s) < 20: return "WAIT"
    c = s["close"]
    e8, e20 = ema(c,8).iloc[-1], ema(c,20).iloc[-1]
    return "LONG" if e8 > e20 else ("SHORT" if e8 < e20 else "WAIT")

def risk_for_score(score):
    if score >= 90: return min(MAX_RISK * 1.5, 400.0), 0.01
    if score >= 80: return MAX_RISK, 0.005
    return MAX_RISK * 0.7, 0.0025


pair_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

for pair, df1 in h1_data.items():
    dfd = d1_data.get(pair, pd.DataFrame())
    open_pos = []
    for idx in range(80, len(df1)):
        bar = df1.iloc[idx]
        dt = df1.index[idx]
        if pd.Timestamp(dt).weekday() >= 5: continue
        hour_utc = pd.Timestamp(dt).hour
        if hour_utc < 13 or hour_utc >= 20: continue

        new_open = []
        for pos in open_pos:
            (eidx, direction, entry, sl, tp, vol_p, sl_dist, partial_done, be_sl, pip_v, pair_p) = pos
            pnl = None
            cur_h, cur_l = bar["high"], bar["low"]
            if not partial_done:
                one_r_level = (entry + sl_dist) if direction=="LONG" else (entry - sl_dist)
                if direction == "LONG" and cur_l <= sl:
                    pnl = -vol_p * sl_dist * pip_v / PIP_SZ[pair_p]
                elif direction == "SHORT" and cur_h >= sl:
                    pnl = -vol_p * sl_dist * pip_v / PIP_SZ[pair_p]
                elif direction == "LONG" and cur_h >= one_r_level:
                    partial_pnl = (vol_p * 0.5) * sl_dist * pip_v / PIP_SZ[pair_p]
                    pair_stats[pair_p]["trades"] += 1
                    pair_stats[pair_p]["wins"] += 1
                    pair_stats[pair_p]["pnl"] += partial_pnl
                    new_be = entry
                    new_open.append((eidx, direction, entry, new_be, tp, vol_p*0.5, sl_dist, True, new_be, pip_v, pair_p))
                    continue
                elif direction == "SHORT" and cur_l <= one_r_level:
                    partial_pnl = (vol_p * 0.5) * sl_dist * pip_v / PIP_SZ[pair_p]
                    pair_stats[pair_p]["trades"] += 1
                    pair_stats[pair_p]["wins"] += 1
                    pair_stats[pair_p]["pnl"] += partial_pnl
                    new_be = entry
                    new_open.append((eidx, direction, entry, new_be, tp, vol_p*0.5, sl_dist, True, new_be, pip_v, pair_p))
                    continue
            else:
                if direction == "LONG":
                    if cur_h >= tp: pnl = vol_p * sl_dist * RR * pip_v / PIP_SZ[pair_p]
                    elif cur_l <= be_sl: pnl = 0.0
                else:
                    if cur_l <= tp: pnl = vol_p * sl_dist * RR * pip_v / PIP_SZ[pair_p]
                    elif cur_h >= be_sl: pnl = 0.0

            if pnl is not None:
                if pnl != 0.0:
                    pair_stats[pair_p]["trades"] += 1
                    pair_stats[pair_p]["wins"] += int(pnl > 0)
                    pair_stats[pair_p]["pnl"] += pnl
            else:
                new_open.append(pos)

        open_pos = new_open
        if len(open_pos) >= 2: continue

        d_dir = d1_trend(dfd, dt)
        if d_dir == "UNKNOWN": continue
        h4_d = h4_bias(df1, dt)
        if h4_d not in (d_dir, "WAIT"): continue

        sig, score, atr_v = smc_signal(df1, idx, d_dir)
        if sig == "WAIT": continue

        thr = 80 if h4_d != "WAIT" else 90
        if score < thr: continue

        max_r, r_pct = risk_for_score(score)
        sl_dist_p = atr_v * 1.5
        pip_v = PIP_VAL[pair]
        pip_s = PIP_SZ[pair]
        sl_pips = sl_dist_p / pip_s
        if sl_pips <= 0: continue
        vol = min(2.0, (97_000.0 * r_pct) / (sl_pips * pip_v))
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
        open_pos.append((idx, sig, entry, sl_p, tp_p, vol, sl_dist_p, False, entry, pip_v, pair))

print("\n" + "=" * 72)
print("  RESULTADOS POR PAR CANDIDATO (2 años, mismas reglas del bot)")
print("=" * 72)
print(f"  {'Par':8s} | {'Trades':7s} | {'WR':6s} | {'Total P&L':11s} | {'Avg P&L':8s} | Veredicto")
print("  " + "-"*70)
for p in sorted(pair_stats.keys(), key=lambda x: pair_stats[x]["pnl"], reverse=True):
    st = pair_stats[p]
    if st["trades"] < 5:
        print(f"  {p:8s} | {st['trades']:7d} | {'--':6s} | {'--':11s} | {'--':8s} | pocos trades, sin conclusion")
        continue
    wr = st["wins"] / st["trades"] * 100
    avg = st["pnl"] / st["trades"]
    verdict = "CANDIDATO A AGREGAR" if st["pnl"] > 0 and avg > -20 else "RECHAZAR"
    print(f"  {p:8s} | {st['trades']:7d} | {wr:5.0f}% | ${st['pnl']:9.0f}   | ${avg:6.0f}   | {verdict}")

print("\nNota: pip_value aproximado a $10/pip para todos (misma convencion")
print("ya usada para USDCAD en scripts/backtest_multiyear.py) -- es un cribado")
print("preliminar, no el modelo de ejecucion real.")
