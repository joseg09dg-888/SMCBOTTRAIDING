"""
BACKTEST VWAP — usa datos reales de MT5 (tick_volume real, a diferencia de
yfinance que devuelve volume=0 siempre para forex, haciendo VWAP imposible
de probar con esa fuente). Misma metodologia y parametros que
backtest_multiyear.py (RR=3.0, MAX_RISK=275, DEAD_HOURS_UTC, MAX_OPEN=2)
para que la comparacion A/B sea valida.

Prueba: filtro VWAP (ancla diaria UTC) — solo LONG si precio > VWAP del dia,
solo SHORT si precio < VWAP del dia — contra el baseline sin ese filtro.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collections import defaultdict
import MetaTrader5 as mt5

print("=" * 72)
print("  BACKTEST VWAP — datos reales MT5 (tick_volume)")
print("=" * 72)

CAPITAL = 96_184.0
RISK_PCT = 0.005
MAX_RISK = 275.0
RR = 3.0
PAIRS = ["EURUSD", "USDCAD", "NZDUSD", "USDCHF", "EURAUD", "GBPCAD"]
PIP_SZ  = {"EURUSD":0.0001,"USDCAD":0.0001,"NZDUSD":0.0001,"USDCHF":0.0001,"EURAUD":0.0001,"GBPCAD":0.0001}
PIP_VAL = {"EURUSD":10.0,"USDCAD":10.0,"NZDUSD":10.0,"USDCHF":10.0,"EURAUD":6.6,"GBPCAD":7.1}
DEAD_HOURS_UTC = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19}

rng = np.random.default_rng(42)

if not mt5.initialize():
    print("[ERROR] no se pudo conectar a MT5:", mt5.last_error())
    sys.exit(1)

print("\n[DATA] Descargando H1 (20000 barras) + D1 (2600 barras) desde MT5...")
h1_data, d1_data = {}, {}
for pair in PAIRS:
    r1 = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_H1, 0, 20000)
    rd = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_D1, 0, 2600)
    if r1 is None or rd is None:
        print(f"  {pair}: ERROR sin datos")
        continue
    df1 = pd.DataFrame(r1)
    df1["time"] = pd.to_datetime(df1["time"], unit="s")
    df1.set_index("time", inplace=True)
    df1.rename(columns={"tick_volume": "volume"}, inplace=True)
    h1_data[pair] = df1

    dfd = pd.DataFrame(rd)
    dfd["time"] = pd.to_datetime(dfd["time"], unit="s")
    dfd.set_index("time", inplace=True)
    dfd.rename(columns={"tick_volume": "volume"}, inplace=True)
    d1_data[pair] = dfd
    print(f"  {pair}: H1={len(df1)} bars ({df1.index[0].date()} a {df1.index[-1].date()}) | D1={len(dfd)}")

mt5.shutdown()

# ── Utils (idénticos a backtest_multiyear.py) ──────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

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

# ── VWAP anclado por dia UTC (usa tick_volume real de MT5) ─────────────
def add_daily_vwap(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    tpv = typical * df["volume"]
    cum_tpv = tpv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    df["vwap"] = (cum_tpv / cum_vol).ffill()
    return df

for pair in h1_data:
    add_daily_vwap(h1_data[pair])

# ── Simulacion: corre baseline y variante VWAP en la misma pasada ─────
def run_backtest(use_vwap_filter: bool):
    trade_log = []
    daily_pnl = defaultdict(float)

    for pair, df1 in h1_data.items():
        dfd = d1_data.get(pair, pd.DataFrame())
        open_pos = []

        for idx in range(80, len(df1)):
            bar = df1.iloc[idx]
            dt = df1.index[idx]
            if pd.Timestamp(dt).weekday() >= 5: continue
            hour_utc = pd.Timestamp(dt).hour
            if hour_utc in DEAD_HOURS_UTC: continue
            day_str = str(pd.Timestamp(dt).date())

            new_open = []
            for pos in open_pos:
                (eidx, direction, entry, sl, tp, vol_p, sl_dist, pip_v, pair_p) = pos
                pnl = None
                cur_h, cur_l = bar["high"], bar["low"]
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
                        trade_log.append({"pair": pair_p, "pnl": pnl, "win": pnl > 0})
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

            if use_vwap_filter:
                vwap_now = bar["vwap"]
                if pd.isna(vwap_now): continue
                cur_price = bar["close"]
                if sig == "LONG" and not (cur_price > vwap_now): continue
                if sig == "SHORT" and not (cur_price < vwap_now): continue

            thr = 80 if h4_d != "WAIT" else 90
            if score < thr: continue

            max_r, r_pct = risk_for_score(score)
            _PAIR_RISK_MULT = {"EURUSD": 1.8}
            max_r *= _PAIR_RISK_MULT.get(pair, 1.0)

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

            open_pos.append((idx, sig, entry, sl_p, tp_p, vol, sl_dist_p, pip_v, pair))

    return trade_log, daily_pnl

def monte_carlo_stats(daily_pnl, label):
    daily_vals = list(daily_pnl.values())
    if len(daily_vals) < 20:
        print(f"\n  [{label}] muy pocos dias con trades ({len(daily_vals)}) — no confiable")
        return None
    daily_arr = np.array(daily_vals)
    sims_month = rng.choice(daily_arr, size=(100_000, 22), replace=True).sum(axis=1)
    sims_day   = rng.choice(daily_arr, size=100_000, replace=True)
    e_monthly  = np.mean(sims_month)
    p_pass_axi = np.mean(sims_month >= 4851) * 100
    p250       = np.mean(sims_day >= 250) * 100
    sharpe     = np.mean(sims_month) / max(1, np.std(sims_month))
    p_neg5     = np.mean(sims_month <= -4851) * 100
    print(f"\n  === {label} ===")
    print(f"  Dias con trades:        {len(daily_vals)}")
    print(f"  P(dia >= $250):         {p250:.0f}%")
    print(f"  E[mensual]:             ${e_monthly:.0f}")
    print(f"  P(pass Axi Select 5%):  {p_pass_axi:.0f}%")
    print(f"  Sharpe mensual:         {sharpe:.2f}")
    print(f"  P(mes < -5%):           {p_neg5:.0f}%")
    return dict(p_pass_axi=p_pass_axi, e_monthly=e_monthly, sharpe=sharpe, p250=p250, n_days=len(daily_vals))

print("\n[RUN] Baseline (sin filtro VWAP)...")
trades_base, daily_base = run_backtest(use_vwap_filter=False)
n_final_base = len(trades_base)
wr_base = sum(1 for t in trades_base if t["win"]) / max(1, n_final_base) * 100
print(f"  Trades: {n_final_base} | WR: {wr_base:.1f}%")
r_base = monte_carlo_stats(daily_base, "BASELINE (sin VWAP)")

print("\n[RUN] Con filtro VWAP (LONG solo sobre VWAP, SHORT solo bajo VWAP)...")
trades_vwap, daily_vwap = run_backtest(use_vwap_filter=True)
n_final_vwap = len(trades_vwap)
wr_vwap = sum(1 for t in trades_vwap if t["win"]) / max(1, n_final_vwap) * 100
print(f"  Trades: {n_final_vwap} | WR: {wr_vwap:.1f}%")
r_vwap = monte_carlo_stats(daily_vwap, "CON FILTRO VWAP")

print("\n" + "=" * 72)
print("  COMPARACION FINAL")
print("=" * 72)
if r_base and r_vwap:
    print(f"  {'Metrica':28s} | {'Sin VWAP':>12s} | {'Con VWAP':>12s} | Delta")
    print("  " + "-"*70)
    print(f"  {'Trades totales':28s} | {n_final_base:>12d} | {n_final_vwap:>12d} | {n_final_vwap-n_final_base:+d}")
    print(f"  {'WR':28s} | {wr_base:>11.1f}% | {wr_vwap:>11.1f}% | {wr_vwap-wr_base:+.1f}pp")
    print(f"  {'P(pass Axi 5%)':28s} | {r_base['p_pass_axi']:>11.0f}% | {r_vwap['p_pass_axi']:>11.0f}% | {r_vwap['p_pass_axi']-r_base['p_pass_axi']:+.0f}pp")
    print(f"  {'E[mensual]':28s} | ${r_base['e_monthly']:>10.0f} | ${r_vwap['e_monthly']:>10.0f} | {r_vwap['e_monthly']-r_base['e_monthly']:+.0f}")
    print(f"  {'Sharpe mensual':28s} | {r_base['sharpe']:>12.2f} | {r_vwap['sharpe']:>12.2f} | {r_vwap['sharpe']-r_base['sharpe']:+.2f}")
    veredicto = "MEJORA -- considerar activar en vivo" if r_vwap['p_pass_axi'] > r_base['p_pass_axi'] + 2 else \
                ("EMPEORA -- no activar" if r_vwap['p_pass_axi'] < r_base['p_pass_axi'] - 2 else "SIN DIFERENCIA SIGNIFICATIVA")
    print(f"\n  VEREDICTO: {veredicto}")
print("\nDONE")
