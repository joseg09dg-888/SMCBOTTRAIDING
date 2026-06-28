"""
Optimization backtest — find parameters that maximize P(daily >= $250).

Tests:
1. RR ratio (1.5, 2.0, 2.5, 3.0)
2. H1 threshold (80, 85, 90, 100)
3. Kill zone filter (13-21 UTC vs full 13-23 UTC)
4. Trailing stop at 1R breakeven effect

Uses same historical data — runs grid search.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

import yfinance as yf

CAPITAL = 97_022.0
RISK_PCT = 0.005
MAX_DOLLAR_RISK = 275.0
MAX_POSITIONS = 3
DAILY_TARGET = 250.0

PAIRS_YFINANCE = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NAS100":  "^NDX",
}
PIP_SIZE  = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001, "NAS100": 1.0}
PIP_VALUE = {"EURUSD": 10.0,   "GBPUSD": 10.0,   "AUDUSD": 10.0,   "USDCAD": 10.0,   "NAS100": 1.0}
MIN_VOL   = {"EURUSD": 0.01,   "GBPUSD": 0.01,   "AUDUSD": 0.01,   "USDCAD": 0.01,   "NAS100": 1.0}
MAX_VOL   = {"EURUSD": 2.0,    "GBPUSD": 2.0,    "AUDUSD": 2.0,    "USDCAD": 2.0,    "NAS100": 1.0}

# ── Download data once ────────────────────────────────────────────────
END   = datetime.now()
START = END - timedelta(days=185)

print("Descargando datos historicos (6 meses)...")
h1_data, h4_data, d1_data = {}, {}, {}
for pair, ticker in PAIRS_YFINANCE.items():
    try:
        df1 = yf.download(ticker, start=START, end=END, interval="1h", progress=False, auto_adjust=True)
        df4d = yf.download(ticker, start=START, end=END, interval="1d", progress=False, auto_adjust=True)
        if df1.empty: continue
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.get_level_values(0)
        if isinstance(df4d.columns, pd.MultiIndex):
            df4d.columns = df4d.columns.get_level_values(0)
        df1.columns = [c.lower() for c in df1.columns]
        df4d.columns = [c.lower() for c in df4d.columns]
        df4h = df1.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        h1_data[pair] = df1
        h4_data[pair] = df4h
        d1_data[pair] = df4d
    except:
        pass
print(f"  Pares disponibles: {list(h1_data.keys())}")

# ── Helpers ───────────────────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def atr_calc(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def d1_dir(d1df, dt):
    d = d1df[d1df.index.date <= pd.Timestamp(dt).date()]
    if len(d) < 50: return "UNKNOWN"
    e = ema(d["close"], 50)
    return "LONG" if d["close"].iloc[-1] > e.iloc[-1] else "SHORT"

def h4_dir(h4df, dt):
    s = h4df[h4df.index <= pd.Timestamp(dt)]
    if len(s) < 20: return "WAIT"
    ef = ema(s["close"], 8).iloc[-1]
    es = ema(s["close"], 20).iloc[-1]
    return "LONG" if ef > es else ("SHORT" if ef < es else "WAIT")

def smc_signal(df, idx, bias, score_boost=0):
    if idx < 50: return False, "WAIT", 0, 0.0
    w = df.iloc[max(0, idx-50):idx+1]
    if len(w) < 30: return False, "WAIT", 0, 0.0
    atr_v = atr_calc(w, 14).iloc[-1]
    if pd.isna(atr_v) or atr_v <= 0: return False, "WAIT", 0, 0.0
    close = w["close"]
    high  = w["high"]
    low   = w["low"]
    e8  = ema(close, 8).iloc[-1]
    e21 = ema(close, 21).iloc[-1]
    cur = close.iloc[-1]
    prev_high = high.iloc[-15:-5].max()
    prev_low  = low.iloc[-15:-5].min()
    bullish_bos = cur > prev_high and close.iloc[-2] <= prev_high
    bearish_bos = cur < prev_low  and close.iloc[-2] >= prev_low
    momentum_up   = cur > close.iloc[-4]
    momentum_down = cur < close.iloc[-4]
    score = 0
    signal_dir = "WAIT"
    if bias == "LONG":
        if bullish_bos:     score += 30
        if e8 > e21:        score += 20
        if momentum_up:     score += 15
        if cur > e21:       score += 15
        body = (w["close"] - w["open"]).iloc[-10:]
        if body.max() > atr_v * 0.8: score += 20
        signal_dir = "LONG" if score >= 50 else "WAIT"
    elif bias == "SHORT":
        if bearish_bos:     score += 30
        if e8 < e21:        score += 20
        if momentum_down:   score += 15
        if cur < e21:       score += 15
        body = (w["open"] - w["close"]).iloc[-10:]
        if body.max() > atr_v * 0.8: score += 20
        signal_dir = "SHORT" if score >= 50 else "WAIT"
    score = min(100, int(score * 1.2) + score_boost)
    return signal_dir != "WAIT", signal_dir, score, float(atr_v)

def calc_vol(sl_pips, pair):
    pip_v = PIP_VALUE[pair]
    if sl_pips <= 0 or pip_v <= 0: return 0.0
    vol = (CAPITAL * RISK_PCT) / (sl_pips * pip_v)
    vol = max(MIN_VOL[pair], min(MAX_VOL[pair], vol))
    actual = vol * sl_pips * pip_v
    if actual > MAX_DOLLAR_RISK:
        vol = MAX_DOLLAR_RISK / (sl_pips * pip_v)
        vol = round(int(vol / 0.01) * 0.01, 2)
        vol = max(MIN_VOL[pair], vol)
    return round(vol, 2)

# ── Backtest function ─────────────────────────────────────────────────
def run_backtest(rr, h1_thresh, kill_zone_only, trailing_stop, label=""):
    """Run backtest with given parameters. Returns (avg_daily, p_above_250, win_rate, signals_per_day, pf)."""
    dead_hours = set(range(0, 13))
    if kill_zone_only:
        active_hours = set(range(13, 21))  # 13-20 UTC (London/NY overlap + NY)
    else:
        active_hours = set(range(13, 24))  # 13-23 UTC

    all_trades = []
    daily_pnl  = defaultdict(float)
    sig_per_day = defaultdict(int)

    for pair in h1_data:
        df1 = h1_data[pair]
        df4 = h4_data[pair]
        d1  = d1_data[pair]
        open_pos = []  # (idx, dir, entry, sl, tp, vol, sl_pips, breakeven_done)

        for idx in range(50, len(df1)):
            bar = df1.iloc[idx]
            dt  = df1.index[idx]
            if pd.Timestamp(dt).weekday() >= 5: continue
            hour_utc = pd.Timestamp(dt).hour
            if hour_utc not in active_hours: continue
            day_str = pd.Timestamp(dt).date()

            # Close/manage open positions
            closed = []
            for i, pos in enumerate(open_pos):
                eidx, direction, entry, sl, tp, vol, sl_pips, be_done = pos
                pip_v = PIP_VALUE[pair]
                pnl = None

                if trailing_stop and not be_done:
                    # Move SL to breakeven at 1R
                    if direction == "LONG":
                        pnl_dist = bar["high"] - entry
                        if pnl_dist >= (entry - sl):
                            sl = entry  # move to BE
                            open_pos[i] = (eidx, direction, entry, sl, tp, vol, sl_pips, True)
                    else:
                        pnl_dist = entry - bar["low"]
                        if pnl_dist >= (sl - entry):
                            sl = entry
                            open_pos[i] = (eidx, direction, entry, sl, tp, vol, sl_pips, True)

                if direction == "LONG":
                    if bar["high"] >= tp:
                        pnl = vol * sl_pips * pip_v * rr
                    elif bar["low"] <= sl:
                        pnl = -vol * sl_pips * pip_v
                else:
                    if bar["low"] <= tp:
                        pnl = vol * sl_pips * pip_v * rr
                    elif bar["high"] >= sl:
                        pnl = -vol * sl_pips * pip_v

                if pnl is not None:
                    daily_pnl[day_str] += pnl
                    all_trades.append({"pnl": pnl, "win": pnl > 0})
                    closed.append(i)

            for i in sorted(closed, reverse=True):
                open_pos.pop(i)

            if len(open_pos) >= MAX_POSITIONS: continue

            d_dir = d1_dir(d1, dt)
            h4_d  = h4_dir(df4, dt)
            if d_dir == "UNKNOWN": continue
            if h4_d not in (d_dir, "WAIT"): continue

            _, sig_dir, score, atr_v = smc_signal(df1, idx, d_dir)
            if sig_dir == "WAIT": continue

            # Score boost for H4 confirmation
            score_bonus = 15 if h4_d == d_dir else 0
            if h4_d == "WAIT":
                eff_thresh = 115
            else:
                eff_thresh = h1_thresh

            if score + score_bonus < eff_thresh: continue

            sig_per_day[day_str] += 1
            sl_atr = atr_v * 1.5
            sl_pips = sl_atr / PIP_SIZE[pair]
            vol = calc_vol(sl_pips, pair)
            if vol < MIN_VOL[pair] * 0.99: continue

            entry = bar["close"]
            if sig_dir == "LONG":
                sl_price = entry - sl_atr
                tp_price = entry + sl_atr * rr
            else:
                sl_price = entry + sl_atr
                tp_price = entry - sl_atr * rr

            open_pos.append((idx, sig_dir, entry, sl_price, tp_price, vol, sl_pips, False))

    if not all_trades:
        return 0, 0, 0, 0, 0

    total = len(all_trades)
    wins  = sum(1 for t in all_trades if t["win"])
    wr    = wins / total if total else 0
    daily_list = list(daily_pnl.values())
    avg_d  = np.mean(daily_list) if daily_list else 0
    p250   = sum(1 for v in daily_list if v >= 250) / len(daily_list) if daily_list else 0
    avg_win  = np.mean([t["pnl"] for t in all_trades if t["win"]]) if wins else 0
    avg_loss = abs(np.mean([t["pnl"] for t in all_trades if not t["win"]])) if (total-wins) else 1
    pf = (wins * avg_win) / ((total-wins) * avg_loss) if (total-wins) > 0 and avg_loss > 0 else 0
    n_days = len(daily_list)
    avg_sig = np.mean(list(sig_per_day.values())) if sig_per_day else 0
    return avg_d, p250, wr, avg_sig, pf, n_days, wins, total

# ── Grid Search ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  GRID SEARCH — Optimizando parámetros")
print("=" * 65)

configs = []
for rr in [1.5, 2.0, 2.5, 3.0]:
    for h1_thr in [75, 80, 85, 90]:
        for kz in [False, True]:
            for trail in [False, True]:
                result = run_backtest(rr, h1_thr, kz, trail)
                if len(result) < 6: continue
                avg_d, p250, wr, avg_sig, pf, n_days, wins, total = result
                configs.append({
                    "rr": rr, "h1_thr": h1_thr, "kill_zone": kz, "trailing": trail,
                    "avg_daily": avg_d, "p250": p250, "wr": wr,
                    "signals_day": avg_sig, "pf": pf, "n_days": n_days,
                    "wins": wins, "total": total
                })

# Sort by P(day >= $250)
configs.sort(key=lambda x: x["p250"], reverse=True)

print(f"\n  {'RR':4s} {'THR':5s} {'KZ':6s} {'TRAIL':7s} | {'WR':5s} {'PF':5s} {'Sig/D':6s} {'AvgPnL':8s} {'P>=250':7s}")
print("  " + "-" * 60)
for c in configs[:15]:
    kz   = "Y" if c["kill_zone"] else "N"
    trl  = "Y" if c["trailing"] else "N"
    print(f"  {c['rr']:.1f}  {c['h1_thr']:3d}   {kz}      {trl}      | "
          f"{c['wr']*100:4.0f}% {c['pf']:4.2f}  {c['signals_day']:5.1f}  "
          f"${c['avg_daily']:6.0f}   {c['p250']*100:4.0f}%  "
          f"({c['wins']}/{c['total']} trades)")

# Best config
best = configs[0]
print(f"\n  MEJOR CONFIG:")
print(f"    RR={best['rr']} | H1_threshold={best['h1_thr']} | kill_zone={best['kill_zone']} | trailing={best['trailing']}")
print(f"    Win rate: {best['wr']*100:.1f}%")
print(f"    Profit factor: {best['pf']:.2f}")
print(f"    Signals/day: {best['signals_day']:.1f}")
print(f"    Avg daily: ${best['avg_daily']:.0f}")
print(f"    P(dia >= $250): {best['p250']*100:.0f}%")

# ── Deeper analysis of best config ────────────────────────────────────
print("\n" + "=" * 65)
print(f"  ANALISIS PROFUNDO — Mejor config")
print("=" * 65)

# Run best config and capture daily distribution
b = best
dead_hours = set(range(0, 13))
active_hours = set(range(13, 21)) if b["kill_zone"] else set(range(13, 24))

all_trades_best = []
daily_pnl_best  = defaultdict(float)

for pair in h1_data:
    df1 = h1_data[pair]
    df4 = h4_data[pair]
    d1  = d1_data[pair]
    open_pos = []

    for idx in range(50, len(df1)):
        bar = df1.iloc[idx]
        dt  = df1.index[idx]
        if pd.Timestamp(dt).weekday() >= 5: continue
        hour_utc = pd.Timestamp(dt).hour
        if hour_utc not in active_hours: continue
        day_str = pd.Timestamp(dt).date()

        closed = []
        for i, pos in enumerate(open_pos):
            eidx, direction, entry, sl, tp, vol, sl_pips, be_done = pos
            pip_v = PIP_VALUE[pair]

            if b["trailing"] and not be_done:
                if direction == "LONG" and (bar["high"] - entry) >= (entry - sl):
                    sl = entry
                    open_pos[i] = (eidx, direction, entry, sl, tp, vol, sl_pips, True)
                elif direction == "SHORT" and (entry - bar["low"]) >= (sl - entry):
                    sl = entry
                    open_pos[i] = (eidx, direction, entry, sl, tp, vol, sl_pips, True)

            pnl = None
            if direction == "LONG":
                if bar["high"] >= tp: pnl = vol * sl_pips * pip_v * b["rr"]
                elif bar["low"] <= sl: pnl = -vol * sl_pips * pip_v
            else:
                if bar["low"] <= tp: pnl = vol * sl_pips * pip_v * b["rr"]
                elif bar["high"] >= sl: pnl = -vol * sl_pips * pip_v

            if pnl is not None:
                daily_pnl_best[day_str] += pnl
                all_trades_best.append({"pair": pair, "dir": direction, "date": day_str, "pnl": pnl, "win": pnl > 0})
                closed.append(i)

        for i in sorted(closed, reverse=True):
            open_pos.pop(i)

        if len(open_pos) >= MAX_POSITIONS: continue
        d_dir = d1_dir(d1, dt)
        h4_d  = h4_dir(df4, dt)
        if d_dir == "UNKNOWN": continue
        if h4_d not in (d_dir, "WAIT"): continue
        _, sig_dir, score, atr_v = smc_signal(df1, idx, d_dir)
        if sig_dir == "WAIT": continue
        score_bonus = 15 if h4_d == d_dir else 0
        eff_thresh = 115 if h4_d == "WAIT" else b["h1_thr"]
        if score + score_bonus < eff_thresh: continue
        sl_atr = atr_v * 1.5
        sl_pips = sl_atr / PIP_SIZE[pair]
        vol = calc_vol(sl_pips, pair)
        if vol < MIN_VOL[pair] * 0.99: continue
        entry = bar["close"]
        if sig_dir == "LONG":
            sl_p = entry - sl_atr
            tp_p = entry + sl_atr * b["rr"]
        else:
            sl_p = entry + sl_atr
            tp_p = entry - sl_atr * b["rr"]
        open_pos.append((idx, sig_dir, entry, sl_p, tp_p, vol, sl_pips, False))

daily_list = list(daily_pnl_best.values())
total_t = len(all_trades_best)
wins_t  = sum(1 for t in all_trades_best if t["win"])
avg_win_t  = np.mean([t["pnl"] for t in all_trades_best if t["win"]]) if wins_t > 0 else 0
avg_loss_t = np.mean([t["pnl"] for t in all_trades_best if not t["win"]]) if (total_t - wins_t) > 0 else 0

print(f"\n  Total trades: {total_t} ({wins_t} wins, {total_t-wins_t} losses)")
print(f"  Win rate: {wins_t/total_t*100:.1f}%")
print(f"  Avg win: ${avg_win_t:.0f} | Avg loss: ${avg_loss_t:.0f}")
print(f"  Trading days: {len(daily_list)}")
print(f"  Avg daily: ${np.mean(daily_list):.0f} | Std: ${np.std(daily_list):.0f}")
print(f"  Median daily: ${np.median(daily_list):.0f}")

# Distribution
bins = [-1200, -500, -250, 0, 250, 500, 1000, 3000]
labels = ["<-$500", "-$500/-$250", "-$250/$0", "$0/$250", "$250/$500", "$500/$1K", ">$1K"]
hist, _ = np.histogram(daily_list, bins=bins)
print("\n  DISTRIBUCION DIARIA:")
for label, count in zip(labels, hist):
    pct = count / len(daily_list) * 100
    bar_chart = "█" * int(pct / 2)
    print(f"    {label:15s}: {count:3d} dias ({pct:4.0f}%) {bar_chart}")

# Monte Carlo weekly
rng = np.random.default_rng(42)
weekly = rng.choice(daily_list, size=(1000, 5), replace=True).sum(axis=1)
monthly = rng.choice(daily_list, size=(1000, 22), replace=True).sum(axis=1)
print(f"\n  MONTE CARLO (1000 sim):")
print(f"    Avg semana: ${np.mean(weekly):.0f} | P(semana > $1250): {np.mean(weekly > 1250)*100:.0f}%")
print(f"    Avg mes:    ${np.mean(monthly):.0f} | P(mes >= $5K / 5% FTMO): {np.mean(monthly >= 5000)*100:.0f}%")

# ── Config to implement ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CAMBIOS RECOMENDADOS EN SUPERVISOR.PY")
print("=" * 65)

current_rr    = 2.5  # from supervisor MIN_RR
current_h1thr = 100

print(f"\n  Parametros actuales → Optimos:")
print(f"    RR:             {current_rr} → {best['rr']}")
print(f"    H1 threshold:   {current_h1thr} → {best['h1_thr']}")
print(f"    Kill zone only: No → {'Si (13-21 UTC)' if best['kill_zone'] else 'No'}")
print(f"    Trailing stop:  {'Si' if best['trailing'] else 'No'} → {'Si' if best['trailing'] else 'No'}")

improvement = best["p250"] - (15/48)  # vs original 31%
print(f"\n  Mejora en P(dia >= $250): {(15/48)*100:.0f}% → {best['p250']*100:.0f}% (+{improvement*100:.0f}pts)")
print(f"  Mejora en avg daily: $137 → ${best['avg_daily']:.0f}")
