"""
HistoricalDataAgent — downloads and analyzes complete market history.

Data sources (all optional — degrades gracefully):
  - Crypto  : Binance API (genesis → today)
  - Forex / Stocks : yfinance (~1970s → today)
  - Macro   : FRED API (1950s → today, needs FRED_API_KEY in .env)

Storage    : SQLite at memory/historical_data.db
Integration: score_adjustment() plugs into DecisionFilter (+10/+5/+5)
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── Market catalogue ──────────────────────────────────────────────────────────

MARKET_ORIGINS: Dict[str, Tuple[str, str]] = {
    "BTCUSDT":  ("2009-01-03", "crypto"),
    "ETHUSDT":  ("2015-07-30", "crypto"),
    "BNBUSDT":  ("2017-07-14", "crypto"),
    "SOLUSDT":  ("2020-09-14", "crypto"),
    "ADAUSDT":  ("2017-10-02", "crypto"),
    "XRPUSDT":  ("2014-08-04", "crypto"),
    "EURUSD=X": ("1971-01-01", "forex"),
    "GBPUSD=X": ("1971-01-01", "forex"),
    "JPY=X":    ("1971-01-01", "forex"),
    "GC=F":     ("1973-01-01", "forex"),
    "^DJI":     ("1896-05-26", "index"),
    "^IXIC":    ("1971-02-05", "index"),
    "^GSPC":    ("1928-01-03", "index"),
    "CL=F":     ("1946-01-01", "commodity"),
    "NG=F":     ("1990-01-01", "commodity"),
}

SYMBOL_ALIASES: Dict[str, str] = {
    "BTC": "BTCUSDT",  "ETH": "ETHUSDT",  "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",  "ADA": "ADAUSDT",  "XRP": "XRPUSDT",
    "EURUSD":  "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "XAUUSD":  "GC=F",     "GOLD": "GC=F",
    "US30":    "^DJI",     "DOW":  "^DJI",
    "NAS100":  "^IXIC",    "NASDAQ": "^IXIC",
    "SPX500":  "^GSPC",    "SP500":  "^GSPC",
    "USOIL":   "CL=F",     "OIL": "CL=F",
    "NATGAS":  "NG=F",
}

BTC_HALVINGS = [
    {"date": "2012-11-28", "number": 1, "cycle_end": "2013-12-04"},
    {"date": "2016-07-09", "number": 2, "cycle_end": "2017-12-17"},
    {"date": "2020-05-11", "number": 3, "cycle_end": "2021-11-10"},
    {"date": "2024-04-20", "number": 4, "cycle_end": None},
]

MONTH_NAMES = [
    "", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

# Historical BTC seasonality (empirical monthly bias %)
_BTC_SEASONALITY = {
    1: (13.5, 73), 2: (14.8, 64), 3: (8.2, 55), 4: (16.4, 73),
    5: (-6.1, 36), 6: (-5.0, 45), 7: (18.3, 73), 8: (4.5, 64),
    9: (-7.2, 36), 10: (22.1, 82), 11: (43.3, 91), 12: (4.1, 55),
}  # {month: (avg_return_pct, pct_positive)}


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class SeasonalityResult:
    symbol: str
    month: int
    month_name: str
    avg_return: float
    pct_positive: float
    sample_count: int
    is_bullish: bool

    def summary(self) -> str:
        direction = "alcista" if self.is_bullish else "bajista"
        return (
            f"{self.month_name}: {direction} el {self.pct_positive:.0f}% de los años "
            f"(retorno promedio {self.avg_return:+.1f}%, n={self.sample_count})"
        )


@dataclass
class HalvingPhase:
    halving_number: int
    halving_date: str
    phase_days: int
    phase_name: str
    cycle_year: int
    historical_note: str

    def summary(self) -> str:
        return (
            f"Halving #{self.halving_number} ({self.halving_date}) — "
            f"año {self.cycle_year} del ciclo, fase: {self.phase_name}. "
            f"{self.historical_note}"
        )


@dataclass
class HistoricalBonus:
    points: int
    reasons: List[str] = field(default_factory=list)

    def breakdown_str(self) -> str:
        return " | ".join(self.reasons) if self.reasons else "sin contexto histórico"


# ── SQL schema ────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS seasonality (
    symbol       TEXT    NOT NULL,
    month        INTEGER NOT NULL,
    avg_return   REAL,
    pct_positive REAL,
    sample_count INTEGER,
    PRIMARY KEY (symbol, month)
);

CREATE TABLE IF NOT EXISTS market_cycles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    cycle_type   TEXT    NOT NULL,
    start_date   TEXT,
    end_date     TEXT,
    pct_change   REAL,
    duration_days INTEGER
);

CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv_daily(symbol, date);
"""


class HistoricalDataAgent:
    """
    Downloads full market history, computes cycles and seasonality,
    and provides score_adjustment() for DecisionFilter integration.
    """

    def __init__(self, db_path: str = "memory/historical_data.db"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self):
        self._conn.close()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _init_db(self):
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ── Symbol resolution ─────────────────────────────────────────────────────

    def resolve_symbol(self, user_input: str) -> str:
        """Convert user alias ('BTC', 'EURUSD') to canonical market symbol."""
        s = user_input.strip().upper()
        return SYMBOL_ALIASES.get(s, s)

    # ── Download — Crypto (Binance) ───────────────────────────────────────────

    async def download_crypto(
        self,
        symbol: str,
        since: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Returns number of rows inserted. Skips silently if Binance unavailable."""
        try:
            from binance.client import Client
            from core.config import config
            client = Client(config.binance_api_key, config.binance_api_secret)
        except Exception:
            if on_progress:
                on_progress(f"[Glint] Binance no disponible para {symbol} — omitido")
            return 0

        rows = 0
        try:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, since)
            data = [
                (
                    symbol,
                    datetime.fromtimestamp(k[0] / 1000).strftime("%Y-%m-%d"),
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
                )
                for k in klines
            ]
            self._conn.executemany(
                "INSERT OR IGNORE INTO ohlcv_daily VALUES (?,?,?,?,?,?,?)", data
            )
            self._conn.commit()
            rows = len(data)
            if on_progress:
                on_progress(f"📊 {symbol}: {rows} velas diarias descargadas (Binance)")
        except Exception as e:
            if on_progress:
                on_progress(f"[Histórico] Error en {symbol}: {e}")
        return rows

    # ── Download — Forex / Stocks (yfinance) ─────────────────────────────────

    async def download_forex(
        self,
        symbol: str,
        since: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Returns number of rows inserted. Skips silently if yfinance unavailable."""
        try:
            import yfinance as yf
        except ImportError:
            if on_progress:
                on_progress(f"[Histórico] yfinance no instalado — omitiendo {symbol}")
            return 0

        rows = 0
        try:
            df = yf.download(symbol, start=since, progress=False, auto_adjust=True)
            if df.empty:
                return 0

            # yfinance returns MultiIndex columns when auto_adjust=True in newer versions
            if hasattr(df.columns, "get_level_values"):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception:
                    pass

            data = []
            for dt, row in df.iterrows():
                try:
                    data.append((
                        symbol,
                        str(dt.date()),
                        float(row.get("Open",  row.get("open",  0))),
                        float(row.get("High",  row.get("high",  0))),
                        float(row.get("Low",   row.get("low",   0))),
                        float(row.get("Close", row.get("close", 0))),
                        float(row.get("Volume", row.get("volume", 0))),
                    ))
                except Exception:
                    continue

            self._conn.executemany(
                "INSERT OR IGNORE INTO ohlcv_daily VALUES (?,?,?,?,?,?,?)", data
            )
            self._conn.commit()
            rows = len(data)
            if on_progress:
                on_progress(f"📊 {symbol}: {rows} velas diarias descargadas (yfinance)")
        except Exception as e:
            if on_progress:
                on_progress(f"[Histórico] Error en {symbol}: {e}")
        return rows

    # ── Download — Macro (FRED) ───────────────────────────────────────────────

    async def download_macro(
        self,
        series_id: str,
        since: str = "1950-01-01",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Downloads a FRED macro series. Needs FRED_API_KEY in .env."""
        try:
            from fredapi import Fred
            from core.config import config
            api_key = getattr(config, "fred_api_key", "") or ""
            if not api_key:
                return 0
            fred = Fred(api_key=api_key)
        except ImportError:
            return 0

        rows = 0
        try:
            series = fred.get_series(series_id, observation_start=since)
            data = [
                (f"FRED:{series_id}", str(dt.date()), None, None, None, float(v), None)
                for dt, v in series.items()
                if v is not None
            ]
            self._conn.executemany(
                "INSERT OR IGNORE INTO ohlcv_daily VALUES (?,?,?,?,?,?,?)", data
            )
            self._conn.commit()
            rows = len(data)
            if on_progress:
                on_progress(f"📊 FRED:{series_id}: {rows} observaciones descargadas")
        except Exception as e:
            if on_progress:
                on_progress(f"[Histórico] FRED {series_id}: {e}")
        return rows

    # ── Orchestrate full download ─────────────────────────────────────────────

    async def download_all(
        self, on_progress: Optional[Callable[[str], None]] = None
    ) -> Dict[str, int]:
        """Downloads all markets. Safe to call multiple times — uses INSERT OR IGNORE."""
        results: Dict[str, int] = {}
        total = len(MARKET_ORIGINS)
        done = 0

        for symbol, (since, market_type) in MARKET_ORIGINS.items():
            done += 1
            pct = int(done / total * 100)
            if on_progress:
                on_progress(f"📊 Descargando histórico {symbol} desde {since}... {pct}%")

            if market_type == "crypto":
                n = await self.download_crypto(symbol, since, on_progress)
            else:
                n = await self.download_forex(symbol, since, on_progress)
            results[symbol] = n

        # Compute seasonality for all symbols we have data for
        for symbol in results:
            self.compute_seasonality(symbol)

        self._set_meta("last_download", datetime.utcnow().isoformat())
        if on_progress:
            total_rows = sum(results.values())
            on_progress(f"✅ Histórico completo: {total_rows:,} filas en {len(results)} mercados")
        return results

    # ── Seasonality ───────────────────────────────────────────────────────────

    def compute_seasonality(self, symbol: str) -> int:
        """
        Calculates average monthly return and % of positive months from stored
        OHLCV data. Upserts into the seasonality table.
        Returns number of months computed.
        """
        rows = self._conn.execute(
            "SELECT date, close FROM ohlcv_daily WHERE symbol=? AND close IS NOT NULL ORDER BY date",
            (symbol,),
        ).fetchall()
        if len(rows) < 24:
            return 0

        # Build monthly_returns[month] = [pct_return, ...]
        monthly: Dict[int, List[float]] = {m: [] for m in range(1, 13)}

        for i in range(1, len(rows)):
            prev_close = rows[i - 1]["close"]
            curr_close = rows[i]["close"]
            curr_date  = rows[i]["date"]
            if not prev_close or prev_close == 0:
                continue
            month = int(curr_date[5:7])
            ret = (curr_close - prev_close) / prev_close * 100
            monthly[month].append(ret)

        inserted = 0
        for month, returns in monthly.items():
            if not returns:
                continue
            avg_ret = sum(returns) / len(returns)
            pct_pos = sum(1 for r in returns if r > 0) / len(returns) * 100
            self._conn.execute(
                """INSERT OR REPLACE INTO seasonality
                   (symbol, month, avg_return, pct_positive, sample_count)
                   VALUES (?,?,?,?,?)""",
                (symbol, month, round(avg_ret, 4), round(pct_pos, 2), len(returns)),
            )
            inserted += 1
        self._conn.commit()
        return inserted

    def get_seasonality(self, symbol: str, month: int) -> Optional[SeasonalityResult]:
        """
        Returns seasonality for symbol+month. For BTC falls back to empirical
        table if DB has no data yet.
        """
        canon = self.resolve_symbol(symbol)
        row = self._conn.execute(
            "SELECT * FROM seasonality WHERE symbol=? AND month=?",
            (canon, month),
        ).fetchone()

        if row:
            avg_ret  = row["avg_return"]
            pct_pos  = row["pct_positive"]
            n        = row["sample_count"]
        elif canon == "BTCUSDT" and month in _BTC_SEASONALITY:
            avg_ret, pct_pos = _BTC_SEASONALITY[month]
            n = 14  # empirical years
        else:
            return None

        return SeasonalityResult(
            symbol=canon,
            month=month,
            month_name=MONTH_NAMES[month],
            avg_return=avg_ret,
            pct_positive=pct_pos,
            sample_count=n,
            is_bullish=(avg_ret > 0 and pct_pos > 50),
        )

    # ── BTC Halving Cycle ─────────────────────────────────────────────────────

    def get_btc_halving_phase(self, as_of: Optional[date] = None) -> HalvingPhase:
        """Returns the current BTC halving phase based on elapsed days."""
        today = as_of or date.today()
        today_str = today.isoformat()

        # Find most recent halving before today
        last_halving = None
        for h in sorted(BTC_HALVINGS, key=lambda x: x["date"]):
            if h["date"] <= today_str:
                last_halving = h

        if last_halving is None:
            last_halving = BTC_HALVINGS[0]

        halving_date = date.fromisoformat(last_halving["date"])
        phase_days   = (today - halving_date).days
        cycle_year   = min(phase_days // 365 + 1, 4)

        # Phase detection based on historical 4-year cycle
        if phase_days <= 180:
            phase_name = "acumulación post-halving"
            note = "Ciclos previos: lateral 6 meses tras el halving"
        elif phase_days <= 540:
            phase_name = "bull temprano"
            note = "Ciclos previos: +200-500% en esta fase"
        elif phase_days <= 900:
            phase_name = "bull tardío / pico"
            note = "Ciclos previos: máximos históricos, alta volatilidad"
        else:
            phase_name = "bear / acumulación"
            note = "Ciclos previos: corrección -70-80%, zona de acumulación"

        return HalvingPhase(
            halving_number=last_halving["number"],
            halving_date=last_halving["date"],
            phase_days=phase_days,
            phase_name=phase_name,
            cycle_year=cycle_year,
            historical_note=note,
        )

    # ── Cycle detection ───────────────────────────────────────────────────────

    def detect_cycles(self, symbol: str) -> List[Dict]:
        """
        Detects bull/bear market cycles from stored daily OHLCV.
        Uses 20% rule: 20% decline from peak = bear, 20% rise from trough = bull.
        """
        rows = self._conn.execute(
            "SELECT date, close FROM ohlcv_daily WHERE symbol=? AND close IS NOT NULL ORDER BY date",
            (symbol,),
        ).fetchall()
        if len(rows) < 60:
            return []

        closes = [(r["date"], r["close"]) for r in rows]
        cycles = []
        peak_date, peak_price = closes[0]
        trough_date, trough_price = closes[0]
        in_bear = False

        for dt, price in closes[1:]:
            if price > peak_price:
                if in_bear and (price - trough_price) / trough_price >= 0.20:
                    cycles.append({
                        "symbol": symbol, "cycle_type": "bear",
                        "start_date": peak_date, "end_date": trough_date,
                        "pct_change": round((trough_price - peak_price) / peak_price * 100, 2),
                        "duration_days": (
                            date.fromisoformat(trough_date) - date.fromisoformat(peak_date)
                        ).days,
                    })
                    in_bear = False
                peak_price, peak_date = price, dt
            elif (peak_price - price) / peak_price >= 0.20:
                if not in_bear:
                    in_bear = True
                if price < trough_price:
                    trough_price, trough_date = price, dt

        return cycles

    def save_cycles(self, cycles: List[Dict]):
        """Upserts detected cycles into market_cycles table."""
        for c in cycles:
            self._conn.execute(
                """INSERT OR IGNORE INTO market_cycles
                   (symbol, cycle_type, start_date, end_date, pct_change, duration_days)
                   VALUES (:symbol,:cycle_type,:start_date,:end_date,:pct_change,:duration_days)""",
                c,
            )
        self._conn.commit()

    # ── Historical price levels ───────────────────────────────────────────────

    def get_historical_levels(
        self, symbol: str, current_price: float, pct_range: float = 0.03
    ) -> List[Dict]:
        """
        Returns dates where price was within ±pct_range of current_price.
        These are potential historical OB/FVG zones.
        """
        lo = current_price * (1 - pct_range)
        hi = current_price * (1 + pct_range)
        rows = self._conn.execute(
            """SELECT date, close FROM ohlcv_daily
               WHERE symbol=? AND close BETWEEN ? AND ?
               ORDER BY date DESC LIMIT 20""",
            (self.resolve_symbol(symbol), lo, hi),
        ).fetchall()
        return [{"date": r["date"], "close": r["close"]} for r in rows]

    # ── score_adjustment ─────────────────────────────────────────────────────

    def score_adjustment(
        self,
        symbol: str,
        bias: str,
        month: int,
        price: float,
    ) -> HistoricalBonus:
        """
        Returns bonus points (0-20) for DecisionFilter.
          +10 — macro cycle aligned with trade bias (BTC halving phase)
          +5  — monthly seasonality favorable
          +5  — historical price level confirmed near entry
        """
        bonus = HistoricalBonus(points=0)
        canon = self.resolve_symbol(symbol)

        # +10: BTC halving cycle alignment
        if canon == "BTCUSDT":
            phase = self.get_btc_halving_phase()
            cycle_is_bullish = phase.phase_name in ("bull temprano", "bull tardío / pico",
                                                     "acumulación post-halving")
            if cycle_is_bullish and bias == "bullish":
                bonus.points += 10
                bonus.reasons.append(f"ciclo halving #{phase.halving_number} bullish +10")
            elif not cycle_is_bullish and bias == "bearish":
                bonus.points += 10
                bonus.reasons.append(f"ciclo halving #{phase.halving_number} bearish +10")

        # +5: Seasonality
        seasonal = self.get_seasonality(symbol, month)
        if seasonal:
            if seasonal.is_bullish and bias == "bullish":
                bonus.points += 5
                bonus.reasons.append(
                    f"estacionalidad {MONTH_NAMES[month]} alcista "
                    f"({seasonal.pct_positive:.0f}%) +5"
                )
            elif not seasonal.is_bullish and bias == "bearish":
                bonus.points += 5
                bonus.reasons.append(
                    f"estacionalidad {MONTH_NAMES[month]} bajista "
                    f"({seasonal.pct_positive:.0f}%) +5"
                )

        # +5: Historical price level near entry
        if price > 0:
            levels = self.get_historical_levels(symbol, price)
            if len(levels) >= 3:
                bonus.points += 5
                bonus.reasons.append(
                    f"nivel histórico confirmado ({len(levels)} toques) +5"
                )

        bonus.points = min(bonus.points, 20)
        return bonus

    # ── /history command output ───────────────────────────────────────────────

    def get_market_summary(self, symbol: str, as_of: Optional[date] = None) -> str:
        """Returns a formatted Telegram-ready summary for /history [symbol]."""
        canon = self.resolve_symbol(symbol)
        today = as_of or date.today()
        month = today.month

        lines = [f"📊 *Análisis Histórico — {symbol.upper()}*\n"]

        # BTC halving
        if canon == "BTCUSDT":
            phase = self.get_btc_halving_phase(today)
            lines.append(f"⚡ *Ciclo Halving:* {phase.summary()}\n")

        # Seasonality
        seasonal = self.get_seasonality(symbol, month)
        if seasonal:
            lines.append(f"📅 *Estacionalidad ({MONTH_NAMES[month]}):* {seasonal.summary()}\n")

        # Historical levels from DB
        last_row = self._conn.execute(
            "SELECT close, date FROM ohlcv_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (canon,),
        ).fetchone()
        if last_row:
            last_price = last_row["close"]
            levels = self.get_historical_levels(symbol, last_price)
            if levels:
                lines.append(
                    f"🎯 *Nivel histórico cerca del precio actual:* "
                    f"{len(levels)} toques entre ${last_price*0.97:,.0f}–${last_price*1.03:,.0f}\n"
                )

        # Data availability
        count = self._conn.execute(
            "SELECT COUNT(*) as n FROM ohlcv_daily WHERE symbol=?", (canon,)
        ).fetchone()
        if count and count["n"] > 0:
            lines.append(f"📈 *Datos disponibles:* {count['n']:,} velas diarias en BD")
        else:
            lines.append("ℹ️ Sin datos históricos descargados aún. Ejecuta /train para iniciar.")

        return "\n".join(lines)

    # ── Scheduled daily refresh ───────────────────────────────────────────────

    async def daily_refresh(self, on_progress: Optional[Callable[[str], None]] = None):
        """Downloads only the last 7 days for all symbols (fast daily update)."""
        since = (date.today() - timedelta(days=7)).isoformat()
        for symbol, (_, market_type) in MARKET_ORIGINS.items():
            if market_type == "crypto":
                await self.download_crypto(symbol, since, on_progress)
            else:
                await self.download_forex(symbol, since, on_progress)
        # Recompute seasonality after refresh
        for symbol in MARKET_ORIGINS:
            self.compute_seasonality(symbol)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_meta(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?,?,?)",
            (key, value, datetime.utcnow().isoformat()),
        )
        self._conn.commit()

    def _get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None
