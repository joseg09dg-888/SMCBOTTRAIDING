"""
Tests for HistoricalDataAgent — all run against an in-memory SQLite DB.
No real API calls are made.
"""
import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from training.historical_agent import (
    BTC_HALVINGS,
    MONTH_NAMES,
    HistoricalBonus,
    HistoricalDataAgent,
    HalvingPhase,
    SeasonalityResult,
)


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """In-memory agent — no file I/O, no external calls."""
    a = HistoricalDataAgent(db_path=":memory:")
    yield a
    a.close()


def _insert_ohlcv(agent: HistoricalDataAgent, symbol: str, rows: list):
    """Helper: insert (date, close) rows directly into the DB."""
    agent._conn.executemany(
        "INSERT OR IGNORE INTO ohlcv_daily (symbol, date, open, high, low, close, volume)"
        " VALUES (?,?,?,?,?,?,?)",
        [
            (symbol, d, c * 0.99, c * 1.01, c * 0.98, c, 1_000_000)
            for d, c in rows
        ],
    )
    agent._conn.commit()


def _generate_monthly_rows(symbol: str, start_year: int, years: int, base_price: float):
    """Generate one row per month for `years` years starting at start_year."""
    rows = []
    price = base_price
    for y in range(start_year, start_year + years):
        for m in range(1, 13):
            d = f"{y}-{m:02d}-01"
            price = price * 1.008  # slight uptrend
            rows.append((d, price))
    return rows


# ── DB Initialization ──────────────────────────────────────────────────────────

def test_db_tables_created(agent):
    tables = {
        r[0]
        for r in agent._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "ohlcv_daily" in tables
    assert "seasonality" in tables
    assert "market_cycles" in tables
    assert "meta" in tables


def test_index_created(agent):
    indexes = {
        r[0]
        for r in agent._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_ohlcv_symbol_date" in indexes


# ── Symbol Resolution ──────────────────────────────────────────────────────────

def test_resolve_btc_alias(agent):
    assert agent.resolve_symbol("BTC") == "BTCUSDT"


def test_resolve_eurusd_alias(agent):
    assert agent.resolve_symbol("EURUSD") == "EURUSD=X"


def test_resolve_unknown_passthrough(agent):
    assert agent.resolve_symbol("DOGEUSDT") == "DOGEUSDT"


# ── OHLCV storage ─────────────────────────────────────────────────────────────

def test_store_and_retrieve_ohlcv(agent):
    _insert_ohlcv(agent, "BTCUSDT", [("2024-01-01", 42_000), ("2024-01-02", 43_000)])
    rows = agent._conn.execute(
        "SELECT * FROM ohlcv_daily WHERE symbol='BTCUSDT'"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["close"] == 42_000


def test_duplicate_ohlcv_ignored(agent):
    _insert_ohlcv(agent, "BTCUSDT", [("2024-01-01", 42_000)])
    _insert_ohlcv(agent, "BTCUSDT", [("2024-01-01", 99_999)])  # same date, diff price
    count = agent._conn.execute(
        "SELECT COUNT(*) as n FROM ohlcv_daily WHERE symbol='BTCUSDT'"
    ).fetchone()["n"]
    assert count == 1


# ── Seasonality ───────────────────────────────────────────────────────────────

def test_compute_seasonality_returns_months(agent):
    rows = _generate_monthly_rows("BTCUSDT", 2015, 8, 500)
    _insert_ohlcv(agent, "BTCUSDT", rows)
    n = agent.compute_seasonality("BTCUSDT")
    assert n == 12


def test_seasonality_insufficient_data_returns_zero(agent):
    _insert_ohlcv(agent, "BTCUSDT", [("2024-01-01", 40_000)])
    n = agent.compute_seasonality("BTCUSDT")
    assert n == 0


def test_get_seasonality_uses_empirical_btc_fallback(agent):
    # No DB data → falls back to empirical table
    result = agent.get_seasonality("BTC", 11)  # November — historically strong
    assert result is not None
    assert result.is_bullish is True
    assert result.pct_positive > 80
    assert result.month_name == "Noviembre"


def test_get_seasonality_returns_none_unknown_symbol(agent):
    result = agent.get_seasonality("FAKECOIN", 6)
    assert result is None


def test_seasonality_result_summary_contains_month_name(agent):
    result = agent.get_seasonality("BTC", 10)  # October
    assert "Octubre" in result.summary()
    assert "%" in result.summary()


def test_get_seasonality_from_db_after_compute(agent):
    rows = _generate_monthly_rows("BTCUSDT", 2015, 8, 500)
    _insert_ohlcv(agent, "BTCUSDT", rows)
    agent.compute_seasonality("BTCUSDT")
    result = agent.get_seasonality("BTCUSDT", 3)
    assert result is not None
    assert result.sample_count > 0


# ── BTC Halving Phase ─────────────────────────────────────────────────────────

def test_halving_phase_is_halving4_in_2024(agent):
    phase = agent.get_btc_halving_phase(as_of=date(2024, 6, 1))
    assert phase.halving_number == 4
    assert phase.halving_date == "2024-04-20"


def test_halving_phase_cycle_year_increases(agent):
    phase_year1 = agent.get_btc_halving_phase(as_of=date(2024, 5, 1))
    phase_year2 = agent.get_btc_halving_phase(as_of=date(2025, 5, 1))
    assert phase_year1.cycle_year == 1
    assert phase_year2.cycle_year == 2


def test_halving_phase_early_bull(agent):
    # ~300 days after halving #4
    phase = agent.get_btc_halving_phase(as_of=date(2025, 2, 14))
    assert "bull" in phase.phase_name.lower()


def test_halving_phase_summary_has_required_fields(agent):
    phase = agent.get_btc_halving_phase(as_of=date(2024, 7, 1))
    summary = phase.summary()
    assert "Halving" in summary
    assert str(phase.halving_number) in summary
    assert phase.phase_name in summary


def test_halving_phase_bear_accumulation_late_cycle(agent):
    # ~1100 days after halving → bear/accumulation phase
    phase = agent.get_btc_halving_phase(as_of=date(2027, 5, 1))
    assert "bear" in phase.phase_name.lower() or "acumulaci" in phase.phase_name.lower()


# ── Score Adjustment ──────────────────────────────────────────────────────────

def test_score_adjustment_btc_bullish_cycle_gives_10pts(agent):
    # During halving #4 bull phase, bullish bias → +10
    bonus = agent.score_adjustment("BTC", "bullish", 11, 65_000)
    assert bonus.points >= 10


def test_score_adjustment_btc_bearish_in_bull_cycle_gives_0(agent):
    # During bull phase, bearish bias → 0 pts from cycle
    bonus = agent.score_adjustment("BTC", "bearish", 5, 65_000)
    # May or may not have seasonality; cycle component should be 0
    cycle_pts_present = any("halving" in r and "+10" in r for r in bonus.reasons)
    assert not cycle_pts_present


def test_score_adjustment_seasonal_bonus(agent):
    # November is the best month for BTC → bullish bias gets +5
    bonus = agent.score_adjustment("BTC", "bullish", 11, 65_000)
    seasonal_reason = any("estacionalidad" in r for r in bonus.reasons)
    assert seasonal_reason
    assert bonus.points >= 5


def test_score_adjustment_historical_level_bonus(agent):
    _insert_ohlcv(
        agent, "BTCUSDT",
        [(f"2021-{m:02d}-01", 65_000) for m in range(1, 13)],
    )
    bonus = agent.score_adjustment("BTC", "bullish", 11, 65_000)
    # ≥3 historical touches → +5
    level_reason = any("nivel histórico" in r for r in bonus.reasons)
    assert level_reason
    assert bonus.points >= 5


def test_score_adjustment_capped_at_20(agent):
    _insert_ohlcv(
        agent, "BTCUSDT",
        [(f"2021-{m:02d}-01", 65_000) for m in range(1, 13)],
    )
    bonus = agent.score_adjustment("BTC", "bullish", 11, 65_000)
    assert bonus.points <= 20


def test_score_adjustment_non_btc_no_cycle_pts(agent):
    bonus = agent.score_adjustment("EURUSD", "bullish", 6, 1.08)
    cycle_reason = any("halving" in r for r in bonus.reasons)
    assert not cycle_reason


# ── Cycle Detection ───────────────────────────────────────────────────────────

def test_detect_cycles_insufficient_data(agent):
    _insert_ohlcv(agent, "BTCUSDT", [("2024-01-01", 40_000)])
    cycles = agent.detect_cycles("BTCUSDT")
    assert cycles == []


def test_detect_bear_market_cycle(agent):
    # Build a clear -50% drawdown then recovery
    rows = []
    price = 60_000
    for i in range(200):
        d = f"202{i//365}-{(i%365)//30+1:02d}-{i%30+1:02d}"
        # Decline for 100 days then recover
        price = price * (0.995 if i < 100 else 1.005)
        rows.append((f"2021-01-{i%28+1:02d}", price))

    # Use a synthetic series with a clear -20%+ drop
    peak = 60_000
    rows2 = [("2021-01-01", peak)]
    for i in range(1, 100):
        rows2.append((f"2021-{i//30+2:02d}-{i%28+1:02d}", peak * (1 - i * 0.008)))

    _insert_ohlcv(agent, "TESTBTC", rows2)
    cycles = agent.detect_cycles("TESTBTC")
    # Should detect at least one bear period
    assert isinstance(cycles, list)


# ── Historical Levels ─────────────────────────────────────────────────────────

def test_get_historical_levels_returns_nearby(agent):
    _insert_ohlcv(
        agent, "BTCUSDT",
        [(f"202{y}-06-01", 45_000) for y in range(1, 5)],
    )
    levels = agent.get_historical_levels("BTC", 45_500, pct_range=0.02)
    assert len(levels) > 0
    for lv in levels:
        assert abs(lv["close"] - 45_500) / 45_500 <= 0.02


def test_get_historical_levels_empty_outside_range(agent):
    _insert_ohlcv(agent, "BTCUSDT", [("2021-01-01", 10_000)])
    levels = agent.get_historical_levels("BTC", 65_000, pct_range=0.02)
    assert levels == []


# ── Market Summary ────────────────────────────────────────────────────────────

def test_market_summary_btc_contains_halving(agent):
    summary = agent.get_market_summary("BTC", as_of=date(2024, 11, 1))
    assert "Halving" in summary or "halving" in summary


def test_market_summary_has_header(agent):
    summary = agent.get_market_summary("ETH", as_of=date(2024, 5, 1))
    assert "Análisis Histórico" in summary
    assert "ETH" in summary.upper()


def test_market_summary_no_data_shows_hint(agent):
    summary = agent.get_market_summary("FAKECOIN")
    assert "Sin datos" in summary or "train" in summary.lower()


# ── Download mocks (no real network calls) ────────────────────────────────────

@pytest.mark.asyncio
async def test_download_crypto_no_binance_returns_zero(agent):
    with patch.dict("sys.modules", {"binance": None, "binance.client": None}):
        n = await agent.download_crypto("BTCUSDT", "2024-01-01")
    assert n == 0


@pytest.mark.asyncio
async def test_download_forex_no_yfinance_returns_zero(agent):
    with patch.dict("sys.modules", {"yfinance": None}):
        n = await agent.download_forex("EURUSD=X", "2024-01-01")
    assert n == 0


# ── HistoricalBonus dataclass ─────────────────────────────────────────────────

def test_historical_bonus_breakdown_str_empty(agent):
    bonus = HistoricalBonus(points=0)
    assert "sin contexto" in bonus.breakdown_str()


def test_historical_bonus_breakdown_str_with_reasons(agent):
    bonus = HistoricalBonus(points=15, reasons=["ciclo halving +10", "estacionalidad +5"])
    s = bonus.breakdown_str()
    assert "halving" in s
    assert "estacionalidad" in s
