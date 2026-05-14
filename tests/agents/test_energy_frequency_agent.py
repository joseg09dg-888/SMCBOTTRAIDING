"""Tests for EnergyFrequencyAgent — TDD first pass.
All tests are written BEFORE the implementation.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from agents.energy_frequency_agent import (
    EnergyFrequencyAgent,
    EnergyReading,
    TAROT_ARCANA,
    calculate_day_number,
    calculate_year_number,
    calculate_symbol_number,
    get_number_meaning,
    get_tarot_card,
    get_price_vibration,
    get_planetary_influences,
    calculate_hurst_exponent,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _date(day: int, month: int, year: int) -> datetime:
    return datetime(year, month, day, 12, 0, tzinfo=timezone.utc)


# ── NUMEROLOGÍA ────────────────────────────────────────────────────────────────

def test_calculate_day_number_basic():
    """14/05/2026 → 1+4+0+5+2+0+2+6=20 → 2+0=2"""
    result = calculate_day_number(_date(14, 5, 2026))
    assert result == 2


def test_calculate_day_number_master_11():
    """Find a date that sums to 11 — 29/01/2000: 2+9+0+1+2+0+0+0=14 → 1+4=5, nope.
    Try 29/06/1966: 2+9+0+6+1+9+6+6=39→12→3, nope.
    Use 02/09/2000: 0+2+0+9+2+0+0+0=13→4, nope.
    Use 11/09/2000: 1+1+0+9+2+0+0+0=13→4, nope.
    Use 29/09/1991: 2+9+0+9+1+9+9+1=40→4, nope.
    Use 02/09/1998: 0+2+0+9+1+9+9+8=38→11 ✓"""
    result = calculate_day_number(_date(2, 9, 1998))
    assert result == 11


def test_calculate_day_number_master_22():
    """Find date summing to 22.
    04/09/1999: 0+4+0+9+1+9+9+9=41→5, nope.
    04/09/2000: 0+4+0+9+2+0+0+0=15→6, nope.
    Use 09/04/1999: 0+9+0+4+1+9+9+9=41→5, nope.
    Use 04/09/2009: 0+4+0+9+2+0+0+9=24→6, nope.
    Use 28/08/2002: 2+8+0+8+2+0+0+2=22 ✓"""
    result = calculate_day_number(_date(28, 8, 2002))
    assert result == 22


def test_calculate_year_number():
    """2026 → 2+0+2+6=10 → 1+0=1"""
    assert calculate_year_number(2026) == 1


def test_calculate_symbol_number_btc():
    """BTC → B=2, T=20→2+0=2, C=3 → 2+2+3=7"""
    assert calculate_symbol_number("BTC") == 7


def test_get_number_meaning_8_bullish():
    """número 8 → bias='bullish', pts >= 2"""
    m = get_number_meaning(8)
    assert m["bias"] == "bullish"
    assert m["pts"] >= 2


def test_get_number_meaning_9_bearish():
    """número 9 → bias='bearish', pts <= -1"""
    m = get_number_meaning(9)
    assert m["bias"] == "bearish"
    assert m["pts"] <= -1


def test_get_number_meaning_master_22():
    """número maestro 22 → pts >= 2"""
    m = get_number_meaning(22)
    assert m["pts"] >= 2


def test_get_number_meaning_has_required_keys():
    """meaning dict has all required keys"""
    m = get_number_meaning(5)
    for key in ("number", "meaning", "bias", "pts"):
        assert key in m


# ── TAROT ──────────────────────────────────────────────────────────────────────

def test_get_tarot_card_date_gives_el_juicio():
    """14/05/2026 → digit sum=20 → El Juicio"""
    card = get_tarot_card(_date(14, 5, 2026))
    assert card["name"] == "El Juicio"


def test_get_tarot_card_never_exceeds_21():
    """for any date, returned arcano is between 0 and 21"""
    test_dates = [
        _date(31, 12, 2025),
        _date(1, 1, 2000),
        _date(15, 7, 2030),
        _date(28, 2, 2028),
    ]
    for d in test_dates:
        card = get_tarot_card(d)
        assert 0 <= card["index"] <= 21, f"Card index out of range for {d}: {card['index']}"


def test_tarot_arcana_has_22_cards():
    """TAROT_ARCANA has exactly 22 cards (0 to 21)"""
    assert len(TAROT_ARCANA) == 22
    assert 0 in TAROT_ARCANA
    assert 21 in TAROT_ARCANA


def test_tarot_la_torre_bearish():
    """arcano 16 La Torre → bias='bearish'"""
    assert TAROT_ARCANA[16]["bias"] == "bearish"


def test_tarot_el_sol_bullish():
    """arcano 19 El Sol → bias='bullish', pts=3"""
    assert TAROT_ARCANA[19]["bias"] == "bullish"
    assert TAROT_ARCANA[19]["pts"] == 3


def test_tarot_card_has_required_keys():
    """each TAROT_ARCANA entry has name, bias, pts, desc"""
    for idx, card in TAROT_ARCANA.items():
        for key in ("name", "bias", "pts", "desc"):
            assert key in card, f"Missing '{key}' in arcano {idx}"


# ── HURST ──────────────────────────────────────────────────────────────────────

def test_hurst_trending_series():
    """strongly trending series 1,2,...,50 → H > 0.55"""
    prices = list(range(1, 51))
    h = calculate_hurst_exponent(prices)
    assert h > 0.55, f"Expected H > 0.55 for trending series, got {h}"


def test_hurst_random_walk_returns_05_when_short():
    """fewer than 20 prices → H = 0.5 (insufficient data fallback)"""
    h = calculate_hurst_exponent([1.0, 2.0, 3.0, 4.0, 5.0])
    assert h == 0.5


def test_hurst_range_valid():
    """with 30 prices, 0.0 <= H <= 1.0"""
    import random
    random.seed(42)
    prices = [100.0 + random.gauss(0, 1) for _ in range(30)]
    h = calculate_hurst_exponent(prices)
    assert 0.0 <= h <= 1.0


def test_hurst_empty_returns_05():
    """empty list → 0.5"""
    assert calculate_hurst_exponent([]) == 0.5


# ── VIBRACIÓN DEL PRECIO ───────────────────────────────────────────────────────

def test_price_vibration_master_number():
    """67450 → 6+7+4+5+0=22 → is_master=True"""
    v = get_price_vibration(67450.0)
    assert v["digits_sum"] == 22
    assert v["is_master"] is True


def test_price_vibration_normal():
    """10000 → 1+0+0+0+0=1 → is_master=False"""
    v = get_price_vibration(10000.0)
    assert v["digits_sum"] == 1
    assert v["is_master"] is False


def test_price_vibration_has_required_keys():
    """vibration dict has digits_sum, is_master, pts, meaning"""
    v = get_price_vibration(50000.0)
    for key in ("digits_sum", "is_master", "pts", "meaning"):
        assert key in v


# ── PLANETAS ───────────────────────────────────────────────────────────────────

def test_planetary_influences_returns_dict():
    """get_planetary_influences() returns dict with key 'mercury_retrograde'"""
    result = get_planetary_influences(_date(14, 5, 2026))
    assert isinstance(result, dict)
    assert "mercury_retrograde" in result


def test_planetary_influences_has_pts():
    """returns dict with key 'pts' of type int"""
    result = get_planetary_influences(_date(14, 5, 2026))
    assert "pts" in result
    assert isinstance(result["pts"], int)


def test_planetary_influences_has_all_keys():
    """returns all expected planet keys"""
    result = get_planetary_influences(_date(14, 5, 2026))
    for key in ("mercury_retrograde", "jupiter_favorable", "saturn_restricts",
                "mars_volatile", "venus_favorable", "pts"):
        assert key in result


# ── ENERGY READING ─────────────────────────────────────────────────────────────

@pytest.fixture
def basic_reading():
    agent = EnergyFrequencyAgent()
    return agent.analyze("BTC", price=67450.0, as_of=_date(14, 5, 2026))


def test_energy_reading_score_range(basic_reading):
    """analyze() returns energy_score between -10 and +10"""
    assert -10.0 <= basic_reading.energy_score <= 10.0


def test_energy_reading_bias_valid_values(basic_reading):
    """energy_bias is one of the valid values"""
    valid = {"muy_alcista", "alcista", "neutral", "bajista", "muy_bajista"}
    assert basic_reading.energy_bias in valid


def test_energy_reading_has_tarot_card(basic_reading):
    """EnergyReading.tarot_card is not empty"""
    assert basic_reading.tarot_card != ""
    assert len(basic_reading.tarot_card) > 0


def test_energy_reading_has_all_fields(basic_reading):
    """EnergyReading has all required fields"""
    assert hasattr(basic_reading, "energy_score")
    assert hasattr(basic_reading, "energy_bias")
    assert hasattr(basic_reading, "dominant_energy")
    assert hasattr(basic_reading, "mercury_retrograde")
    assert hasattr(basic_reading, "eclipse_warning")
    assert hasattr(basic_reading, "numerology_pts")
    assert hasattr(basic_reading, "tarot_pts")
    assert hasattr(basic_reading, "lunar_pts")
    assert hasattr(basic_reading, "planetary_pts")
    assert hasattr(basic_reading, "price_vibration_pts")
    assert hasattr(basic_reading, "hurst_pts")
    assert hasattr(basic_reading, "day_number")
    assert hasattr(basic_reading, "reading")


def test_energy_reading_day_number_correct(basic_reading):
    """analyze with 14/05/2026 → day_number == 2"""
    assert basic_reading.day_number == 2


def test_energy_reading_tarot_card_el_juicio(basic_reading):
    """14/05/2026 → tarot card is El Juicio"""
    assert "Juicio" in basic_reading.tarot_card


# ── to_decision_pts ────────────────────────────────────────────────────────────

def _make_reading_with_score(score: float) -> EnergyReading:
    return EnergyReading(
        date=datetime.now(timezone.utc),
        symbol="BTC",
        energy_score=score,
        energy_bias="neutral",
        dominant_energy="test",
        warning=None,
        mercury_retrograde=False,
        eclipse_warning=False,
        numerology_pts=0,
        tarot_pts=0,
        lunar_pts=0,
        planetary_pts=0,
        price_vibration_pts=0,
        hurst_pts=0,
        tarot_card="Test Card",
        day_number=1,
        reading="test reading",
    )


def test_to_decision_pts_high_score():
    """energy_score=9.0 → to_decision_pts() == 15"""
    r = _make_reading_with_score(9.0)
    assert r.to_decision_pts() == 15


def test_to_decision_pts_negative_extreme():
    """energy_score=-9.0 → to_decision_pts() == -100 (block signal)"""
    r = _make_reading_with_score(-9.0)
    assert r.to_decision_pts() == -100


def test_to_decision_pts_neutral():
    """energy_score=0.5 → to_decision_pts() == 0"""
    r = _make_reading_with_score(0.5)
    assert r.to_decision_pts() == 0


def test_to_decision_pts_mid_positive():
    """energy_score=6.0 → to_decision_pts() == 10"""
    r = _make_reading_with_score(6.0)
    assert r.to_decision_pts() == 10


def test_to_decision_pts_mid_negative():
    """energy_score=-6.0 → to_decision_pts() == -10"""
    r = _make_reading_with_score(-6.0)
    assert r.to_decision_pts() == -10


def test_to_decision_pts_low_positive():
    """energy_score=3.0 → to_decision_pts() == 5"""
    r = _make_reading_with_score(3.0)
    assert r.to_decision_pts() == 5


# ── format_telegram ────────────────────────────────────────────────────────────

def test_format_telegram_contains_energy_score(basic_reading):
    """format_telegram() contains 'ENERGY SCORE'"""
    text = basic_reading.format_telegram()
    assert "ENERGY SCORE" in text.upper() or "energy score" in text.lower()


def test_format_telegram_contains_symbol(basic_reading):
    """format_telegram() contains the symbol passed"""
    text = basic_reading.format_telegram()
    assert "BTC" in text


def test_format_telegram_contains_tarot(basic_reading):
    """format_telegram() contains the tarot card name"""
    text = basic_reading.format_telegram()
    # card name or partial match
    assert "Juicio" in text or "tarot" in text.lower() or "arcano" in text.lower()


def test_format_telegram_is_string(basic_reading):
    """format_telegram() returns a non-empty string"""
    text = basic_reading.format_telegram()
    assert isinstance(text, str)
    assert len(text) > 50


# ── INTEGRACIÓN CON LUNAR ──────────────────────────────────────────────────────

def test_analyze_uses_lunar_agent_when_provided():
    """If lunar_agent mock with eclipse_warning=True → reading.eclipse_warning=True"""
    mock_lunar = MagicMock()
    mock_signal = MagicMock()
    mock_signal.eclipse_warning = True
    mock_signal.bias = "bearish"
    mock_signal.score_bonus = 3
    mock_lunar.get_current_phase.return_value = mock_signal

    agent = EnergyFrequencyAgent(lunar_agent=mock_lunar)
    reading = agent.analyze("BTC", price=50000.0, as_of=_date(14, 5, 2026))

    assert reading.eclipse_warning is True


def test_analyze_without_lunar_eclipse_warning_is_bool():
    """Without lunar_agent, eclipse_warning is still a bool"""
    agent = EnergyFrequencyAgent()
    reading = agent.analyze("BTC", price=50000.0, as_of=_date(14, 5, 2026))
    assert isinstance(reading.eclipse_warning, bool)


def test_analyze_works_without_lunar_agent():
    """EnergyFrequencyAgent() without args → analyze() does not raise"""
    agent = EnergyFrequencyAgent()
    reading = agent.analyze("ETH", price=3000.0)
    assert reading is not None
    assert isinstance(reading, EnergyReading)


def test_analyze_with_prices_history():
    """analyze() accepts prices_history list without raising"""
    agent = EnergyFrequencyAgent()
    prices = list(range(1, 51))  # trending
    reading = agent.analyze("BTC", price=50.0, prices_history=prices)
    assert isinstance(reading, EnergyReading)
    assert -10.0 <= reading.energy_score <= 10.0


def test_analyze_default_datetime_is_utc():
    """analyze() without as_of uses current time, returns valid reading"""
    agent = EnergyFrequencyAgent()
    reading = agent.analyze("BTC")
    assert reading.date is not None
    assert isinstance(reading.date, datetime)
