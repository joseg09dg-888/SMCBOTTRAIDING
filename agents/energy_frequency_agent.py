"""EnergyFrequencyAgent — numerología, tarot, Hurst, astrología básica y vibración de precio.

Dependencias opcionales: ephem, numpy.
Si no están instalados, las funciones degradan gracefully a valores neutros.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# ── Constantes de Tarot ────────────────────────────────────────────────────────

TAROT_ARCANA: dict[int, dict] = {
    0:  {"name": "El Loco",        "bias": "bullish",  "pts": 2,  "desc": "nueva tendencia"},
    1:  {"name": "El Mago",        "bias": "bullish",  "pts": 2,  "desc": "oportunidad"},
    2:  {"name": "La Sacerdotisa", "bias": "neutral",  "pts": 0,  "desc": "esperar"},
    3:  {"name": "La Emperatriz",  "bias": "bullish",  "pts": 1,  "desc": "crecimiento"},
    4:  {"name": "El Emperador",   "bias": "bullish",  "pts": 3,  "desc": "tendencia alcista"},
    5:  {"name": "El Hierofante",  "bias": "neutral",  "pts": 0,  "desc": "seguir reglas"},
    6:  {"name": "Los Amantes",    "bias": "bullish",  "pts": 1,  "desc": "decisión favorable"},
    7:  {"name": "El Carro",       "bias": "bullish",  "pts": 2,  "desc": "momentum"},
    8:  {"name": "La Fuerza",      "bias": "bullish",  "pts": 2,  "desc": "tendencia sostenida"},
    9:  {"name": "El Ermitaño",    "bias": "bearish",  "pts": -1, "desc": "esperar"},
    10: {"name": "La Rueda",       "bias": "neutral",  "pts": 1,  "desc": "cambio de ciclo"},
    11: {"name": "La Justicia",    "bias": "neutral",  "pts": 0,  "desc": "equilibrio"},
    12: {"name": "El Colgado",     "bias": "bearish",  "pts": -3, "desc": "pausa, no operar"},
    13: {"name": "La Muerte",      "bias": "bearish",  "pts": -2, "desc": "fin de ciclo"},
    14: {"name": "La Templanza",   "bias": "neutral",  "pts": 0,  "desc": "moderación"},
    15: {"name": "El Diablo",      "bias": "bearish",  "pts": -2, "desc": "trampa"},
    16: {"name": "La Torre",       "bias": "bearish",  "pts": -3, "desc": "crash extremo"},
    17: {"name": "La Estrella",    "bias": "bullish",  "pts": 2,  "desc": "esperanza"},
    18: {"name": "La Luna",        "bias": "bearish",  "pts": -2, "desc": "ilusión, engaño"},
    19: {"name": "El Sol",         "bias": "bullish",  "pts": 3,  "desc": "claridad alcista"},
    20: {"name": "El Juicio",      "bias": "bullish",  "pts": 1,  "desc": "renovación"},
    21: {"name": "El Mundo",       "bias": "bearish",  "pts": -1, "desc": "ciclo completo"},
}


# ── Numerología ────────────────────────────────────────────────────────────────

def _reduce_to_single(n: int) -> int:
    """Reduce n by summing its digits until single digit or master number."""
    while n > 9 and n not in (11, 22, 33):
        n = sum(int(d) for d in str(n))
    return n


def _digit_sum(n: int) -> int:
    """Sum all digits of n (no reduction)."""
    return sum(int(d) for d in str(abs(n)))


def calculate_day_number(date: datetime) -> int:
    """Sum digits of full date until 1-9 or master number (11, 22, 33).

    e.g. 14/05/2026 → 1+4+0+5+2+0+2+6 = 20 → 2+0 = 2
    EXCEPTIONS: if intermediate sum is 11, 22 or 33 → return that master number.
    """
    date_str = f"{date.day:02d}{date.month:02d}{date.year:04d}"
    total = sum(int(c) for c in date_str)
    return _reduce_to_single(total)


def calculate_year_number(year: int) -> int:
    """2026 → 2+0+2+6=10 → 1+0=1"""
    total = sum(int(d) for d in str(year))
    return _reduce_to_single(total)


def calculate_symbol_number(symbol: str) -> int:
    """BTC → B=2, T=20→2, C=3 → 2+2+3=7.
    Letter values: A=1..Z=26, reduce each >9 by summing its digits.
    """
    total = 0
    for ch in symbol.upper():
        if ch.isalpha():
            val = ord(ch) - ord("A") + 1  # A=1 .. Z=26
            if val > 9:
                val = _digit_sum(val)
            total += val
    return _reduce_to_single(total) if total > 9 else total


def get_number_meaning(n: int) -> dict:
    """Return meaning dict for numerological number n."""
    meanings = {
        1:  ("nuevos inicios",        "bullish",  2),
        2:  ("dualidad",              "neutral",  0),
        3:  ("creatividad",           "bullish",  1),
        4:  ("estabilidad",           "neutral",  0),
        5:  ("cambio",                "neutral",  0),
        6:  ("armonía",               "bullish",  1),
        7:  ("cierre de ciclos",      "bearish", -1),
        8:  ("abundancia",            "bullish",  2),
        9:  ("finales",               "bearish", -2),
        11: ("señal especial maestro","bullish",  2),
        22: ("manifestación maestro", "bullish",  2),
        33: ("maestría universal",    "bullish",  2),
    }
    meaning_text, bias, pts = meanings.get(n, ("desconocido", "neutral", 0))
    return {"number": n, "meaning": meaning_text, "bias": bias, "pts": pts}


# ── Tarot ──────────────────────────────────────────────────────────────────────

def get_tarot_card(date: datetime) -> dict:
    """Sum all digits of the date to get arcana index (0-21).

    e.g. 14/05/2026 → 1+4+0+5+2+0+2+6 = 20 → arcano 20 = El Juicio
    If sum > 21: reduce by summing digits. If still > 21: modulo 22.
    """
    date_str = f"{date.day:02d}{date.month:02d}{date.year:04d}"
    total = sum(int(c) for c in date_str)

    # Reduce if > 21 by summing digits (once)
    if total > 21:
        total = sum(int(d) for d in str(total))
    # If still > 21, use modulo 22
    if total > 21:
        total = total % 22

    arcana = TAROT_ARCANA[total].copy()
    arcana["index"] = total
    return arcana


# ── Hurst Exponent ─────────────────────────────────────────────────────────────

def calculate_hurst_exponent(prices: list) -> float:
    """Calculate Hurst exponent via R/S analysis.

    H > 0.5 → trending  (positive memory)
    H < 0.5 → mean-reverting
    H ≈ 0.5 → random walk
    Requires at least 20 prices; returns 0.5 if insufficient.
    """
    if len(prices) < 20:
        return 0.5

    try:
        import numpy as np
        ts = np.array(prices, dtype=float)
        n = len(ts)

        # Build a range of lags
        lags = range(10, n // 2 + 1, max(1, n // 20))
        rs_vals = []
        lag_vals = []

        for lag in lags:
            sub = ts[:lag]
            mean = np.mean(sub)
            deviation = np.cumsum(sub - mean)
            r = np.max(deviation) - np.min(deviation)
            s = np.std(sub, ddof=1)
            if s > 0:
                rs_vals.append(np.log(r / s))
                lag_vals.append(np.log(lag))

        if len(lag_vals) < 2:
            return 0.5

        # Linear regression slope = Hurst exponent
        coeffs = np.polyfit(lag_vals, rs_vals, 1)
        h = float(coeffs[0])
        return max(0.0, min(1.0, h))

    except Exception:
        # numpy not available or numerical failure → neutral
        return 0.5


# ── Vibración del precio ───────────────────────────────────────────────────────

def get_price_vibration(price: float) -> dict:
    """Sum digits of the integer part of the price.

    BTC $67,450 → 6+7+4+5+0 = 22 (master number)
    """
    int_part = int(abs(price))
    digits_sum = sum(int(d) for d in str(int_part))
    # Reduce to single or master
    reduced = _reduce_to_single(digits_sum)
    is_master = reduced in (11, 22, 33)

    # pts: master → +1, bullish numbers (1,3,8) → +1, bearish (7,9) → -1
    if is_master:
        pts = 1
        meaning = f"Número maestro {reduced} — vibración especial"
    elif reduced in (1, 3, 6, 8):
        pts = 1
        meaning = get_number_meaning(reduced)["meaning"]
    elif reduced in (7, 9):
        pts = -1
        meaning = get_number_meaning(reduced)["meaning"]
    else:
        pts = 0
        meaning = get_number_meaning(reduced)["meaning"]

    return {
        "digits_sum": reduced,
        "is_master": is_master,
        "pts": pts,
        "meaning": meaning,
    }


# ── Planetas ───────────────────────────────────────────────────────────────────

def get_planetary_influences(date: datetime) -> dict:
    """Calculate basic planetary influences using ephem if available.

    Returns dict with mercury_retrograde, jupiter_favorable, saturn_restricts,
    mars_volatile, venus_favorable, pts.
    If ephem unavailable → all False, pts=0.
    """
    result = {
        "mercury_retrograde": False,
        "jupiter_favorable": False,
        "saturn_restricts": False,
        "mars_volatile": False,
        "venus_favorable": False,
        "pts": 0,
    }

    try:
        import ephem

        # Normalize datetime for ephem
        dt = date if date.tzinfo is None else date.replace(tzinfo=None)

        # Mercury retrograde proxy: check if elongation is decreasing
        # Compare elongation today vs yesterday
        yesterday = ephem.Date(dt - __import__("datetime").timedelta(days=1))
        today_ep = ephem.Date(dt)

        mercury = ephem.Mercury()
        mercury.compute(today_ep)
        elong_today = float(mercury.elong)

        mercury.compute(yesterday)
        elong_yest = float(mercury.elong)

        # Elongation decreasing towards superior conjunction → proxy for retro
        mercury_retro = elong_today < elong_yest and abs(elong_today) < 0.5
        result["mercury_retrograde"] = mercury_retro

        # Jupiter: favorable if in even-numbered zodiac sign (expansion signs)
        jupiter = ephem.Jupiter()
        jupiter.compute(today_ep)
        # ecliptic longitude in degrees
        ecl = ephem.Ecliptic(jupiter.a_ra, jupiter.a_dec, epoch=today_ep)
        sign_idx = int(float(ecl.lon) * 180.0 / math.pi / 30.0) % 12
        result["jupiter_favorable"] = (sign_idx % 2 == 0)

        # Saturn: restricts if in restrictive signs (Capricorn=9, Aquarius=10, Scorpio=7)
        saturn = ephem.Saturn()
        saturn.compute(today_ep)
        ecl_sat = ephem.Ecliptic(saturn.a_ra, saturn.a_dec, epoch=today_ep)
        sat_sign = int(float(ecl_sat.lon) * 180.0 / math.pi / 30.0) % 12
        result["saturn_restricts"] = sat_sign in (7, 9, 10)

        # Mars: volatile if in Aries(0) or Scorpio(7)
        mars = ephem.Mars()
        mars.compute(today_ep)
        ecl_mars = ephem.Ecliptic(mars.a_ra, mars.a_dec, epoch=today_ep)
        mars_sign = int(float(ecl_mars.lon) * 180.0 / math.pi / 30.0) % 12
        result["mars_volatile"] = mars_sign in (0, 7)

        # Venus: favorable if in Taurus(1) or Libra(6)
        venus = ephem.Venus()
        venus.compute(today_ep)
        ecl_venus = ephem.Ecliptic(venus.a_ra, venus.a_dec, epoch=today_ep)
        venus_sign = int(float(ecl_venus.lon) * 180.0 / math.pi / 30.0) % 12
        result["venus_favorable"] = venus_sign in (1, 6)

        # Calculate pts
        pts = 0
        if result["mercury_retrograde"]:
            pts -= 2
        if result["jupiter_favorable"]:
            pts += 1
        if result["saturn_restricts"]:
            pts -= 1
        if result["mars_volatile"]:
            pts -= 1
        if result["venus_favorable"]:
            pts += 1
        result["pts"] = pts

    except Exception:
        pass  # graceful degradation — all defaults

    return result


# ── EnergyReading dataclass ────────────────────────────────────────────────────

@dataclass
class EnergyReading:
    date: datetime
    symbol: str
    energy_score: float      # -10.0 to +10.0
    energy_bias: str         # "muy_alcista", "alcista", "neutral", "bajista", "muy_bajista"
    dominant_energy: str     # description of dominant factor
    warning: Optional[str]   # e.g. "Mercurio retrógrado activo"
    mercury_retrograde: bool
    eclipse_warning: bool
    numerology_pts: int      # -2 to +2
    tarot_pts: int           # -3 to +3
    lunar_pts: int           # -1 to +1
    planetary_pts: int       # -2 to +2
    price_vibration_pts: int # -1 to +1
    hurst_pts: int           # -1 to +1
    tarot_card: str          # e.g. "El Sol (19)"
    day_number: int
    reading: str             # narrative text

    def to_decision_pts(self) -> int:
        """Convert energy_score to points for DecisionFilter.

        +8 to +10 → +15 pts
        +5 to +7  → +10 pts
        +2 to +4  → +5 pts
        -1 to +1  → 0 pts
        -2 to -4  → -5 pts
        -5 to -7  → -10 pts
        -8 to -10 → -15 pts (and block: return -100 as signal)
        """
        s = self.energy_score
        if s >= 8.0:
            return 15
        elif s >= 5.0:
            return 10
        elif s >= 2.0:
            return 5
        elif s >= -1.0:
            return 0
        elif s >= -4.0:
            return -5
        elif s >= -7.0:
            return -10
        else:
            return -100  # block signal

    def format_telegram(self) -> str:
        """Format reading as Telegram message with emojis and sections."""
        bias_emoji = {
            "muy_alcista": "🚀",
            "alcista":     "📈",
            "neutral":     "⚖️",
            "bajista":     "📉",
            "muy_bajista": "🌑",
        }.get(self.energy_bias, "⚖️")

        score_bar = self._score_bar(self.energy_score)
        warning_line = f"\n⚠️ *ADVERTENCIA:* {self.warning}" if self.warning else ""

        lines = [
            f"🔮 *ENERGY FREQUENCY READING — {self.symbol}*",
            f"📅 {self.date.strftime('%d/%m/%Y %H:%M UTC')}",
            "",
            f"{bias_emoji} *Energy Score: {self.energy_score:+.1f}/10*  {score_bar}",
            f"Sesgo: *{self.energy_bias.replace('_', ' ').upper()}*",
            warning_line,
            "",
            "━━━ *COMPONENTES* ━━━",
            f"🔢 Numerología: {self.numerology_pts:+d} pts  (número del día: {self.day_number})",
            f"🃏 Tarot: {self.tarot_pts:+d} pts  [{self.tarot_card}]",
            f"🌙 Lunar: {self.lunar_pts:+d} pts",
            f"🪐 Planetas: {self.planetary_pts:+d} pts"
            + (" ⚠️ Mercurio ℞" if self.mercury_retrograde else ""),
            f"💰 Vibración precio: {self.price_vibration_pts:+d} pts",
            f"📊 Hurst: {self.hurst_pts:+d} pts",
            "",
            f"✨ *Factor dominante:* {self.dominant_energy}",
            "",
            "━━━ *LECTURA* ━━━",
            self.reading,
            "",
            f"🎯 DecisionFilter pts: {self.to_decision_pts():+d}",
        ]
        return "\n".join(line for line in lines if line is not None)

    @staticmethod
    def _score_bar(score: float) -> str:
        """Visual bar from -10 to +10."""
        filled = int((score + 10) / 20 * 10)
        filled = max(0, min(10, filled))
        return "[" + "█" * filled + "░" * (10 - filled) + "]"


# ── EnergyFrequencyAgent ───────────────────────────────────────────────────────

class EnergyFrequencyAgent:
    """Calculates holistic energy reading combining numerology, tarot, Hurst,
    planetary influences, and price vibration."""

    def __init__(self, lunar_agent=None):
        self._lunar = lunar_agent

    def analyze(
        self,
        symbol: str,
        price: float = 0.0,
        prices_history: Optional[list] = None,
        as_of: Optional[datetime] = None,
    ) -> EnergyReading:
        """Compute full energy reading.

        as_of=None → datetime.now(UTC)
        """
        now = as_of or datetime.now(timezone.utc)

        # ── Numerología
        day_num = calculate_day_number(now)
        num_meaning = get_number_meaning(day_num)
        numerology_pts = num_meaning["pts"]

        sym_num = calculate_symbol_number(symbol)
        sym_meaning = get_number_meaning(sym_num)
        # Blend: average rounded to int
        numerology_pts = round((numerology_pts + sym_meaning["pts"]) / 2)

        # ── Tarot
        tarot = get_tarot_card(now)
        tarot_pts = tarot["pts"]
        tarot_card_str = f"{tarot['name']} ({tarot['index']})"

        # ── Lunar
        lunar_pts = 0
        eclipse_warning = False
        if self._lunar is not None:
            try:
                lunar_signal = self._lunar.get_current_phase(now)
                eclipse_warning = bool(lunar_signal.eclipse_warning)
                if lunar_signal.bias == "bullish":
                    lunar_pts = 1
                elif lunar_signal.bias == "bearish":
                    lunar_pts = -1
                else:
                    lunar_pts = 0
            except Exception:
                pass

        # ── Planetas
        planet_info = get_planetary_influences(now)
        planetary_pts = planet_info["pts"]
        mercury_retro = planet_info["mercury_retrograde"]

        # ── Vibración del precio
        price_vib = get_price_vibration(price) if price > 0 else {"pts": 0, "digits_sum": 0, "is_master": False, "meaning": "sin precio"}
        price_vib_pts = price_vib["pts"]

        # ── Hurst
        hurst_pts = 0
        if prices_history and len(prices_history) >= 20:
            h = calculate_hurst_exponent(prices_history)
            if h > 0.6:
                hurst_pts = 1
            elif h < 0.4:
                hurst_pts = -1
            else:
                hurst_pts = 0

        # ── Calcular energy_score total
        raw_score = (
            numerology_pts
            + tarot_pts
            + lunar_pts
            + planetary_pts
            + price_vib_pts
            + hurst_pts
        )
        # Scale to -10..+10 range (raw can be ~-9 to +9)
        energy_score = float(max(-10.0, min(10.0, raw_score)))

        # ── Bias
        if energy_score >= 6:
            energy_bias = "muy_alcista"
        elif energy_score >= 2:
            energy_bias = "alcista"
        elif energy_score >= -2:
            energy_bias = "neutral"
        elif energy_score >= -6:
            energy_bias = "bajista"
        else:
            energy_bias = "muy_bajista"

        # ── Dominant energy factor
        components = {
            "Tarot":         abs(tarot_pts),
            "Numerología":   abs(numerology_pts),
            "Lunar":         abs(lunar_pts),
            "Planetas":      abs(planetary_pts),
            "Vibr. precio":  abs(price_vib_pts),
            "Hurst":         abs(hurst_pts),
        }
        dominant_energy = max(components, key=components.get)
        if components[dominant_energy] == 0:
            dominant_energy = "Energía neutra equilibrada"
        else:
            dominant_energy = f"{dominant_energy} ({tarot['desc'] if dominant_energy == 'Tarot' else num_meaning['meaning']})"

        # ── Warnings
        warnings = []
        if mercury_retro:
            warnings.append("Mercurio retrógrado activo")
        if eclipse_warning:
            warnings.append("Eclipse próximo — alta volatilidad")
        if tarot.get("index") in (12, 16):
            warnings.append(f"Carta de alta alerta: {tarot['name']}")
        warning_str = " | ".join(warnings) if warnings else None

        # ── Lectura narrativa
        reading = _build_reading(symbol, energy_score, energy_bias, day_num, tarot, num_meaning, eclipse_warning)

        return EnergyReading(
            date=now,
            symbol=symbol,
            energy_score=energy_score,
            energy_bias=energy_bias,
            dominant_energy=dominant_energy,
            warning=warning_str,
            mercury_retrograde=mercury_retro,
            eclipse_warning=eclipse_warning,
            numerology_pts=numerology_pts,
            tarot_pts=tarot_pts,
            lunar_pts=lunar_pts,
            planetary_pts=planetary_pts,
            price_vibration_pts=price_vib_pts,
            hurst_pts=hurst_pts,
            tarot_card=tarot_card_str,
            day_number=day_num,
            reading=reading,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_reading(
    symbol: str,
    score: float,
    bias: str,
    day_num: int,
    tarot: dict,
    num_meaning: dict,
    eclipse: bool,
) -> str:
    """Build a short narrative reading."""
    bias_map = {
        "muy_alcista": "Las energías convergen con fuerza alcista",
        "alcista":     "El campo vibracional favorece posiciones largas",
        "neutral":     "Las fuerzas se equilibran; cautela recomendada",
        "bajista":     "La frecuencia señala presión bajista",
        "muy_bajista": "Energía de alta alerta; evitar nuevas posiciones",
    }
    intro = bias_map.get(bias, "Energía indefinida")
    day_msg = f"El número del día {day_num} vibra en '{num_meaning['meaning']}'."
    tarot_msg = f"El arcano '{tarot['name']}' indica: {tarot['desc']}."
    eclipse_msg = " ⚠️ Eclipse en ventana — volatilidad extrema esperada." if eclipse else ""
    return f"{symbol} — {intro}. {day_msg} {tarot_msg}{eclipse_msg}"
