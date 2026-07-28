"""Shared config constants used by both TradingSupervisor and PositionGuardsMixin.

SIMPLIFY-2026-07-23: split out of supervisor.py so core/position_guards.py
(extracted from supervisor.py's _manage_open_positions) doesn't need to import
from the module it's mixed into. Single source of truth -- values unchanged.
"""

MT5_REAL_SCORE_THRESHOLD = 95   # techo absoluto WR<40% (fallback en excepciones) — backtest 2026-07-01: 95 NO mejora WR vs 80, solo reduce volumen
MT5_SCORE_AUTO_REDUCE    = 80   # recalibrado 2026-07-01: barrido thr x RR en 2 años reales muestra 80+RR3.0 = optimo (WR=41.7%, P(pasar 5%)=28.4% vs 8.5% con 90-95)
MAX_OPEN_POSITIONS       = 4    # 2026-07-17: backtest_multiyear.py confirmo 2 veces
                                 # (sesiones separadas) que MAX_OPEN=3 supera a 2:
                                 # P(mes>=5%) 44%->49%, E[mensual] $4104->$5287.
                                 # Subido de nuevo (era 3 originalmente, se bajo a 2 sin
                                 # evidencia registrada de por que).
                                 # 2026-07-28: re-barrido contra la config completa actual
                                 # (hora-14 bloqueada, H4-FILTER removido, D1 informativo,
                                 # STAGNANT=6h, PEAK_GUARD=400, RR=4.5) sobre 16 anios reales
                                 # -- MAX_OPEN=4 supera a 3: E[mensual] $12303->$14477 (+18%),
                                 # P(pasar Axi 5%) 68%->70%, Sharpe igual (0.84). Riesgo de cola
                                 # sube poco (P(mes<-5%) 11%->12%). Subido de 3 a 4.
DAILY_PROFIT_TARGET      = 250.0  # $250/dia → 5% mensual Axi Select
INITIAL_CAPITAL          = 100_000.0

# Recovery — simplificado: solo para emergencias
RECOVERY_SCALP_TP        = 10.0  # igual que normal
RECOVERY_SCALP_SL        = -4.0  # igual que normal
RECOVERY_TRIGGER_LOSS    = -150.0  # recovery si pierde $150 en el día (era -50: demasiado agresivo)
ACCEL_TRIGGER_PROFIT     = 50.0   # aceleración si gana $50 en el día
ACCEL_SCALP_TP           = 10.0
ACCEL_SCALP_SL           = -4.0
ACCEL_MAX_SCALPS         = 5
