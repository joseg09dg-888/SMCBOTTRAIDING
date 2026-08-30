"""Shared config constants used by both TradingSupervisor and PositionGuardsMixin.

SIMPLIFY-2026-07-23: split out of supervisor.py so core/position_guards.py
(extracted from supervisor.py's _manage_open_positions) doesn't need to import
from the module it's mixed into. Single source of truth -- values unchanged.
"""

MT5_REAL_SCORE_THRESHOLD = 95   # techo absoluto WR<40% (fallback en excepciones) — backtest 2026-07-01: 95 NO mejora WR vs 80, solo reduce volumen
MT5_SCORE_AUTO_REDUCE    = 80   # recalibrado 2026-07-01: barrido thr x RR en 2 años reales muestra 80+RR3.0 = optimo (WR=41.7%, P(pasar 5%)=28.4% vs 8.5% con 90-95)
MAX_OPEN_POSITIONS       = 16   # 2026-08-30: HALLAZGO CRITICO -- el backtest de sesiones
                                 # previas (que fijo esto en 4) tenia un bug real: aplicaba
                                 # el limite POR PAR en vez de GLOBAL en toda la cuenta,
                                 # permitiendo simular hasta 4x6=24 posiciones simultaneas
                                 # reales, muy por encima del limite que este codigo en vivo
                                 # realmente impone (ver core/supervisor.py:2151-2156, cuenta
                                 # "existing" de TODOS los simbolos, no filtra por symbol).
                                 # Corregido el motor de simulacion (linea de tiempo unificada
                                 # entre pares) y re-barrido MAX_OPEN=2..24 sobre 16 anios
                                 # reales: sube P(pasar Axi) de forma consistente hasta ~16
                                 # (57.8%->79.9%), con rendimientos decrecientes claros despues
                                 # (24 ya revierte Sharpe). Con solo 6 pares activos, 16 en la
                                 # practica casi elimina el limite real (el bot rara vez tendra
                                 # 16+ señales simultaneas genuinas), asi que subir esto no
                                 # fuerza mas riesgo del que el propio flujo de señales genera.
                                 # Ver SESION_ACTUAL.md sección "VEREDICTO FINAL" para el
                                 # detalle completo del bug y el re-barrido.
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
