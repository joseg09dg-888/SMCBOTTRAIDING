# BUGS_HISTORIAL — SMC Trading Bot

Cada bug documentado: causa raíz, fix aplicado, verificación.
Leer SIEMPRE antes de arreglar un bug nuevo.

| Fecha | Bug | Causa raíz | Fix | Archivo | Sigue arreglado? |
|-------|-----|------------|-----|---------|-----------------|
| 2026-06-29 | Bot no operaba (score=0) | momentum filter bloqueaba todas las señales por H4=WAIT permanente | Preservar LONG/SHORT previo si signal devuelve WAIT | supervisor.py:3517 | ✅ |
| 2026-06-29 | RECOVERY mode falso en cada ciclo | `_balance_peak=INITIAL_CAPITAL=$100K` vs balance real $96K → delta siempre > $500 | Usar `self.capital` en lugar de `INITIAL_CAPITAL`; luego eliminar `_below_peak` completamente | supervisor.py:430,2966 | ✅ |
| 2026-06-29 | SL demasiado ancho (USDCAD 112 pips) | ATR(14)×1.5 sin cap → TP inalcanzable en mismo día → SL hit overnight | Cap 40 pips para forex H1 | agents/signal_agent.py:142 | ✅ |
| 2026-06-30 | ACCEL mode nunca activaba | Comparaba balance vs `INITIAL_CAPITAL=$100K` en vez de `self.capital` | Cambiar a `self.capital * 0.98` | supervisor.py:1904 | ✅ |
| 2026-06-30 | Posiciones duplicadas (2 USDCAD, 2 EURUSD) | Lógica "permitir segunda posición si score>=120" | MAX 1 posición por símbolo sin excepciones | supervisor.py:1927 | ✅ |
| 2026-06-30 | Bot re-abría par inmediatamente después de cierre manual | No había cooldown post-cierre (SL, TP, o manual) | Cooldown 2h en cualquier cierre via `_ticket_info` tracking en `_position_monitor_loop` | supervisor.py:2386,2573 | ✅ |
| 2026-06-30 | Bot operaba en horas 17-19 UTC (WR=24%) | Solo tenía multiplicador KZ (0.85x) insuficiente para scores altos (120+) | Bloqueo duro en DEAD_HOURS_UTC | supervisor.py:121 | ✅ |
| 2026-06-30 | EURUSD se reactivaba cada pocas horas con WR=0% | RiskGovernor cooldown expirado, se reactivó automáticamente | Suspensión permanente en risk_governor_state.json | memory/risk_governor_state.json | ✅ |

## Patrón de bugs recurrentes

- **Fix que arregla A y rompe B**: Evitar cambios en supervisor.py sin correr pytest completo
- **Umbrales hardcodeados**: Cualquier constante numérica en supervisor.py debe tener un comentario con el razonamiento
- **Duplicados de posición**: NUNCA permitir "segunda posición por condición especial" — max 1 por símbolo
- **Re-entrada post-cierre**: Siempre registrar cooldown cuando una posición cierra (cualquier razón)
