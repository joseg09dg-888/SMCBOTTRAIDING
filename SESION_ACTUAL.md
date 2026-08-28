# SESIÓN ACTUAL — memoria de continuidad

> Este archivo se actualiza al final de cada respuesta de Claude Code en este proyecto.
> Si el PC se apaga o se cierra la sesión, la próxima sesión debe leer esto primero
> (además de CLAUDE.md y BUGS_HISTORIAL.md) para retomar exactamente donde quedó.

---

## Última instrucción textual del usuario

> "Lee CLAUDE.md completo antes de cualquier acción. CONTEXTO: El PC se dañó y se
> reinstalaron todos los programas. El código está intacto en GitHub... [checklist de
> 9 puntos: recuperar estado, crear SESION_ACTUAL.md, verificar instalación, revisar
> backtesting completo, revisar los 22 agentes, revisar rutas/conexiones, historial de
> bugs, estrategia Axi Select, verificación final]"

## Fecha de esta sesión
2026-08-28

## Reglas fijadas por el usuario (2026-08-28) — ver sección 14 de CLAUDE.md
1. Nunca añadir más agentes sin evidencia de backtest real.
2. Siempre actualizar este archivo (SESION_ACTUAL.md) al terminar.
3. Si algo no funciona, decirlo explícitamente (no reportar éxito sin verificar).

## Próximo paso pendiente
Verificar el export MT5 con datos frescos: hoy solo hay 5 pares exportados
(EURAUD, EURUSD, NZDUSD, USDCAD, USDCHF, H1+D1). Faltan confirmar/exportar el
resto de pares activos según CLAUDE.md sección 19 (GBPCAD) y decidir si se
necesitan USDJPY/GBPJPY/XAUUSD/US30 pese a estar suspendidos por RiskGovernor.
Luego correr `scripts/backtest_multiyear.py` sobre el set completo antes de
tocar la config de agentes.

## Bugs activos conocidos
Ver BUGS_HISTORIAL.md (7 documentados, todos verificados como siguen arreglados por
grep de spot-check 2026-08-28). Nota: hay ~100+ commits `fix:` en git log posteriores
al último bug documentado (2026-06-30) que NO están individualmente catalogados ahí —
BUGS_HISTORIAL.md es un resumen curado de clases de bug, no exhaustivo del git log.

## Último estado del bot (verificado 2026-08-28, no supuesto)
- Repo: `git status` limpio salvo cambios locales sin commitear (ecosystem.config.js,
  requirements.txt, start_bot.bat modificados; scripts de export MT5 nuevos sin trackear).
  Rama al día con `origin/main` — **nada nuevo se subió a GitHub esta sesión** más allá
  de un fix de seguridad (credenciales MT5 fuera de CLAUDE.md, a .env) y un .gitignore
  para archivos runtime .lock/.bak.
- `data/historical/`: 10 archivos CSV (5 pares × H1/D1), datos reales descargados vía
  MT5 (ej. EURAUD_D1.csv arranca 2004-06-16, 5928 filas) — **untracked, no subido a
  GitHub**, solo local.
- No se añadió ningún agente nuevo esta sesión. No hay backtest nuevo corrido todavía
  sobre los datos recién exportados — pendiente (ver "Próximo paso pendiente").
- Bot NO se verificó corriendo en este chequeo (no se consultó PM2/proceso vivo) — decirlo
  explícito en vez de asumir: **estado del proceso en vivo desconocido, falta confirmar**.
