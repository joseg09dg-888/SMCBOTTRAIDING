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
> bugs, estrategia Axi Select, verificación final]" + luego pidió backup completo y
> arrancar el bot en vivo cuanto antes ("cada segundo que no funciona es plata perdida").

## Fecha de esta sesión
2026-08-28

## Reglas fijadas por el usuario (2026-08-28) — ver sección 14 de CLAUDE.md
1. Nunca añadir más agentes sin evidencia de backtest real.
2. Siempre actualizar este archivo (SESION_ACTUAL.md) al terminar.
3. Si algo no funciona, decirlo explícitamente (no reportar éxito sin verificar).

## Próximo paso pendiente
1. Backtest relanzado con fix de encoding (ver abajo) — confirmar que termina las
   8 dimensiones + Monte Carlo esta vez, y decidir sobre la config de agentes SOLO
   con esos resultados reales (regla 1).
2. Arrancar el bot en vivo (PM2) — NO se ha hecho todavía. Ver "hallazgos" abajo.
3. Verificar conexión Binance (solo se confirmó MT5, Binance no se probó esta sesión).
4. Correr `pytest tests/ -q` completo — NO se corrió esta sesión por RAM crítica
   (ver hallazgos). Correr cuando la RAM se libere (tras terminar el backtest).

## HALLAZGOS CRÍTICOS de esta sesión (verificados, no supuestos)

### 1. Backup incompleto — YA CORREGIDO Y SUBIDO A GITHUB
`.gitignore` tenía `*.db` global, así que `memory/episodes.db`, `memory/scores.db` y
`memory/historical_data.db` (el historial real de trades/scores/precios) NUNCA se
subieron a GitHub pese a instrucciones previas de "subir todo". Se agregaron
excepciones explícitas en `.gitignore` y se subieron los 3 archivos + los 6 pares de
históricos MT5 nuevos (`data/historical/`, H1+D1, hasta 19 años) vía la tarea
`AutoCommit-Proyectos` (corre cada 30 min, `C:\Users\JOSÉ\Projects\auto-commit.ps1`).
Confirmado con `git push` → "Everything up-to-date". **Consecuencia irreversible**:
las tablas `episodes`/`scores` en esta PC están en 0 filas — el detalle de trades
reales de antes del daño (última actividad registrada 2026-07-17/21 en
`daily_trades.json`/`axi_select_state.json`) se perdió, no había backup fuera de
GitHub. A futuro esto ya no puede volver a pasar (backup automático cada 30 min ya
cubre los 3 archivos).

### 2. Bot NO está corriendo en vivo
`pm2 status`/`pm2 list` no muestra ningún proceso — ni "smc-bot" ni nada. Tampoco
existe la tarea de Windows "SMC-TradingBot" en esta PC nueva (`setup_autostart.ps1`
apunta a la ruta vieja `C:\Users\jose-\projects\trading_agent`, nunca se re-registró
tras el cambio de PC). `ecosystem.config.js` SÍ tiene ya las rutas correctas de esta
PC. No se arrancó todavía porque la RAM estaba crítica (ver punto 3) mientras corría
el backtest — arrancar el bot completo (20+ agentes, Telegram, ML) encima de eso
arriesgaba crashear todo. Arrancar apenas la RAM se libere.

### 3. RAM crítica en esta PC
Solo **3.83 GB de RAM total**, y con MT5 + backtest corriendo llegó a quedar
**0.2 GB libres (94.9% usado)**. Es la causa más probable de que el backtest tardara
horas en vez de los ~5 min documentados en el propio script (posible swap/thrashing).
Tenerlo en cuenta para cualquier plan de correr varias cosas pesadas a la vez en
esta máquina.

### 4. `scripts/audit_imports.py` estaba roto — YA CORREGIDO
Tenía hardcodeado el path de la PC vieja (`C:\Users\jose-\projects\trading_agent`),
fallaba con `FileNotFoundError` antes de importar nada. Se cambió a usar el path del
propio repo dinámicamente. Resultado tras el fix: **18/18 módulos críticos OK**.

### 5. Conteo de agentes desactualizado en CLAUDE.md ("22 agentes")
Verificado contra el código real (no contra la doc): **30 archivos** en `agents/`,
**29 referenciados de verdad** en `core/supervisor.py` y/o
`dashboard/telegram_commander.py`. **1 completamente muerto/huérfano**:
`mql5_reader.py` — su docstring dice "corre cada 6h en background" pero ningún
archivo del proyecto lo importa. No afecta nada (no está conectado), pero conviene
borrarlo o conectarlo, no dejarlo como falso positivo de funcionalidad.
El "22" quedó desactualizado porque después de la ablación documentada en sección 16
(elliott/chaos/quant_optimizer/quant_intel eliminados, -4) se agregaron 7 agentes
nuevos no documentados: `axi_capital_adjuster`, `axi_select_guard`,
`axi_select_tracker`, `axi_vision_agent`, `consistency_enforcer`, `eight_dim_agent`,
y el uso de `portfolio_tracker`/`report_agent` vía Telegram. Estos 6 "axi_*" +
consistency_enforcer son el sistema de guardias de Axi Select (día a día: P&L límite
diario -4%, tracking mensual vs meta 5%, detección de nuevo capital asignado, regla
anti-"día de suerte" >30% del mes). CLAUDE.md sección 6/16 debería actualizarse con
esta lista, pendiente.

### 6. Backtest crasheó en el primer intento — YA CORREGIDO, RELANZADO
`scripts/backtest_multiyear.py` línea 632 imprime emojis (🔥✅⚠️❌) en la tabla de
Dimensión 4. La consola de esta PC usa cp1252 por defecto → `UnicodeEncodeError`,
crash antes de llegar a Dimensiones 5-8 y al Monte Carlo (las métricas que más
importan: P(pasar Axi 5%), E[mes], Sharpe). Sí alcanzó a completar Dimensiones 1-3
antes de morir (~3.5h de reloj en esta PC, dato real): 34,114 trades simulados sobre
~16 años H1 reales, 14,419 cierres "final", WR=17.6% sobre finals (pero eso es
engañoso — ver desglose de tipos de cierre abajo, la mayoría de cierres NO son
TP/SL sino guardias). Relanzado con `PYTHONIOENCODING=utf-8` + salida sin buffer
para verlo en vivo — en progreso, ver estado real en el próximo mensaje de Claude.

## Bugs activos conocidos
Ver BUGS_HISTORIAL.md (7 documentados, todos verificados como siguen arreglados por
grep de spot-check 2026-08-28). Nota: hay ~100+ commits `fix:` en git log posteriores
al último bug documentado (2026-06-30) que NO están individualmente catalogados ahí —
BUGS_HISTORIAL.md es un resumen curado de clases de bug, no exhaustivo del git log.

## Último estado verificado (2026-08-28)
- Repo: todo commiteado y pusheado a `origin/main` (incluye el fix de backup).
- Instalación: venv OK (Python 3.14.7), `pip check` sin problemas, 18/18 módulos
  críticos importan bien.
- MT5: conecta bien (login=10042896, server=Axi-US50-Demo), export de históricos
  funcionó para los 6 pares activos.
- Binance: NO verificado esta sesión.
- PM2/bot en vivo: NO está corriendo (ver hallazgo 2).
- pytest completo: NO corrido esta sesión (RAM crítica, ver hallazgo 3).
- Backtest: primer intento crasheó (hallazgo 6), relanzado con fix, resultado
  pendiente de confirmar al cierre de esta sesión.
