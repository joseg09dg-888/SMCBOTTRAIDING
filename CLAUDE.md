# SMC TRADING BOT — MEMORIA PERMANENTE PARA CLAUDE CODE

> **LEE ESTE ARCHIVO PRIMERO EN CADA SESIÓN.**
> Es la fuente de verdad del proyecto. Actualizar al final de cada sesión.

---

## 1. QUÉ ES ESTE PROYECTO

Bot de trading algorítmico multi-agente con Python que opera en:
- **Binance Testnet** (crypto) — funcionando ✅
- **MT5 Demo** (forex/indices) — pendiente conexión ⚠️
- **Objetivo**: Pasar challenge FTMO → cuenta fondeada $200K → $9,000/mes

**Usuario:** Jose David | joseg09.dg@gmail.com  
**Repo:** github.com/joseg09dg-888/SMCBOTTRAIDING  
**Stack:** Python 3.12, asyncio, pytest, PM2, Windows 11

---

## 2. CÓMO ARRANCAR EL BOT

```powershell
cd C:\Users\jose-\projects\trading_agent

# Con PM2 (auto-restart 24/7) — PREFERIDO
pm2 start ecosystem.config.js
pm2 status

# Manual
.venv\Scripts\python startup.py --auto --capital 1000

# Forzar trade demo ahora
.venv\Scripts\python scripts/force_demo_trade.py BTCUSDT 1h

# Diagnóstico MT5
.venv\Scripts\python scripts/mt5_full_test.py
```

---

## 3. ESTADO ACTUAL (2026-06-03)

| Componente | Estado | Notas |
|-----------|--------|-------|
| Tests | ✅ 1288/1288 pasando | `pytest tests/ -q` |
| Binance Testnet | ⚠️ DNS falla en PM2 | datos cacheados funcionan |
| Scan crypto | ✅ ACTIVO | BTCUSDT/ETHUSDT/SOLUSDT/BNBUSDT/XRPUSDT/ADAUSDT |
| Scan forex | ✅ MT5 REAL | EURUSD/GBPUSD/XAUUSD/USDJPY/GBPJPY/NAS100/US30 via MT5 |
| MT5 Axi Demo | ✅ CONECTADO | login=10042896 server=Axi-US50-Demo balance=$99,470.20 |
| MT5 ordenes reales | ✅ FUNCIONANDO | Ticket #61754943 USDJPY BUY entry=159.937 TP=162.611 |
| MT5 scan loop | ✅ ACTIVO | _scan_mt5_symbol() H1+H4 para 7 pares forex |
| MT5 auto-reconexion | ✅ | delay 2s en connect() + loop cada 30s |
| Telegram polling | ✅ ACTIVO | parse_mode=HTML, 29 comandos |
| PM2 | ✅ smc-bot ONLINE | auto-restart |
| Windows startup | ✅ | Startup folder + .bat |
| SQLite scores | ✅ ACTIVO | memory/scores.db + outcome/pnl_pct columns |
| SQLite episodic | ✅ ACTIVO | memory/episodes.db (WAL mode) |
| Glint | ✅ headless | cookies en memory/glint_session.json |
| Demo TP/SL monitor | ✅ ACTIVO | _monitor_demo_trades() via yfinance cada ciclo |
| Demo persistencia | ✅ ACTIVO | memory/demo_trades_state.json — sobrevive reinicios |
| H4 trend filter | ✅ ACTIVO | crypto 1h trades bloqueados si van contra H4 trend |
| POI zone filter | ✅ ACTIVO | solo OBs dentro del 5% del precio actual |
| ATR SL | ✅ ACTIVO | SL = ATR(14) × 1.5 (era 0.5% fijo) |
| Trailing SL | ✅ ACTIVO | mueve SL a breakeven cuando profit >= 1×SL |
| AutonomousLearner | ✅ ACTIVO | loop cada 1h — ajusta pesos por setup_type/regime |
| ResearchAgent | ✅ ACTIVO | loop cada 2h — arXiv + MQL5 |
| GoalsManager | ✅ ACTIVO | loop cada 30min — 5 metas autónomas |
| NightlyReporter | ✅ ACTIVO | 22:00 UTC — reporte diario vía Telegram |

---

## 4. CREDENCIALES MT5 XM (obtenidas de Gmail MCP 2026-05-20)

```
MT5_LOGIN, MT5_PASSWORD, MT5_SERVER ← ver .env local (NUNCA pegar valores reales aqui)
Cuenta: Demo XM Global
Email: site@xm.com (email de bienvenida)
```

**Servidor accesible:** `mt5.xmglobal.com:443 → TRUE`

**Fix pendiente (una sola vez):**
1. MT5 está abierto y configurado para XMGlobal-MT5 10
2. En la ventana de MT5 → ingresar el Password de `.env` → OK
3. Esperar que aparezcan cotizaciones (EURUSD etc.)
4. Correr: `.venv\Scripts\python scripts/test_xm_mt5.py`

**NO tocar MT5 con Python hasta que esté completamente logueado en la GUI.**
3. Completar formulario → recibir credenciales por email
4. Actualizar `.env`: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER=ICMarketsSC-Demo
5. Correr: `.venv\Scripts\python scripts/mt5_full_test.py`

**NO intentar fix programático** — mt5.initialize() crashea el terminal cuando no hay cuenta activa.

---

## 5. REGLAS IRROMPIBLES (ENCODING)

### ⚠️ CRÍTICO: Encoding en supervisor.py y telegram_commander.py

1. **NUNCA** usar PowerShell `Out-File`/`Set-Content` para escribir Python con emojis/acentos — añade BOM (U+FEFF) que Python rechaza
2. **SIEMPRE** usar el tool `Write` de Claude Code, o un script Python con `open(..., 'w', encoding='utf-8')`
3. Si supervisor.py tiene `SyntaxError: invalid non-printable character U+FEFF` → correr `scripts/deep_fix_supervisor.py`
4. Si telegram_commander.py tiene smart quotes → correr `scripts/rebuild_commander.py`
5. `analysis_text` en `_run_smc_lite()` debe ser ASCII puro: `"setup valido"` NO `"setup válido"`

---

## 6. ARQUITECTURA COMPLETA

```
trading_agent/
├── core/
│   ├── supervisor.py          ← ORQUESTADOR PRINCIPAL (NO tocar con PowerShell)
│   ├── config.py              ← Variables de entorno
│   ├── risk_manager.py        ← Gestión de riesgo
│   ├── decision_filter.py     ← Score 0-100 → REDUCED/FULL/PREMIUM
│   ├── learning_engine.py     ← Aprendizaje automático
│   ├── agent_memory.py        ← Memoria por agente
│   ├── agent_health_check.py  ← Health check 21 agentes
│   ├── continuous_learning.py ← Aprendizaje 24/7
│   ├── wakeup_recovery.py     ← Recuperación post-apagado
│   └── mode_manager.py        ← AUTO/SEMI/PAUSED/HYBRID
│
├── agents/ (20 agentes -- elliott_agent.py, chaos_agent.py, quant_optimizer.py
│              y quant_intel.py ELIMINADOS 2026-07-26, cero uso en vivo, ver seccion 16)
│   ├── signal_agent.py         ← Genera TradeSignal con entry/SL/TP
│   ├── analysis_agent.py       ← SMCAnalysisAgent (usa Claude API)
│   ├── lunar_agent.py          ← Ciclos lunares → sesgo trading (solo display /lunar, sin scoring)
│   ├── institutional_flow_agent.py
│   ├── alternative_data_agent.py
│   ├── microstructure_agent.py
│   ├── fed_sentiment_agent.py
│   ├── onchain_agent.py
│   ├── geopolitical_agent.py
│   ├── retail_psychology_agent.py
│   ├── energy_frequency_agent.py ← Numerología, tarot, planetas
│   ├── report_agent.py           ← Reportes semanal/mensual
│   ├── screen_vision_agent.py    ← Claude Vision + mirror mode
│   ├── footprint_agent.py        ← Delta, absorción, imbalances
│   ├── statistical_edge_agent.py ← QuantEdgeAgent (7 modulos -- quant_optimizer.py
│   │                                y quant_intel.py ELIMINADOS 2026-07-26, cero uso real)
│   ├── quant_stats.py            ← VaR/CVaR/Kelly/Monte Carlo
│   ├── quant_regime.py           ← regimen de mercado (reglas, no HMM real)
│   ├── quant_factors.py          ← IC/IR factor analysis
│   ├── quant_anomalies.py        ← Calendar effects (funding rate nunca llega, siempre 0)
│   ├── quant_ensemble.py         ← heuristico momentum+MA20 (sklearn nunca se entrena en vivo)
│   ├── quant_flow.py             ← OFI, VPIN, Kyle impact (nunca recibe bid/ask reales, siempre +0)
│   └── quant_stress.py           ← 10 crash scenarios históricos
│
├── smc/
│   ├── structure.py            ← BOS/CHoCH/HH/HL/LH/LL
│   ├── orderblocks.py          ← Order Blocks + FVG
│   ├── volume_profile.py       ← POC/VAH/VAL/VWAP
│   ├── ml_predictor.py         ← LSTM predictor
│   └── sentiment.py            ← Análisis de sentimiento
│
├── strategies/
│   ├── ftmo_agent.py           ← Reglas FTMO 2026 hardcodeadas
│   ├── pairs_trading.py        ← Arbitraje estadístico (IC/IR)
│   └── event_driven.py         ← FOMC, NFP, halving, FED
│
├── connectors/
│   ├── binance_connector.py    ← OHLCV + órdenes Binance
│   ├── metatrader_connector.py ← MT5 OHLCV + órdenes
│   ├── market_connector.py     ← Interfaz unificada Binance+MT5
│   ├── glint_connector.py      ← Señales macro via HTTP
│   └── glint_browser.py        ← Playwright headless para Glint
│
├── dashboard/
│   ├── telegram_commander.py   ← 27 comandos Telegram (HTML mode)
│   ├── telegram_bot.py         ← send_signal_demo, send_glint_alert
│   └── screenshot_engine.py    ← Capturas de pantalla
│
├── backtesting/
│   └── lean_backtest.py        ← Backtest numpy/pandas
│
├── execution/
│   └── smart_execution.py      ← TWAP/VWAP/Iceberg
│
├── deployment/
│   ├── cloud_setup.py          ← Dockerfile/docker-compose/PM2
│   └── health_monitor.py       ← Health check via lock/log
│
├── scripts/
│   ├── force_demo_trade.py     ← Forzar trade demo ahora
│   ├── mt5_full_test.py        ← Diagnóstico MT5
│   ├── deep_fix_supervisor.py  ← Reconstruir supervisor.py limpio
│   ├── rebuild_commander.py    ← Reconstruir telegram_commander.py
│   ├── fix_all_encoding.py     ← Fix smart quotes en .py files
│   └── audit_imports.py        ← Verificar 18 módulos críticos
│
├── startup.py         ← Entry point con process lock
├── ecosystem.config.js ← PM2 auto-restart
├── railway.toml       ← Railway deployment
├── Dockerfile         ← Docker container
└── .env               ← Credenciales (NO subir a GitHub)
```

---

## 7. CREDENCIALES (.env — NUNCA a GitHub)

```
ANTHROPIC_API_KEY      ← Claude API
BINANCE_API_KEY        ← Testnet key
BINANCE_API_SECRET     ← Testnet secret
BINANCE_TESTNET=true
MT5_LOGIN              ← BrokerGroup-Live24, demo sin fondos (NO CONECTA), ver .env
MT5_PASSWORD           ← Pendiente actualizar a ICMarkets, ver .env
MT5_SERVER             ← Pendiente → ICMarketsSC-Demo, ver .env
TELEGRAM_BOT_TOKEN     ← @smc_trading_bot
TELEGRAM_CHAT_ID=5371315570
GLINT_EMAIL=joseg09.dg@gmail.com
GLINT_SESSION_TOKEN    ← Cookie de sesión Glint
OPERATION_MODE=semi
MAX_RISK_PER_TRADE=0.005
```

---

## 8. COMANDOS TELEGRAM (27 activos)

```
/status     → Estado completo crypto + MT5
/auto       → Modo 100% automático
/semi       → Modo semi-auto (pide confirmación)
/pause      → Pausa el bot
/resume     → Reanuda el bot
/positions  → Posiciones abiertas
/close_all  → Cierra todas las posiciones
/scores     → Últimos 10 scores DecisionFilter
/risk       → Estado del riesgo
/train      → Curriculum de entrenamiento
/youtube    → Estado aprendizaje YouTube
/history    → Análisis histórico. Ej: /history BTC
/memory     → Estado memoria agentes
/health     → Health check 21 agentes
/energy     → Lectura energética (numerología/tarot)
/reporte_semanal → Reporte semanal HTML
/reporte_mensual → Reporte mensual HTML
/criterios  → Criterios para cuenta real
/proyeccion → Proyección próxima semana
/vision     → Activa/desactiva screen vision
/screenshot → Captura y analiza pantalla
/mirror     → Modo espejo (aprende de ti)
/analysis   → Análisis SMC completo
/onchain    → Métricas on-chain
/lunar      → Ciclos lunares
/edge       → Statistical edge del sistema
/footprint  → Análisis footprint BTCUSDT
/ftmo       → Estado FTMO challenge
```

---

## 9. SISTEMA DE SCORING (0-100 demo / 0-275 con quant)

```
Base score 0-100:
  SMC técnico:    0-30 pts  (estructura, OB, FVG, BOS)
  ML/LSTM:        0-25 pts  (predicción dirección)
  Sentimiento:    0-20 pts  (Glint, macro)
  Risk/session:   0-25 pts  (RR, sesión, drawdown)
  Histórico:      0-20 pts  (bonus contexto)

Extensión quant (+0-175 pts):
  QuantEdgeAgent: +0-50 pts
  FootprintAgent: +0-25 pts
  AnomalyDetector: ±15 pts
  EnergyFrequency: ±15 pts
  OrderFlow:      ±10 pts

Demo threshold: score >= 35 → ejecutar
Real threshold: score >= 60 → REDUCED (25% risk)
                score >= 75 → FULL (100% risk)
                score >= 90 → PREMIUM (alerta 🔥)
```

---

## 10. FLUJO COMPLETO DE DECISIÓN

```
1. BinanceConnector.get_ohlcv(symbol, tf, 200)
2. _run_smc_lite(df) → analysis_text (ASCII puro)
   ↓ deriva dirección de último BOS/CHoCH si bias=neutral
3. SignalAgent.evaluate(analysis_text, ...) → TradeSignal
   ↓ checks "setup" AND "valid/valido" en analysis_text
4. route_signal(signal, df) → DecisionFilter.evaluate()
   ↓ score 0-100
5. Si score >= DEMO_SCORE_THRESHOLD (35):
   → _execute_demo_trade() → Telegram HTML notification
   → DemoTrade registrado en memoria
6. MT5 scan (cuando disponible):
   → _scan_mt5_symbol() → mismo flujo
7. yfinance forex scan (siempre):
   → _scan_forex_yfinance() → mismo flujo
```

---

## 11. REGLAS FTMO 2026 (hardcodeadas en ftmo_agent.py)

```
2-STEP:
  profit_target: 10% → 5% (fase 2)
  max_daily_loss: 5% del balance inicial (estático)
  max_drawdown: 10% del balance inicial (estático)
  min_days: 4
  profit_split: 80% → 90%

1-STEP:
  profit_target: 10%
  max_daily_loss: 3% (más estricto)
  max_drawdown: 10% TRAILING
  consistency: ningún día > 30% del profit total
  profit_split: 90%

Modo seguridad del bot:
  stop_diario: al 60% del límite diario
  stop_drawdown: al 70% del límite total
  pausa_3_perdedores: 24h tras 3 pérdidas seguidas
  no_operar: lunes 00-02 UTC, viernes 16+ UTC
  no_operar_noticias: ±2 min de NFP/FOMC/CPI
```

---

## 12. OBJETIVO FINAL

```
Demo → win_rate > 60% por 4 semanas
     → profit_factor > 1.5
     → max_drawdown < 5%
     → 100+ trades

FTMO Challenge:
  $10K → pasar → $25K → $50K → $100K → $200K

Con $200K fondead al 90% profit split:
  $200K × 5%/mes × 90% = $9,000/mes para Jose
  $9,000/mes × 12 = $108,000/año
```

---

## 13. HISTORIAL DE TESTS

| Sesión | Tests | Módulos añadidos |
|--------|-------|-----------------|
| 1 | 408 | Core + 12 agentes base |
| 2 | 553 | 6 módulos (energy, report, vision, health, continuous, wakeup) |
| 3 | 823 | 10 módulos quant + StatisticalEdgeAgent + FootprintAgent |
| 4 | 861 | Auditoría completa, 26 comandos Telegram |
| 5 | 975 | Backtesting, TWAP/VWAP, Pairs, Events, Deployment |
| 6 | 1003 | FTMOAgent (28 tests) |
| 7 | 1182 | Modo autónomo 24/7: episodic_db, AutonomousLearner, ResearchAgent, GoalsManager, NightlyReporter, reason_with_context |
| 8 | 1204 | VolumeCalculator: riesgo dinamico por etapa Axi Select, /proyeccion Telegram, volumen demo 0.10 lots |
| 9 | 1214 | 13 agentes institucionales activados en pipeline: Lunar, Elliott, Chaos, QuantEdge, Footprint, InstFlow, Microstructure, FED, OnChain, Geopolitical, RetailPsych, AltData, Energy |
| 10 | 1214 | _enrich_with_agents() parallelizado con ThreadPoolExecutor — 13 agentes simultaneos (~13x speedup). Thresholds: MT5_REAL=75, MIN_RR=2.0, MAX_OPEN=2, CONSERVATIVE_MODE=False, H1+H4, auto-reduce a 70 tras 2h idle |
| 11 | 1277 | 6 bugs críticos corregidos: crash loop, enrichment gate, demo skip log, threshold cap, stale H4 entry, Claude API auto-confirm |
| 12 | 1288 | H4 trend filter crypto, ATR SL, POI proximity filter, demo TP/SL monitor, demo persistence, real outcome tracking, /demo /performance commands, trailing SL MT5 |

---

## 14. REGLAS PARA CLAUDE CODE EN ESTE PROYECTO

0. **NUNCA añadir más agentes sin evidencia de backtest real** que respalde que aportan edge (instrucción explícita del usuario 2026-08-28, ver sección 16 sobre agentes ya eliminados por no tener evidencia)
0b. **SIEMPRE actualizar `SESION_ACTUAL.md` al terminar** cada respuesta, con el resultado real (no una promesa de lo que se iba a hacer)
0c. **Si algo no funciona, decirlo explícitamente** — nunca reportar como completo/funcionando algo que no se verificó de verdad
1. **LEER ESTE ARCHIVO** al inicio de cada sesión
2. **NUNCA romper** los 1288 tests existentes — verificar con `pytest tests/ -q`
3. **SIEMPRE verificar** antes de marcar completo (skill: verification-before-completion)
4. **ENCODING**: usar Write tool, NUNCA PowerShell Out-File para archivos .py con emojis
5. **ENCODING**: si supervisor.py falla → correr `scripts/deep_fix_supervisor.py`
6. **ENCODING**: si telegram_commander.py falla → correr `scripts/rebuild_commander.py`
7. **MT5**: NO intentar fix programático — Python crashea el terminal. Solo fix manual por usuario
8. **SIEMPRE** conectar comandos Telegram a datos reales (no hardcoded)
9. **SIEMPRE** hacer TDD: tests primero, luego implementación
10. **SIEMPRE** usar `scripts/audit_imports.py` para verificar módulos antes de deploy
11. **SKILL**: usar `trading-bot-tracker` al inicio de sesión para contexto completo
12. **SKILLS disponibles**: ver `~/.claude/skills/` — 33 skills instalados

---

## 15. PROBLEMAS CONOCIDOS ACTUALES

| Problema | Causa | Fix |
|---------|-------|-----|
| Binance DNS falla | testnet.binance.vision no resuelve en PM2 | usa datos cacheados, opera normal |
| supervisor.py sensible a encoding | PowerShell añade BOM/smart quotes | Siempre usar Write tool o deep_fix_supervisor.py |
| SMCBotEA no en charts | Acción manual requerida | Usuario debe arrastrar SMCBotEA a charts en MT5 GUI |
| FED "sin texto analizado" | FED agent necesita texto FOMC para analizar | Normal — retorna neutral=0 cuando no hay texto |
| USDJPY BUY en drawdown | Entrada 159.937, mercado oscilando | TP 162.611 (+267 pips), SL 159.407, trailing SL activo |

## 16. AUDITORÍA DE AGENTES (2026-05-25)

**Activos en el loop principal:**
- SignalAgent, DecisionFilter, BinanceConnector, MT5Connector, GlintBrowser
- TelegramCommander, TradingTelegramBot, RiskManager
- MarketStructure, OrderBlockDetector, FVGDetector (via _run_smc_lite)
- HistoricalDataAgent (solo via /history)
- AutonomousLearner, ResearchAgent, GoalsManager, NightlyReporter (loops nuevos)

**Activos en enrichment pipeline (_enrich_with_agents):**
- QuantEdgeAgent, FootprintAgent, InstitutionalFlowAgent, MarketMicrostructureAgent
- FEDSentimentAgent, OnChainAgent, GeopoliticalAgent
- RetailPsychologyAgent, AlternativeDataAgent

**Dormant (existen pero NO conectados):**
- SMCAnalysisAgent (Claude API — pendiente conectar para confirmacion pre-orden real)
- ReportAgent, ScreenVisionAgent, LearningEngine, AgentMemory

**ACTUALIZACIÓN 2026-07-26 — eliminación de código muerto (instrucción explícita del usuario: "lo que se ha probado que es irrelevante [lunar y elliott] eso debería eliminarse del código porque eso se demostró que no funciona"):**
- `agents/elliott_agent.py` y `agents/chaos_agent.py` **ELIMINADOS por completo** (archivos borrados junto con sus tests). Confirmado por grep exhaustivo: nunca tuvieron score real (siempre `return 0` hardcodeado en supervisor.py desde antes de esta sesión) NI un comando Telegram funcional (`/elliott` era un texto fijo que jamás llamaba al agente real — también eliminado).
- `LunarCycleAgent`/`EnergyFrequencyAgent` se **mantienen** — a diferencia de Elliott/Chaos, estos SÍ se usan de verdad en vivo vía `/lunar` y `/energy` (Telegram instancia su propia copia para mostrar el dato). Solo su contribución al SCORE de trading fue desactivada antes (sin evidencia de edge estadístico), no su funcionalidad de display.
- `core/supervisor.py`: removidas las instancias muertas `self._lunar`, `self._elliott`, `self._chaos`, `self._energy` (se creaban en `__init__` pero nunca se leían en ningún otro lado del archivo) y sus imports.
- Agentes activos en enrichment pipeline: 9 (antes 13 — bajó de 13 a 9 al eliminar Lunar/Elliott/Chaos/Energy, que igual sumaban 0 puntos siempre).

**ACTUALIZACIÓN 2026-07-26 (2da ronda) — panel de 5 agentes expertos en paralelo auditando el motor SMC técnico (structure.py/orderblocks.py/volume_profile.py/ml_predictor.py) y el suite quant completo (9 módulos), instrucción del usuario "no pares ni preguntes hasta terminar de arreglar todo":**
- **CRÍTICO — `smc/structure.py`**: `detect_bos()`/`detect_choch()` devolvían eventos ordenados por cuándo se FORMÓ el swing que los origina, no por cuándo se CONFIRMARON (`confirmed_at`). Un HH formado en la vela 20 que tarda hasta la 190 en romperse quedaba antes en la lista que un LL formado en la 150 y roto en la 155 — `bos_list[-1]` (usado en `core/supervisor.py` para decidir LONG/SHORT cuando el bias es neutral) podía devolver el evento viejo en vez del real más reciente. Arreglado: ambas listas se ordenan por `confirmed_at` antes de devolverse. Test de regresión con caso adversarial real en `tests/smc/test_structure.py`.
- **`smc/orderblocks.py`**: los Order Blocks nunca se invalidaban aunque el precio ya hubiera cerrado más allá de la zona — podían anclar el entry/OTE de un trade nuevo sobre una zona ya rota. Se agregó el campo `mitigated` (bool) a cada OB, calculado real contra cierres posteriores; `core/supervisor.py` ahora filtra OBs mitigados antes de elegir el POI. También se corrigió un docstring falso que afirmaba `atr_mult=1.5` cuando el default real siempre fue `1.0` desde que se introdujo (verificado con git blame) — NO se cambió el valor real sin backtest primero (ver `backtest_before_anecdote_fixes`), solo se corrigió la documentación falsa y se dejó como tarea abierta.
- **`smc/ml_predictor.py`**: el bonus de +15pts por "direction match" comparaba la dirección del propio predictor (derivada de momentum/trend/htf de la MISMA vela) contra el `bias` que YA viene del sesgo estructural SMC calculado sobre esa misma vela — el bot se premiaba a sí mismo por estar de acuerdo consigo mismo, no una confirmación independiente de "ML". Desactivado (score máximo bajó de 25 a 10); se mantiene el bonus de confianza (agreement interno de sus propias features), que sí varía con datos reales.
- **`smc/volume_profile.py`**: código 100% muerto (POC/VAH/VAL/VWAP matemáticamente correctos, pero `SMCAnalysisAgent` que los usa nunca se llama en el loop de escaneo; `/analysis` en Telegram es un texto fijo). No se tocó (matemática correcta, solo dormant, igual que ReportAgent/ScreenVisionAgent en la lista de dormant de arriba).
- **`agents/quant_stress.py` / `statistical_edge_agent.py`**: el stress test contra 10 crashes históricos SIEMPRE "pasaba" (`stress_passed=True` garantizado) porque `open_positions` nunca se pasaba, y sin eso la fórmula de pérdida cancela matemáticamente el `equity`. Telegram mostraba "Stress Test: ✅ OK" como si fuera un chequeo real en cada trade. Arreglado: `metatrader_connector.get_positions()` ahora incluye `contract_size` real (vía `mt5.symbol_info`), y `core/supervisor.py`/`statistical_edge_agent.py` calculan el notional real (`volume*contract_size*price_open`) de las posiciones abiertas y lo pasan al stress test.
- **`agents/quant_factors.py`**: `factor_ir` era 0.0 garantizado siempre — `calculate_ir()` requiere una serie temporal de ≥2 ICs pero se le pasaba un solo IC envuelto en lista de 1 elemento. El bonus +8/+4 del score nunca podía activarse sin importar la fuerza real de la señal. Arreglado con una serie de IC por ventana rodante (20 velas) real.
- **`agents/quant_optimizer.py`** (Bayesian/Optuna): **ELIMINADO por completo** — nunca se importaba en `statistical_edge_agent.py`, solo existía en un comentario de docstring ("SP7"). Cero uso real, mismo criterio que Elliott/Chaos.
- **`agents/quant_intel.py`** (papers académicos/insider activity): **ELIMINADO por completo** — ya se sabía que `get_consensus_bias()`/`get_insider_activity()` eran `hash(symbol)%100` disfrazado (desactivado 2026-07-14), y esta auditoría confirmó que ni siquiera `calculate_collective_score()` se llama desde ningún lado. Mismo criterio que Elliott/Chaos.
- **`agents/quant_flow.py`** (OFI/VPIN/Kyle impact): confirmado 100% inalcanzable en vivo — `bid_volumes`/`ask_volumes` nunca se pasan desde ningún conector (el bot solo tiene datos OHLCV, no tick/L2 real). No se tocó — inventar un bid/ask falso desde velas sería peor que no tener el dato. Contribuye 0 siempre, sin riesgo.
- **`agents/quant_ensemble.py`**: el "ML ensemble (sklearn)" nunca se entrena en vivo (`.fit()` solo se llama en tests) — cae siempre al heurístico de respaldo (momentum + posición vs MA20). No corrompe el score (el heurístico sí varía con datos reales), pero está mal etiquetado como "ML". No se tocó (requiere decisión de diseño: cargar modelos ya entrenados por `training/run_training.py` que hoy se guardan en `.pkl` y nunca se leen de vuelta — tarea abierta).
- **`agents/quant_regime.py`**: no usa HMM real pese a llamarse "RegimeDetector (HMM)" — es un clasificador de reglas simple sobre std/mean de precios reales. No corrompe el score (sí varía con datos reales), solo mal etiquetado. No se tocó.
- Todos los fixes verificados con tests nuevos de regresión + suite completa pasando antes de deploy.

---

## 17. FIXES APLICADOS EN SESIÓN 2026-06-26 (CRITICAL)

| Fix | Archivo | Descripción |
|-----|---------|-------------|
| Capital fallback | `startup.py:94` | Era $10K → ahora $97K (evita NAS100 0.01L) |
| NAS100.fs VolumeCalculator | `core/volume_calculator.py` | `_norm()` strips `.fs` suffix — ahora 1.0L correcto |
| H4 structural cache | `core/supervisor.py:1249` | Actualiza `_mt5_h4_direction` desde SMC bias ANTES del momentum filter |
| H4 scan loop | `core/supervisor.py:3456` | Preserva LONG/SHORT previo si signal devuelve WAIT (pullback temporal) |
| Skip 0.0 volume | `core/supervisor.py:1820` | Si `calculate_volume` retorna 0.0 → skip trade |
| Skip min-vol swing | `core/supervisor.py:1858` | Si swing vol < 0.11L → skip (evita que monitor cierre como scalp) |
| H1 con H4=WAIT | `core/supervisor.py:3517` | Score >= 115 permite trade D1+H1 sin H4 |
| Recovery mode | `core/supervisor.py:1692,2718` | Eliminado `balance < $100K` como trigger de recovery |
| Portfolio loss filter | `core/supervisor.py:1755` | Cambiado 1% → 2.5% ($2,425 con $97K) |
| Research 24h cooldown | `core/research_agent.py:171` | `_credit_fail_ts` evita spam de error de crédito API |

## 18. DIAGNÓSTICO LUNES

```powershell
# Ejecutar antes del lunes para confirmar que todo está OK
.venv\Scripts\python scripts/monday_ready.py
```

---

## 19. ANÁLISIS 8 DIMENSIONES — BACKTEST 2 AÑOS (2026-06-26)

### Resultados backtest 700 días H1, 6 pares:
```
P(día >= $250):  52%  (con dead_hours incluye 13:00 UTC)
E[día]:          $232
E[mes]:          $5,148 (5.3% sobre $97K — pasa Axi Select)
P(pass Axi 5%): 49%
Sharpe mensual:  0.49
```

### DIM 4 — Sesiones UTC (DATO MÁS CRÍTICO):
```
CORRECCIÓN 2026-07-26: la tabla de abajo (backtest 2 años, 700 días) tenía la
hora 14 y la hora 15 AL REVÉS. El análisis de 16 años reales (commit ae6bdf7,
"hour-14 UTC killzone multiplier was backwards") encontró lo contrario: 14:00
UTC es en realidad la PEOR hora activa (empatada con la 13:00 que sí está
bloqueada), y 15:00 UTC es la MEJOR. core/session_manager.py._HOUR_MULT ya
tiene los números correctos -- esta sección de abajo queda como referencia
histórica de qué tan mal puede fallar un backtest corto, NO como fuente de
verdad. Usar siempre los valores de session_manager.py.

Ranking real 16 años (hora UTC : WR : avg$):
15:00 UTC: WR=57%, avg=+$149 ← MEJOR hora activa
20:00 UTC: WR=50%, avg=+$89
21:00 UTC: WR=51%, avg=+$43
22:00 UTC: WR=50%, avg=+$44
23:00 UTC: WR=50%, avg=+$40
16:00 UTC: WR=46%, avg=+$80
14:00 UTC: WR=29-36%, avg=-$35  ← BLOQUEADA 2026-07-26 (ver abajo)
13:00 UTC: WR=29%, avg=-$97  ← BLOQUEADO (señales rancias post-overnight)
17-19 UTC: WR=24-28%         ← BLOQUEADO
```

**Hora 14 UTC bloqueada por completo (2026-07-26)**: se corrió el backtest
explícito que quedaba pendiente (`scripts/backtest_multiyear.py`,
`EXTRA_DEAD_HOURS=14`, 16 años reales, 100K sims Monte Carlo) comparando
"14 abierta" vs "14 bloqueada":
```
                  14 abierta          14 bloqueada
P(día>=$250):     41%                 40%
E[mes]:           $6,624 (6.8%)       $7,006 (7.2%)
P(pasar Axi 5%):  57%                 58%
Sharpe mensual:   0.86                0.88
```
Mejora real en las 3 métricas que importan, con solo -1pp en P(día>=$250)
(ruido). Además la hora 14 generaba 7,877 trades — más del doble que
cualquier otra hora activa — con WR=36% y avg=-$35/trade: la peor
combinación posible (mucho volumen, mal resultado). `core/supervisor.py`
`DEAD_HOURS_UTC` ahora incluye 14. Horas activas reales: **15, 16, 20, 21,
22, 23 UTC** únicamente.
```

### DIM 8 — Correlaciones (2 años reales):
```
AUDUSD+NZDUSD: r=+0.90 → bloquear 2 simultáneos misma dirección
EURUSD+GBPUSD: r=+0.79 → bloquear 2 simultáneos misma dirección
NAS100: r≈0.00 → siempre independiente, no cuenta para DIM8
```

### Fixes de esta sesión:
| Fix | Archivo | Impacto |
|-----|---------|---------|
| DEAD_HOURS incluye hora 13 | `supervisor.py:113` | +13pp P(≥$250) |
| NZDUSD → MT5_SYMBOLS | `supervisor.py:134` | +1 par (+20% señales) |
| Partial TP + BE simultáneo | `supervisor.py:2298` | -35% varianza |
| 8D DIM8 correlation guard | `supervisor.py:1440` | bloquea riesgo duplicado |
| EightDimensionAgent NUEVO | `agents/eight_dim_agent.py` | análisis permanente |
| Backtest multianual NUEVO | `scripts/backtest_multiyear.py` | 700d H1+10y D1 |
| Skill 8D permanente | `~/.claude/skills/8d-market-analyzer.md` | invocable siempre |

### Comandos de análisis 8D:
```powershell
# Quick read (si backtest_results.json existe):
.venv\Scripts\python -c "import json; r=json.load(open('memory/backtest_results.json')); print(r['stats'])"

# Backtest completo (~5 min):
.venv\Scripts\python scripts/backtest_multiyear.py

# Análisis quantum 1 día (~2 min):
.venv\Scripts\python scripts/backtest_quantum.py
```

## 20. FLUJO OPERATIVO — cómo iterar sin pisarse (2026-08-28)

> Pedido explícito del usuario tras una sesión con fricción real: 5 intentos de
> backtest por desalineaciones de config entre `scripts/backtest_multiyear.py`
> y el bot en vivo, y RAM crítica (3.83GB) por correr cosas pesadas en paralelo
> sin coordinarlas. Esto es el procedimiento fijo para no repetirlo.

### Regla de oro: nunca 2 procesos pesados a la vez en esta PC (3.83GB RAM)
Antes de arrancar CUALQUIER cosa pesada (backtest completo, `pytest` completo,
el bot en vivo con PM2), verificar RAM libre (`Get-CimInstance
Win32_OperatingSystem`). Si hay <1GB libre, esperar a que termine lo que esté
corriendo antes de arrancar otra cosa. Nunca backtest + bot en vivo simultáneo.

### Flujo para cualquier cambio de parámetro/estrategia
```
1. VERIFICAR PARIDAD antes de tocar nada
   -> ¿El script de backtest (scripts/backtest_multiyear.py) tiene los mismos
      defaults que el bot en vivo (core/supervisor.py, core/config.py,
      core/position_guards.py)? Si no se verificó recientemente, grep ambos
      lados para las constantes relevantes (DEAD_HOURS_UTC, PEAK_GUARD_MIN,
      STAGNANT_HOURS, MIN_RR, MAX_OPEN_POSITIONS, REQUIRE_D1/H4) ANTES de
      correr un backtest de 16 años que puede tardar horas con la config
      equivocada.

2. HACER el cambio de código (pequeño, uno a la vez, no varios sin probar)

3. VERIFICAR SINTAXIS barato (ast.parse, no requiere RAM) antes de gastar
   horas en un backtest sobre código roto.

4. SI el cambio toca lógica de sizing/órdenes reales (dinero real en juego):
   correr `pytest tests/ -q` completo ANTES de dar el cambio por bueno --
   solo si hay RAM libre para eso (no en paralelo con un backtest).

5. BACKTEST de validación:
   - Lanzar SIEMPRE con PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 -u, en
     background, redirigiendo a un log real (no confiar en buffer -- los
     runs bufferizados se quedan en blanco horas aunque estén trabajando).
   - Capturar el exit code real (`; echo "EXIT_CODE=$?" >> log`) -- un pipe a
     `tee` sin `pipefail` puede reportar exit 0 aunque el script haya
     crasheado.
   - Monitorear por CPU time (`Get-Process`), no solo por si el archivo de
     log tiene contenido nuevo -- fases pesadas (Dimensiones 1-3, ~16 años ×
     6 pares) no imprimen nada hasta terminar.

6. SI CRASHEA: leer el traceback completo, arreglar la causa raíz (no
   silenciar), y SOLO ENTONCES relanzar -- no reintentar a ciegas.

7. GUARDAR resultado en memory/backtest_results.json (ya trackeado en git,
   se respalda solo via la tarea AutoCommit-Proyectos cada 30 min) Y en
   SESION_ACTUAL.md (hallazgos + qué falta), en el mismo turno en que se
   obtienen los números -- no dejarlo para "después".

8. SOLO CON el resultado del backtest confirmando mejora (o al menos no
   empeoramiento) real: decidir si el cambio se queda o se revierte.
```

### Uso de subagentes de Claude Code (investigación/auditoría, NO trading agents)
Correrlos EN PARALELO al backtest (no consumen la RAM de MT5/Python de forma
significativa, son llamadas a la API de Claude) para no perder tiempo de
reloj. Cada uno con un scope acotado y un entregable concreto (auditoría de
un archivo, investigación de un tema puntual con fuentes citadas) -- nunca
"optimiza todo" sin límite. Sus hallazgos van a `SESION_ACTUAL.md` en cuanto
llegan, no se pierden en el chat.

**Paso explícito, no solo genérico**: en cada ronda de optimización, ANTES de
decidir qué parámetro tocar, lanzar un subagente de investigación web sobre
traders/fondos/papers reales con datos verificables (no marketing) en la
etapa de capital relevante (Axi Select: $5K-$1M) o sobre la técnica puntual
en cuestión (ej. gestión de salidas, sizing, reglas de consistencia) --
igual que el hecho en esta sesión (2026-08-28, ver SESION_ACTUAL.md sección
8: hallazgo real con 2 papers académicos citados sobre salida dinámica vs
R:R teórico fijo). Ese hallazgo alimenta qué hipótesis probar en el
siguiente backtest -- no se cambia código a ciegas, se cambia con evidencia
externa + evidencia propia (backtest) combinadas.

### Regla dura, no negociable
Nunca agregar un agente nuevo al pipeline del bot (`agents/*.py` conectado en
`core/supervisor.py`) sin evidencia de backtest real que lo respalde -- ver
sección 16, ya hay historial de agentes agregados sin evidencia que resultaron
en cero señal real (Elliott, Chaos, quant_optimizer, quant_intel).

---

*Última actualización: 2026-08-28 (sesión de recuperación post-daño de PC:
backup crítico corregido, 4 subagentes de auditoría, 2 bugs de backtest
corregidos -- encoding y filtros desalineados con vivo --, fix de Kelly,
cambio de riesgo 1%->0.5% aplicado, flujo operativo fijado) | Bot: OFFLINE
(pendiente arrancar en PM2 tras liberar RAM) | Dead hours forex: 0-14, 17-19
UTC (15,16,20-23 abiertas) | Backtest 5to intento con config corregida en
curso al cierre de esta sesión, ver SESION_ACTUAL.md para el resultado real*
