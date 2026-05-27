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

## 3. ESTADO ACTUAL (2026-05-25)

| Componente | Estado | Notas |
|-----------|--------|-------|
| Tests | ✅ 1204/1204 pasando | `pytest tests/ -q` |
| Binance Testnet | ⚠️ DNS falla en PM2 | datos cacheados funcionan |
| Scan crypto | ✅ ACTIVO | BTCUSDT/ETHUSDT/SOLUSDT/BNBUSDT/XRPUSDT/ADAUSDT |
| Scan forex | ✅ MT5 REAL | EURUSD/GBPUSD/XAUUSD/USDJPY/GBPJPY/NAS100/US30 via MT5 |
| MT5 Axi Demo | ✅ CONECTADO | login=10042896 server=Axi-US50-Demo balance=$99,955.37 |
| MT5 ordenes reales | ✅ FUNCIONANDO | Ticket #59708384 USDJPY BUY ejecutado 2026-05-25 |
| MT5 scan loop | ✅ ACTIVO | _scan_mt5_symbol() H1+H4 para 7 pares forex |
| MT5 auto-reconexion | ✅ | delay 2s en connect() + loop cada 30s |
| Telegram polling | ✅ ACTIVO | parse_mode=HTML, 27 comandos |
| PM2 | ✅ smc-bot ONLINE | auto-restart |
| Windows startup | ✅ | Startup folder + .bat |
| SQLite scores | ✅ ACTIVO | memory/scores.db |
| SQLite episodic | ✅ ACTIVO | memory/episodes.db (WAL mode) |
| Glint | ✅ headless | cookies en memory/glint_session.json |
| AutonomousLearner | ✅ ACTIVO | loop cada 1h — ajusta pesos por setup_type/regime |
| ResearchAgent | ✅ ACTIVO | loop cada 2h — arXiv + MQL5 |
| GoalsManager | ✅ ACTIVO | loop cada 30min — 5 metas autónomas |
| NightlyReporter | ✅ ACTIVO | 22:00 UTC — reporte diario vía Telegram |

---

## 4. CREDENCIALES MT5 XM (obtenidas de Gmail MCP 2026-05-20)

```
MT5_LOGIN=345308080
MT5_PASSWORD=IMSMCbot*2
MT5_SERVER=XMGlobal-MT5 10
Cuenta: Demo XM Global
Email: site@xm.com (email de bienvenida)
```

**Servidor accesible:** `mt5.xmglobal.com:443 → TRUE`

**Fix pendiente (una sola vez):**
1. MT5 está abierto y configurado para XMGlobal-MT5 10
2. En la ventana de MT5 → ingresar Password: IMSMCbot*2 → OK
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
├── agents/ (24 agentes)
│   ├── signal_agent.py         ← Genera TradeSignal con entry/SL/TP
│   ├── analysis_agent.py       ← SMCAnalysisAgent (usa Claude API)
│   ├── lunar_agent.py          ← Ciclos lunares → sesgo trading
│   ├── elliott_agent.py        ← Ondas de Elliott
│   ├── institutional_flow_agent.py
│   ├── alternative_data_agent.py
│   ├── microstructure_agent.py
│   ├── fed_sentiment_agent.py
│   ├── onchain_agent.py
│   ├── geopolitical_agent.py
│   ├── chaos_agent.py
│   ├── retail_psychology_agent.py
│   ├── energy_frequency_agent.py ← Numerología, tarot, planetas
│   ├── report_agent.py           ← Reportes semanal/mensual
│   ├── screen_vision_agent.py    ← Claude Vision + mirror mode
│   ├── footprint_agent.py        ← Delta, absorción, imbalances
│   ├── statistical_edge_agent.py ← QuantEdgeAgent (10 módulos quant)
│   ├── quant_stats.py            ← VaR/CVaR/Kelly/Monte Carlo
│   ├── quant_regime.py           ← HMM régimen de mercado
│   ├── quant_factors.py          ← IC/IR factor analysis
│   ├── quant_anomalies.py        ← Calendar effects, funding extremos
│   ├── quant_ensemble.py         ← ML ensemble (sklearn)
│   ├── quant_optimizer.py        ← Bayesian optimization (Optuna)
│   ├── quant_flow.py             ← OFI, VPIN, Kyle impact
│   ├── quant_stress.py           ← 10 crash scenarios históricos
│   └── quant_intel.py            ← Papers académicos, insider activity
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
MT5_LOGIN=8889         ← BrokerGroup-Live24 (NO CONECTA)
MT5_PASSWORD=IMSMCbot  ← Pendiente actualizar a ICMarkets
MT5_SERVER=MetaQuotes-Demo ← Pendiente → ICMarketsSC-Demo
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
/elliott    → Ondas de Elliott
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

---

## 14. REGLAS PARA CLAUDE CODE EN ESTE PROYECTO

1. **LEER ESTE ARCHIVO** al inicio de cada sesión
2. **NUNCA romper** los 1182 tests existentes — verificar con `pytest tests/ -q`
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
| Scores siempre SHORT | Mercado bajista actual | Normal — seguir el mercado |
| supervisor.py sensible a encoding | PowerShell añade BOM/smart quotes | Siempre usar Write tool o deep_fix_supervisor.py |
| SMCBotEA no en charts | Acción manual requerida | Usuario debe arrastrar SMCBotEA a charts en MT5 GUI |

## 16. AUDITORÍA DE AGENTES (2026-05-25)

**Activos en el loop principal:**
- SignalAgent, DecisionFilter, BinanceConnector, MT5Connector, GlintBrowser
- TelegramCommander, TradingTelegramBot, RiskManager
- MarketStructure, OrderBlockDetector, FVGDetector (via _run_smc_lite)
- HistoricalDataAgent (solo via /history)
- AutonomousLearner, ResearchAgent, GoalsManager, NightlyReporter (loops nuevos)

**Dormant (existen pero NO en el scan loop):**
- SMCAnalysisAgent (Claude API), LunarAgent, ElliottAgent, InstitutionalFlowAgent
- AlternativeDataAgent, MicrostructureAgent, FedSentimentAgent, OnchainAgent
- GeopoliticalAgent, ChaosAgent, RetailPsychologyAgent, EnergyFrequencyAgent
- ReportAgent, ScreenVisionAgent, FootprintAgent, StatisticalEdgeAgent
- Todos los módulos Quant* (8), LearningEngine, AgentMemory

---

*Última actualización: 2026-05-26 | Tests: 1204 | Bot: PM2 ONLINE | Balance MT5: $99,955.37*
