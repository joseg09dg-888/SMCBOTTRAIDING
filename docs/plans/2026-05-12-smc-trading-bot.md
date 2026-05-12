# SMC Trading Bot — Multi-Agent Implementation Plan
**Date:** 2026-05-12  
**Stack:** Python · Multi-agent · WebSocket · Telegram

---

## Metodología (inmutable)

- **SMC:** HH/HL/LH/LL, BOS, CHoCH, Order Blocks, FVG, Volume Profile, VWAP
- **Riesgo:** 0.5% por trade, SL obligatorio, RR mínimo 1:2
- **Señales:** Tiempo real vía WebSocket (Glint)
- **Modos:** Auto (ejecuta solo) | Semi-auto (aprobación por Telegram)
- **Regla de oro:** "Si no hay setup claro, no se opera"

---

## Task 1 — Setup del Proyecto

**Objetivo:** Estructura base del proyecto Python lista para desarrollo.

### Entregables
- [ ] Directorio del proyecto con estructura estándar
- [ ] `pyproject.toml` con dependencias core
- [ ] `Makefile` con comandos dev
- [ ] `.env.example` con variables requeridas
- [ ] `README.md` básico
- [ ] Git inicializado con `.gitignore`
- [ ] Tests de smoke passing (`pytest`)

### Estructura esperada
```
trading_agent/
├── src/
│   └── trading_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── agents/
│       ├── smc/
│       ├── risk/
│       └── connectors/
├── tests/
│   └── test_smoke.py
├── docs/
│   └── plans/
├── pyproject.toml
├── Makefile
├── .env.example
└── README.md
```

---

## Task 2 — SMC Core: Estructura de Mercado

**Objetivo:** Detectar HH/HL/LH/LL, BOS y CHoCH en datos OHLCV.

### Entregables
- [ ] `src/trading_agent/smc/structure.py` — swing points + market structure
- [ ] `src/trading_agent/smc/bos_choch.py` — BOS/CHoCH detection
- [ ] Tests unitarios con datos históricos sintéticos
- [ ] Cobertura ≥ 80%

---

## Task 3 — SMC Core: Order Blocks y FVG

**Objetivo:** Identificar zonas de interés institucional.

### Entregables
- [ ] `src/trading_agent/smc/order_blocks.py`
- [ ] `src/trading_agent/smc/fvg.py`
- [ ] Tests con casos edge (gaps, wicks extremos)

---

## Task 4 — SMC Core: Volume Profile y VWAP

**Objetivo:** Confluencia de precio con volumen.

### Entregables
- [ ] `src/trading_agent/smc/volume_profile.py`
- [ ] `src/trading_agent/smc/vwap.py`
- [ ] Integración con estructura de mercado

---

## Task 5 — Motor de Riesgo

**Objetivo:** Gestión de riesgo estricta antes de cualquier ejecución.

### Entregables
- [ ] `src/trading_agent/risk/manager.py`
  - Cálculo de tamaño de posición (0.5% del capital)
  - Validación de SL obligatorio
  - Validación de RR ≥ 1:2
  - Filtro "no setup claro → no operar"
- [ ] Tests exhaustivos con edge cases

---

## Task 6 — Conector WebSocket (Glint)

**Objetivo:** Recibir señales en tiempo real.

### Entregables
- [ ] `src/trading_agent/connectors/glint_ws.py`
- [ ] Reconexión automática
- [ ] Parser de mensajes SMC
- [ ] Tests con mock WebSocket

---

## Task 7 — Agentes: Señal + Ejecución

**Objetivo:** Arquitectura multi-agente coordinada.

### Entregables
- [ ] `src/trading_agent/agents/signal_agent.py` — consume señales, evalúa setup
- [ ] `src/trading_agent/agents/execution_agent.py` — ejecuta o solicita aprobación
- [ ] `src/trading_agent/agents/orchestrator.py` — coordina agentes
- [ ] Modo Auto vs Semi-auto

---

## Task 8 — Integración Telegram

**Objetivo:** Notificaciones y aprobación en modo Semi-auto.

### Entregables
- [ ] `src/trading_agent/connectors/telegram_bot.py`
- [ ] Alertas de señal detectada
- [ ] Flujo de aprobación inline (Aprobar / Rechazar)
- [ ] Notificaciones de trade ejecutado / SL / TP

---

## Task 9 — Backtesting

**Objetivo:** Validar la estrategia con datos históricos.

### Entregables
- [ ] `src/trading_agent/backtesting/engine.py`
- [ ] Reporte de métricas: Win Rate, Profit Factor, Max Drawdown, Sharpe
- [ ] Dataset de prueba incluido

---

## Task 10 — Integración Final y CI

**Objetivo:** Bot funcionando end-to-end + pipeline CI.

### Entregables
- [ ] `docker-compose.yml` con todos los servicios
- [ ] GitHub Actions: lint + test en cada PR
- [ ] Documentación de despliegue
- [ ] Test de integración completo (mock exchange)
