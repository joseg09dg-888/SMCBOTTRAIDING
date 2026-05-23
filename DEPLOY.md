# Deploy SMC Trading Bot en Railway

## Deploy en 5 minutos (gratis hasta $5/mes)

### 1. Ir a Railway
**https://railway.app** → Login con GitHub

### 2. Crear proyecto
- Click **"New Project"**
- Click **"Deploy from GitHub repo"**
- Seleccionar: `joseg09dg-888/SMCBOTTRAIDING`

### 3. Añadir variables de entorno
En el panel de Railway → **Variables** → añadir:

```
ANTHROPIC_API_KEY    = sk-ant-api03-...  (de console.anthropic.com)
BINANCE_API_KEY      = xZFkDXjm...      (de testnet.binance.vision)
BINANCE_API_SECRET   = WctNcVRR...
BINANCE_TESTNET      = true
TELEGRAM_BOT_TOKEN   = 8273150769:AAEp...
TELEGRAM_CHAT_ID     = 5371315570
OPERATION_MODE       = auto
```

### 4. Deploy
Click **"Deploy"** — en 3 minutos el bot está corriendo 24/7.

---

## Lo que corre en Railway (Linux)

| Mercado | Fuente | Estado |
|---------|--------|--------|
| Crypto (6 pares) | Binance API | ACTIVO |
| Forex (4 pares) | yfinance | ACTIVO |
| MT5 Axi | NO disponible en Linux | Usa yfinance |

## Para MT5 real en producción

**ForexVPS.net** ($5-10/mes) — VPS Windows con MT5 preinstalado:
1. Contratar VPS Windows
2. Instalar bot con startup.py
3. Conectar Axi MT5 en el VPS
4. El bot opera crypto + forex via MT5 24/7

---

## Monitoreo

El bot envía mensajes a Telegram:
- Cada trade ejecutado
- Cuando MT5 se desconecta/reconecta
- Nuevos artículos de MQL5 cada 6h
- Alertas de riesgo

## Variables opcionales

```
MT5_LOGIN      = 10042896      (solo Windows)
MT5_PASSWORD   = IMSMCbot*3axi (solo Windows)
MT5_SERVER     = Axi-US50-Demo (solo Windows)
GLINT_EMAIL    = joseg09.dg@gmail.com
```
