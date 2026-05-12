# SMC Trading Bot

Multi-agent trading bot based on Smart Money Concepts (SMC).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # Fill in API keys
```

## Run

```bash
python main.py
```

## Test

```bash
pytest tests/ -v
```

## Methodology

- Structure: HH/HL/LH/LL, BOS, CHoCH
- Order Blocks, FVG, Volume Profile, Anchored VWAP
- Risk: 0.5% per trade, mandatory SL, min RR 1:2
- Modes: Auto | Semi-auto (Telegram approval) | Alerts
- Rule: "If there's no clear setup, don't trade"
