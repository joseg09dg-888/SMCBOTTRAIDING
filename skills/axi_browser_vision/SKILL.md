# Skill: axi_browser_vision

## Purpose
Claude reads the MT5 Axi terminal screen visually using Claude Vision API.
This lets Claude see what you see — balance, open positions, charts, P&L — and
make decisions based on real visual information rather than API data alone.

## Commands (Telegram)

| Command | What it does |
|---------|-------------|
| `/ver_mt5` | Captures screen now. Claude analyzes and reports: balance, positions, P&L, setup visible, recommendation |
| `/proteger` | Toggles protection mode. ON = screen checked every 2 min, auto-close if position loses > $500 |

## Auto-monitoring (always active)
The `_vision_monitor_loop` runs every 5 minutes (2 min in protect mode):
1. Captures screen
2. Claude extracts: balance, equity, positions with P&L
3. If balance < $100,000 (start capital) → Telegram alert
4. If any position loses > $100 → warning alert
5. If any position loses > $500 → alert + auto-close the position

## Files
- `agents/axi_vision_agent.py` — AxiVisionAgent class
- `core/supervisor.py` — `_vision_monitor_loop()` integration
- `dashboard/telegram_commander.py` — `/ver_mt5`, `/proteger` commands

## Balance growth check
The bot tracks the starting capital ($100,000 Axi demo seed).
Every screen check verifies if balance grew or shrank and alerts accordingly.
This lets you see at a glance whether the bot is making money.

## Dependencies
- `mss` — screen capture
- `Pillow` — image processing
- `anthropic` — Claude Vision API (claude-sonnet-4-6)

## Usage
No setup needed. The loop starts automatically with `pm2 start ecosystem.config.js`.
Use `/ver_mt5` any time you want an immediate visual snapshot.
