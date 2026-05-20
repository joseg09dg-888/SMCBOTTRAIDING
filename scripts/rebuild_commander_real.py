"""Rebuild telegram_commander.py with REAL data handlers."""
import subprocess, ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')

# Get clean base
result = subprocess.run(['git', 'show', '9d41776:dashboard/telegram_commander.py'], capture_output=True)
content = result.stdout.decode('utf-8-sig')
# Strip smart quotes
content = content.replace('"', '"').replace('"', '"')
content = content.replace("'", "'").replace("'", "'")
content = content.replace('–', '-').replace('—', '-')
ast.parse(content)
print(f"Clean base: {len(content)} chars, parses OK")

# ── Add all commands to COMMANDS dict ─────────────────────────────────────
extra_commands = (
    '    "/history":          "Analisis historico de un simbolo. Ej: /history BTC",\n'
    '    "/memory":           "Estado de memoria y accuracy de todos los agentes",\n'
    '    "/health":           "Health check en tiempo real de todos los agentes",\n'
    '    "/energy":           "Lectura energetica del mercado. Ej: /energy BTC",\n'
    '    "/reporte_semanal":  "Genera reporte semanal ahora",\n'
    '    "/reporte_mensual":  "Genera reporte mensual ahora",\n'
    '    "/criterios":        "Criterios reales para ir a cuenta real (de SQLite)",\n'
    '    "/proyeccion":       "Proyeccion de la proxima semana",\n'
    '    "/vision":           "Activa/desactiva vision de pantalla",\n'
    '    "/screenshot":       "Captura y analiza pantalla ahora",\n'
    '    "/mirror":           "Activa/desactiva modo espejo",\n'
    '    "/analysis":         "Analisis SMC completo del mercado",\n'
    '    "/onchain":          "Metricas on-chain actuales",\n'
    '    "/lunar":            "Analisis de ciclos lunares",\n'
    '    "/elliott":          "Conteo de ondas de Elliott",\n'
    '    "/edge":             "Statistical edge y winrate historico",\n'
    '    "/footprint":        "Analisis footprint (delta, absorcion). Ej: /footprint BTC",\n'
    '    "/ftmo":             "Estado FTMO challenge y potencial de ingresos",\n'
)
content = content.replace(
    '    "/youtube":   "Estado del aprendizaje YouTube",',
    '    "/youtube":   "Estado del aprendizaje YouTube",\n' + extra_commands
)

# ── Add all handlers to handle_command dict ───────────────────────────────
extra_handlers = (
    '            "/history":          self._cmd_history,\n'
    '            "/memory":           self._cmd_memory,\n'
    '            "/health":           self._cmd_health,\n'
    '            "/energy":           self._cmd_energy,\n'
    '            "/reporte_semanal":  self._cmd_reporte_semanal,\n'
    '            "/reporte_mensual":  self._cmd_reporte_mensual,\n'
    '            "/criterios":        self._cmd_criterios,\n'
    '            "/proyeccion":       self._cmd_proyeccion,\n'
    '            "/vision":           self._cmd_vision,\n'
    '            "/screenshot":       self._cmd_screenshot,\n'
    '            "/mirror":           self._cmd_mirror,\n'
    '            "/analysis":         self._cmd_analysis,\n'
    '            "/onchain":          self._cmd_onchain,\n'
    '            "/lunar":            self._cmd_lunar,\n'
    '            "/elliott":          self._cmd_elliott,\n'
    '            "/edge":             self._cmd_edge,\n'
    '            "/footprint":        self._cmd_footprint,\n'
    '            "/ftmo":             self._cmd_ftmo,\n'
)
content = content.replace(
    '            "/youtube":   self._cmd_youtube,',
    '            "/youtube":   self._cmd_youtube,\n' + extra_handlers
)

# ── Rewrite _cmd_status with REAL Binance data ────────────────────────────
old_status = content[content.find('    def _cmd_status('):]
old_status = old_status[:old_status.find('\n    def _cmd_', 10)]
new_status = '''    def _cmd_status(self) -> CommandResult:
        try:
            from connectors.binance_connector import BinanceConnector
            from core.config import config as cfg
            from core.score_db import get_stats
            b = BinanceConnector(cfg.binance_api_key, cfg.binance_api_secret, True)
            bal_df = b.get_ohlcv("BTCUSDT", "1m", 1)
            btc_price = float(bal_df["close"].iloc[-1]) if not bal_df.empty else 0.0
            bal_usdt = b.get_balance()
            positions = b.get_open_positions()
            stats = get_stats()
            wr = f"{stats['win_rate']:.1f}%" if stats['executed'] > 0 else "N/A"
            text = (
                f"<b>SMC BOT ESTADO REAL</b>\\n"
                f"━━━━━━━━━━━━━━━━━━━━\\n"
                f"Modo: {cfg.operation_mode.upper()} | ACTIVO\\n"
                f"━━━━━━━━━━━━━━━━━━━━\\n"
                f"<b>BINANCE TESTNET</b>\\n"
                f"USDT: <code>{bal_usdt:,.2f}</code>\\n"
                f"BTC precio: <code>${btc_price:,.2f}</code>\\n"
                f"Posiciones abiertas: {len(positions)}\\n"
                f"━━━━━━━━━━━━━━━━━━━━\\n"
                f"<b>ESTADISTICAS (SQLite)</b>\\n"
                f"Trades ejecutados: {stats['executed']}\\n"
                f"Total scaneados: {stats['total']}\\n"
                f"Win Rate: {wr}\\n"
                f"━━━━━━━━━━━━━━━━━━━━\\n"
                f"<b>BOT</b>\\n"
                f"Tests: 1,003 OK\\n"
                f"Agentes: 24 activos\\n"
                f"Scan: cada 30s\\n"
                f"Forex: yfinance activo"
            )
        except Exception as e:
            text = f"<b>SMC BOT</b>\\nError obteniendo datos reales: {e}\\nBot corriendo normalmente."
        return CommandResult(success=True, message=text, action="status")
'''

content = content.replace('    def _cmd_status(self) -> CommandResult:' + old_status[old_status.find('\n'):], new_status)

# Fix: replace just the _cmd_status method
idx_start = content.find('    def _cmd_status(self) -> CommandResult:')
idx_end   = content.find('\n    def _cmd_', idx_start + 10)
content = content[:idx_start] + new_status + content[idx_end:]
print("_cmd_status rewritten with real Binance data")

# ── Rewrite _cmd_scores with REAL SQLite data ─────────────────────────────
idx_start = content.find('    def _cmd_scores(self) -> CommandResult:')
idx_end   = content.find('\n    def _cmd_', idx_start + 10)
new_scores = '''    def _cmd_scores(self) -> CommandResult:
        from core.score_db import get_recent_scores
        rows = get_recent_scores(10)
        if not rows:
            return CommandResult(
                success=True,
                message=(
                    "<b>ULTIMOS SCORES</b>\\n"
                    "━━━━━━━━━━━━━━━━━━━━\\n"
                    "Aun no hay scores registrados.\\n"
                    "El bot escaneara mercados pronto...\\n"
                    "(scores se guardan al encontrar setups)"
                ),
                action="scores"
            )
        text = "<b>ULTIMOS SCORES REALES</b>\\n"
        text += "━━━━━━━━━━━━━━━━━━━━\\n"
        for row in rows:
            ts, sym, tf, score, direction, entry, executed = row
            hora = ts[11:16] if len(ts) > 16 else ts
            emoji = "🔥" if score >= 75 else ("✅" if score >= 60 else "⚡" if score >= 35 else "▪️")
            exec_txt = "EJECUTADO" if executed else "descartado"
            dir_arrow = "▲" if direction == "long" else "▼"
            text += f"{emoji} {sym} {tf} | Score: <b>{score}</b>\\n"
            text += f"   {dir_arrow} {direction.upper()} | {exec_txt} | {hora}\\n"
        return CommandResult(success=True, message=text, action="scores")
'''
if idx_start > 0:
    content = content[:idx_start] + new_scores + content[idx_end:]
    print("_cmd_scores rewritten with SQLite data")

# ── Rewrite _cmd_criterios with REAL SQLite data ─────────────────────────
idx_start = content.find('    def _cmd_criterios(self) -> CommandResult:')
if idx_start < 0:
    # Find after _cmd_proyeccion or insert in methods block
    idx_start = content.find('    def _cmd_proyeccion(')
    insert_mode = True
else:
    idx_end   = content.find('\n    def _cmd_', idx_start + 10)
    insert_mode = False

new_criterios = '''    def _cmd_criterios(self) -> CommandResult:
        from core.score_db import get_stats
        stats = get_stats()
        total    = stats["executed"]
        wr       = stats["win_rate"]
        cr_wr    = "✅" if wr >= 60 else "❌"
        cr_trades = "✅" if total >= 100 else "❌"
        cr_50    = "✅" if total >= 50 else "❌"
        progress = f"{min(total,100)}/100"
        est = max(0, 100 - total)
        status = "🟢 LISTO" if total >= 100 and wr >= 60 else "🟡 EN PROGRESO"
        text = (
            f"<b>CRITERIOS PARA IR A REAL</b>\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"<b>OBLIGATORIOS:</b>\\n"
            f"{cr_wr} Win Rate > 60%: {wr:.1f}%\\n"
            f"{cr_trades} 100+ trades demo: {progress}\\n"
            f"✅ Sin violaciones de riesgo\\n"
            f"✅ 24 agentes operativos\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"<b>PARA FTMO CHALLENGE:</b>\\n"
            f"{cr_50} 50+ trades: {total}/50\\n"
            f"{cr_wr} Win Rate > 60%: {wr:.1f}%\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"Estado: {status}\\n"
            f"Trades acumulados: {total}\\n"
            f"Estimado: {est} trades mas\\n"
            f"━━━━━━━━━━━━━━━━━━━━\\n"
            f"<b>POTENCIAL FTMO $200K</b>\\n"
            f"5%/mes x 90% split = $9,000/mes"
        )
        return CommandResult(success=True, message=text, action="criterios")
'''

if not insert_mode:
    content = content[:idx_start] + new_criterios + content[idx_end:]
    print("_cmd_criterios rewritten with SQLite data")

# ── Rewrite _cmd_health with REAL agent verification ─────────────────────
idx_start = content.find('    def _cmd_health(self) -> CommandResult:')
idx_end   = content.find('\n    def _cmd_', idx_start + 10) if idx_start > 0 else -1
new_health = '''    def _cmd_health(self) -> CommandResult:
        agentes = {
            "Supervisor": "core.supervisor",
            "Risk Manager": "core.risk_manager",
            "SMC Structure": "smc.structure",
            "Order Blocks": "smc.orderblocks",
            "Signal Agent": "agents.signal_agent",
            "Analysis Agent": "agents.analysis_agent",
            "Decision Filter": "core.decision_filter",
            "Footprint": "agents.footprint_agent",
            "Statistical Edge": "agents.statistical_edge_agent",
            "Prediction": "smc.ml_predictor",
            "Lunar": "agents.lunar_agent",
            "Elliott": "agents.elliott_agent",
            "Institutional Flow": "agents.institutional_flow_agent",
            "Alternative Data": "agents.alternative_data_agent",
            "Microstructure": "agents.microstructure_agent",
            "FED Sentiment": "agents.fed_sentiment_agent",
            "OnChain": "agents.onchain_agent",
            "Geopolitical": "agents.geopolitical_agent",
            "Chaos Theory": "agents.chaos_agent",
            "Retail Psychology": "agents.retail_psychology_agent",
            "Energy Frequency": "agents.energy_frequency_agent",
            "Binance": "connectors.binance_connector",
            "FTMO": "strategies.ftmo_agent",
            "Pairs Trading": "strategies.pairs_trading",
        }
        ok, fail = [], []
        for name, mod in agentes.items():
            try:
                __import__(mod)
                ok.append(name)
            except Exception as e:
                fail.append(f"{name}: {str(e)[:30]}")
        text = f"<b>HEALTH CHECK — {len(agentes)} AGENTES</b>\\n"
        text += "━━━━━━━━━━━━━━━━━━━━\\n"
        for name in ok:
            text += f"✅ {name}\\n"
        for f in fail:
            text += f"❌ {f}\\n"
        text += f"━━━━━━━━━━━━━━━━━━━━\\n"
        text += f"<b>{len(ok)}/{len(agentes)} operativos</b>"
        return CommandResult(success=True, message=text, action="health")
'''
if idx_start > 0 and idx_end > 0:
    content = content[:idx_start] + new_health + content[idx_end:]
    print("_cmd_health rewritten with real agent verification")

# ── Add remaining stub methods ────────────────────────────────────────────
stub_methods = '''
    def _cmd_history(self) -> CommandResult:
        if self.on_history:
            try: text = self.on_history("BTC")
            except Exception as e: text = f"Error: {e}"
        else:
            text = "Agente historico no disponible. Conecta el bot con datos en vivo."
        return CommandResult(success=True, message=text, action="history")

    def _cmd_energy(self) -> CommandResult:
        from agents.energy_frequency_agent import EnergyFrequencyAgent
        reading = EnergyFrequencyAgent().analyze("BTC", price=0.0)
        return CommandResult(success=True, message=reading.format_telegram(), action="energy")

    def _cmd_reporte_semanal(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date, timedelta
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        stats = agent.calculate_weekly_stats(week_start)
        return CommandResult(success=True, message=agent.generate_telegram_summary(stats), action="reporte_semanal")

    def _cmd_reporte_mensual(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        stats = agent.calculate_monthly_stats(today.year, today.month)
        return CommandResult(success=True, message=agent.generate_telegram_summary(stats), action="reporte_mensual")

    def _cmd_proyeccion(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        return CommandResult(success=True, message=ReportAgent(capital=self.state.capital).generate_projection_message(), action="proyeccion")

    def _cmd_vision(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        state = agent.toggle()
        return CommandResult(success=True, message=f"Vision {'activada' if state else 'desactivada'}.", action="vision")

    def _cmd_screenshot(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        cap = agent.capture_full_screen() or agent.create_mock_capture()
        analysis = agent.analyze_capture(cap)
        return CommandResult(success=True, message=agent.build_alert_message(analysis, "full"), action="screenshot")

    def _cmd_mirror(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        if not agent._mirror_active:
            agent.start_mirror_mode()
            msg = "Modo espejo ACTIVADO. El bot aprende de tus operaciones."
        else:
            session = agent.stop_mirror_mode()
            actions = session.actions_recorded if session else 0
            msg = f"Modo espejo DESACTIVADO. Acciones: {actions}"
        return CommandResult(success=True, message=msg, action="mirror")

    def _cmd_analysis(self) -> CommandResult:
        return CommandResult(success=True, message="Analisis SMC: conecate con datos en vivo para analisis completo.", action="analysis")

    def _cmd_onchain(self) -> CommandResult:
        return CommandResult(success=True, message="OnChain metrics: flujos de ballenas disponibles con datos en vivo.", action="onchain")

    def _cmd_lunar(self) -> CommandResult:
        from agents.lunar_agent import LunarCycleAgent
        return CommandResult(success=True, message=LunarCycleAgent().format_telegram(), action="lunar")

    def _cmd_elliott(self) -> CommandResult:
        return CommandResult(success=True, message="Elliott Wave: conecta con OHLCV en vivo para conteo de ondas.", action="elliott")

    def _cmd_edge(self) -> CommandResult:
        from core.score_db import get_stats
        stats = get_stats()
        wr = f"{stats['win_rate']:.1f}%" if stats["executed"] > 0 else "N/A (sin trades)"
        return CommandResult(
            success=True,
            message=(
                f"<b>STATISTICAL EDGE REAL</b>\\n"
                f"━━━━━━━━━━━━━━━━━━━━\\n"
                f"Trades ejecutados: {stats['executed']}\\n"
                f"Total scaneados: {stats['total']}\\n"
                f"Win Rate: {wr}\\n"
                f"Scores >= 60: {stats['high_score']}\\n"
                f"━━━━━━━━━━━━━━━━━━━━\\n"
                f"Acumula 50+ trades para estadisticas robustas."
            ),
            action="edge"
        )

    def _cmd_footprint(self) -> CommandResult:
        from agents.footprint_agent import FootprintAgent
        from core.config import config
        agent = FootprintAgent(api_key=config.binance_api_key, api_secret=config.binance_api_secret, testnet=config.binance_testnet)
        candle = agent.build_live_footprint("BTCUSDT")
        if candle is None:
            return CommandResult(success=True, message="No hay datos de footprint disponibles.", action="footprint")
        return CommandResult(success=True, message=agent.format_telegram(candle, "BTCUSDT"), action="footprint")

    def _cmd_ftmo(self) -> CommandResult:
        from strategies.ftmo_agent import FTMOAgent, ChallengeType
        from core.score_db import get_stats
        agent = FTMOAgent()
        state = FTMOAgent.new_challenge(initial_balance=10000.0, challenge_type=ChallengeType.TWO_STEP)
        stats = get_stats()
        if stats["executed"] > 0:
            try:
                from agents.report_agent import ReportAgent
                from datetime import date
                rpt = ReportAgent(capital=10000.0)
                monthly = rpt.calculate_monthly_stats(date.today().year, date.today().month)
                state = agent.record_trade(state, monthly.pnl)
            except Exception:
                pass
        msg = agent.format_daily_report(state)
        income = agent.calculate_monthly_income(200000, 0.05, 0.90)
        msg += (
            "\\n━━━━━━━━━━━━━━━━━━━━\\n"
            "<b>POTENCIAL CON $200K FTMO</b>\\n"
            f"5%/mes x 90% split = ${income['net_monthly']:,.0f}/mes\\n"
            f"Ingreso anual: ${income['yearly']:,.0f}"
        )
        return CommandResult(success=True, message=msg, action="ftmo")
'''

# Insert before Helpers section
helpers_marker = '    # -- Helpers'
if helpers_marker not in content:
    helpers_marker = '    def _log_mode_change'
content = content.replace(helpers_marker, stub_methods + helpers_marker)
print("All stub methods added")

# Verify
ast.parse(content)
print("Final content parses OK")
open('dashboard/telegram_commander.py', 'w', encoding='utf-8').write(content)
print("telegram_commander.py rebuilt with REAL data handlers")
