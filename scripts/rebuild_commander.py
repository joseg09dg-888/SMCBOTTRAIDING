"""Rebuild telegram_commander.py from last clean commit + re-apply all additions."""
import subprocess, ast, os, re
os.chdir(r'C:\Users\jose-\projects\trading_agent')

# Try commits from newest to oldest until one parses
commits = [
    '9d41776',   # FootprintAgent (before MT5 forex introduced _cmd_status emojis)
    '43561ac',   # 5 bugs fix
    '44e74f6',   # QuantEdgeAgent
    '89cecb1',   # resilencia
]

base_content = None
for commit in commits:
    result = subprocess.run(
        ['git', 'show', f'{commit}:dashboard/telegram_commander.py'],
        capture_output=True
    )
    content = result.stdout.decode('utf-8-sig')
    # Strip any curly quotes
    content = content.replace('"', '"').replace('"', '"')
    content = content.replace(''', "'").replace(''', "'")
    content = content.replace('–', '-').replace('—', '-')
    try:
        ast.parse(content)
        print(f"Clean base from {commit}: {len(content)} chars")
        base_content = content
        break
    except SyntaxError as e:
        print(f"{commit}: BROKEN line {e.lineno}")

if not base_content:
    print("No clean base found!")
    exit(1)

content = base_content

# ── Apply all additions that were made after the base commit ──────────────

# 1. /health command (from agent_health_check)
if '"/health"' not in content:
    content = content.replace(
        '"/memory":           "Estado de memoria y accuracy de todos los agentes",',
        '"/memory":           "Estado de memoria y accuracy de todos los agentes",\n'
        '    "/health":           "Health check de los 21 agentes del bot",'
    )
    content = content.replace(
        '"/memory":    self._cmd_memory,',
        '"/memory":    self._cmd_memory,\n            "/health":     self._cmd_health,'
    )

# 2. Commands from later commits — add all at once to COMMANDS
new_commands = [
    ('"/energy"',           '"Lectura energetica del mercado. Ej: /energy BTC"'),
    ('"/reporte_semanal"',  '"Genera reporte semanal ahora"'),
    ('"/reporte_mensual"',  '"Genera reporte mensual ahora"'),
    ('"/criterios"',        '"Muestra criterios para ir a cuenta real"'),
    ('"/proyeccion"',       '"Proyeccion de la proxima semana"'),
    ('"/vision"',           '"Activa/desactiva vision de pantalla"'),
    ('"/screenshot"',       '"Captura y analiza pantalla ahora"'),
    ('"/mirror"',           '"Activa/desactiva modo espejo"'),
    ('"/analysis"',         '"Analisis SMC completo del mercado. Ej: /analysis BTC"'),
    ('"/onchain"',          '"Metricas on-chain actuales"'),
    ('"/lunar"',            '"Analisis de ciclos lunares"'),
    ('"/elliott"',          '"Conteo de ondas de Elliott en el simbolo activo"'),
    ('"/edge"',             '"Statistical edge y winrate historico del sistema"'),
    ('"/footprint"',        '"Analisis footprint (delta, absorcion, imbalances). Ej: /footprint BTC"'),
    ('"/ftmo"',             '"Estado del FTMO challenge y criterios para cuentas fondeadas"'),
    ('"/history"',          '"Analisis historico de un simbolo. Ej: /history BTC"'),
]

# Add to COMMANDS dict (only if not already there)
commands_insert = ''
for cmd, desc in new_commands:
    if cmd not in content:
        commands_insert += f'    {cmd}: {desc},\n'

if commands_insert:
    content = content.replace(
        '    "/youtube":   "Estado del aprendizaje YouTube",',
        '    "/youtube":   "Estado del aprendizaje YouTube",\n' + commands_insert
    )
    print(f"Added commands to COMMANDS dict")

# 3. Add all new commands to handlers dict
handlers_to_add = [
    ('"/health"',        'self._cmd_health'),
    ('"/energy"',        'self._cmd_energy'),
    ('"/reporte_semanal"', 'self._cmd_reporte_semanal'),
    ('"/reporte_mensual"', 'self._cmd_reporte_mensual'),
    ('"/criterios"',     'self._cmd_criterios'),
    ('"/proyeccion"',    'self._cmd_proyeccion'),
    ('"/vision"',        'self._cmd_vision'),
    ('"/screenshot"',    'self._cmd_screenshot'),
    ('"/mirror"',        'self._cmd_mirror'),
    ('"/analysis"',      'self._cmd_analysis'),
    ('"/onchain"',       'self._cmd_onchain'),
    ('"/lunar"',         'self._cmd_lunar'),
    ('"/elliott"',       'self._cmd_elliott'),
    ('"/edge"',          'self._cmd_edge'),
    ('"/footprint"',     'self._cmd_footprint'),
    ('"/ftmo"',          'self._cmd_ftmo'),
    ('"/history"',       'self._cmd_history'),
]

handlers_insert = ''
for cmd, method in handlers_to_add:
    if cmd not in content:
        handlers_insert += f'            {cmd}: {method},\n'

if handlers_insert:
    content = content.replace(
        '"/youtube":   self._cmd_youtube,',
        '"/youtube":   self._cmd_youtube,\n' + handlers_insert
    )
    print("Added handlers to handle_command()")

# 4. Add all stub methods before Helpers section
stub_methods = '''
    def _cmd_history(self) -> CommandResult:
        if self.on_history:
            try:
                text = self.on_history("BTC")
            except Exception as e:
                text = f"Error: {e}"
        else:
            text = "Agente historico no disponible."
        return CommandResult(success=True, message=text, action="history")

    def _cmd_health(self) -> CommandResult:
        from core.agent_health_check import AgentHealthCheck
        checker = AgentHealthCheck()
        report = checker.run_full_check()
        return CommandResult(success=True, message=report.format_telegram(), action="health")

    def _cmd_energy(self) -> CommandResult:
        from agents.energy_frequency_agent import EnergyFrequencyAgent
        agent = EnergyFrequencyAgent()
        reading = agent.analyze("BTC", price=0.0)
        return CommandResult(success=True, message=reading.format_telegram(), action="energy")

    def _cmd_reporte_semanal(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date, timedelta
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        stats = agent.calculate_weekly_stats(week_start)
        summary = agent.generate_telegram_summary(stats)
        return CommandResult(success=True, message=summary, action="reporte_semanal")

    def _cmd_reporte_mensual(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        from datetime import date
        agent = ReportAgent(capital=self.state.capital)
        today = date.today()
        stats = agent.calculate_monthly_stats(today.year, today.month)
        summary = agent.generate_telegram_summary(stats)
        return CommandResult(success=True, message=summary, action="reporte_mensual")

    def _cmd_criterios(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        agent = ReportAgent(capital=self.state.capital)
        return CommandResult(success=True, message=agent.generate_criteria_message(), action="criterios")

    def _cmd_proyeccion(self) -> CommandResult:
        from agents.report_agent import ReportAgent
        agent = ReportAgent(capital=self.state.capital)
        return CommandResult(success=True, message=agent.generate_projection_message(), action="proyeccion")

    def _cmd_vision(self) -> CommandResult:
        from agents.screen_vision_agent import ScreenVisionAgent
        agent = ScreenVisionAgent()
        new_state = agent.toggle()
        status = "activada" if new_state else "desactivada"
        return CommandResult(success=True, message=f"Vision de pantalla {status}.", action="vision")

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
            msg = "Modo espejo ACTIVADO."
        else:
            session = agent.stop_mirror_mode()
            actions = session.actions_recorded if session else 0
            msg = f"Modo espejo DESACTIVADO. Acciones: {actions}"
        return CommandResult(success=True, message=msg, action="mirror")

    def _cmd_analysis(self) -> CommandResult:
        return CommandResult(success=True, message="Analisis SMC: conecta el bot con datos en vivo para analisis completo.", action="analysis")

    def _cmd_onchain(self) -> CommandResult:
        return CommandResult(success=True, message="OnChain metrics: conecta el bot para ver flujos de ballenas y exchange netflow.", action="onchain")

    def _cmd_lunar(self) -> CommandResult:
        from agents.lunar_agent import LunarCycleAgent
        agent = LunarCycleAgent()
        return CommandResult(success=True, message=agent.format_telegram(), action="lunar")

    def _cmd_elliott(self) -> CommandResult:
        return CommandResult(success=True, message="Elliott Wave: conecta con datos OHLCV para conteo de ondas.", action="elliott")

    def _cmd_edge(self) -> CommandResult:
        return CommandResult(success=True, message="Statistical Edge: winrate historico disponible despues de 50+ trades demo.", action="edge")

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
        agent = FTMOAgent()
        state = FTMOAgent.new_challenge(initial_balance=self.state.capital, challenge_type=ChallengeType.TWO_STEP)
        if self.state.total_pnl != 0:
            try:
                state = agent.record_trade(state, self.state.total_pnl)
            except Exception:
                pass
        msg = agent.format_daily_report(state)
        income = agent.calculate_monthly_income(200000, 0.05, 0.90)
        msg += (
            "\\n━━━━━━━━━━━━━━━━━━━━\\n"
            "<b>POTENCIAL CON $200K FTMO</b>\\n"
            f"5%/mes: ${income['net_monthly']:,.0f}/mes | ${income['yearly']:,.0f}/anio"
        )
        return CommandResult(success=True, message=msg, action="ftmo")

'''

helpers_marker = '    # ── Helpers'
if helpers_marker not in content:
    helpers_marker = '    def _log_mode_change'
if stub_methods.strip() not in content:
    content = content.replace(helpers_marker, stub_methods + helpers_marker)
    print("All stub methods added")

# Final check
ast.parse(content)
print("Final content parses OK")
open('dashboard/telegram_commander.py', 'w', encoding='utf-8').write(content)
print("telegram_commander.py rebuilt successfully")
