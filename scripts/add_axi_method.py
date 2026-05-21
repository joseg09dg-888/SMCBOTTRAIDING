"""Add _cmd_axi as actual method to TelegramCommander class."""
import ast, os
os.chdir(r'C:\Users\jose-\projects\trading_agent')
content = open('dashboard/telegram_commander.py', encoding='utf-8').read()

# Check if method definition exists
has_method = 'def _cmd_axi(self)' in content
print(f"_cmd_axi method defined: {has_method}")
print(f"_cmd_axi in handlers: {'_cmd_axi' in content}")

if not has_method:
    axi_method = (
        '\n    def _cmd_axi(self) -> CommandResult:\n'
        '        from strategies.axi_select_agent import AxiSelectAgent\n'
        '        from core.score_db import get_stats\n'
        '        agent = AxiSelectAgent()\n'
        '        state = AxiSelectAgent.new_state(initial_balance=500.0)\n'
        '        stats = get_stats()\n'
        '        if stats["executed"] > 0:\n'
        '            state.trades_closed = stats["executed"]\n'
        '            state.wins   = stats["high_score"]\n'
        '            state.losses = max(0, stats["executed"] - stats["high_score"])\n'
        '            state.edge_score = agent.calculate_edge_score(state)\n'
        '            state.stage = agent.get_current_stage(state)\n'
        '        return CommandResult(success=True, message=agent.format_telegram(state), action="axi")\n'
    )
    # Insert before _log_mode_change
    target = '\n    def _log_mode_change'
    if target in content:
        content = content.replace(target, axi_method + target)
        print("_cmd_axi method inserted before _log_mode_change")
    else:
        # Find the Helpers comment section
        target2 = '\n    # -- Helpers'
        if target2 in content:
            content = content.replace(target2, axi_method + target2)
            print("_cmd_axi method inserted before Helpers section")
        else:
            print("ERROR: Could not find insertion point")
            exit(1)

ast.parse(content)
open('dashboard/telegram_commander.py', 'w', encoding='utf-8').write(content)
print(f"Method defined now: {'def _cmd_axi(self)' in content}")
print("telegram_commander.py updated")
