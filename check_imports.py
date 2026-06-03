import sys
sys.path.insert(0, '.')
modules = [
    'core.config','core.risk_manager','core.supervisor','core.learning_engine',
    'core.mode_manager','core.agent_memory','core.agent_health_check',
    'core.continuous_learning','core.wakeup_recovery','core.decision_filter',
    'smc.structure','smc.orderblocks','smc.volume_profile','smc.sentiment','smc.ml_predictor',
    'agents.analysis_agent','agents.signal_agent','agents.lunar_agent',
    'agents.elliott_agent','agents.institutional_flow_agent',
    'agents.alternative_data_agent','agents.microstructure_agent',
    'agents.fed_sentiment_agent','agents.onchain_agent','agents.geopolitical_agent',
    'agents.chaos_agent','agents.retail_psychology_agent',
    'agents.energy_frequency_agent','agents.report_agent','agents.screen_vision_agent',
    'agents.quant_stats','agents.quant_regime','agents.quant_factors',
    'agents.quant_anomalies','agents.quant_ensemble','agents.quant_optimizer',
    'agents.quant_flow','agents.quant_stress','agents.quant_intel',
    'agents.statistical_edge_agent',
    'connectors.binance_connector','connectors.metatrader_connector',
    'connectors.market_connector','connectors.glint_connector','connectors.glint_browser',
    'dashboard.telegram_bot','dashboard.telegram_commander','dashboard.screenshot_engine',
    'training.youtube_trainer','training.historical_agent','training.curriculum',
]
passed, failed = [], []
for m in modules:
    try:
        __import__(m)
        passed.append(m)
        print(f'OK  {m}')
    except Exception as e:
        failed.append((m, str(e)))
        print(f'FAIL {m}: {e}')
print(f'\n{len(passed)}/{len(modules)} modules OK')
if failed:
    print('\nBROKEN:')
    for m, e in failed:
        print(f'  FAIL {m}: {e}')
