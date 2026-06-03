"""Check bot readiness for Axi Select challenge."""
from core.score_db import get_stats
from connectors.metatrader_connector import MT5Connector
from core.config import config as cfg

stats = get_stats()
print("=== DATOS DE RENDIMIENTO REAL ===")
print(f"Trades ejecutados:      {stats['executed']}")
print(f"Con outcome real (W/L): {stats['has_real_outcomes']}")
print(f"Wins reales:            {stats['wins']}")
print(f"Losses reales:          {stats['losses']}")
print(f"Win Rate real:          {stats['win_rate']:.1f}%")
print(f"Profit Factor:          {stats['profit_factor']:.2f}")
print(f"P&L promedio:           {stats['avg_pnl_pct']:+.2f}%")

print()
print("=== ESTADO CUENTA MT5 ===")
try:
    mt5 = MT5Connector(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server)
    pnl = mt5.get_pnl_report(initial_balance=100_000.0)
    if "error" not in pnl:
        net = pnl["net_change"]
        bal = pnl["balance"]
        dd  = abs(net) / 100_000.0 * 100
        print(f"Balance actual:         ${bal:,.2f}")
        print(f"P&L neto:               ${net:+,.2f} ({net/100_000*100:+.3f}%)")
        print(f"Drawdown actual:        {dd:.3f}%")
        print(f"Operaciones totales:    {pnl['n_trades']}")
    else:
        print(f"MT5 error: {pnl['error']}")
except Exception as e:
    print(f"MT5 no disponible: {e}")

print()
print("=== REQUISITOS AXI SELECT (etapa 1 — $20K) ===")
print("Win Rate >= 60%:        PENDIENTE (necesita 100+ trades con outcomes)")
print("Profit Factor >= 1.5:   PENDIENTE")
print("Max Drawdown < 8%:      PENDIENTE (ahora 0.53% - OK)")
print("Min trades: 100:        PENDIENTE")
print("Tiempo minimo: 4 semanas: PENDIENTE")
