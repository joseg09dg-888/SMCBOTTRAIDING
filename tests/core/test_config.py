import pytest
from core.config import Config


def test_config_loads_defaults():
    cfg = Config()
    assert cfg.max_risk_per_trade == 0.005
    assert cfg.max_open_positions == 3
    assert cfg.operation_mode == "hybrid"
    assert "1h" in cfg.timeframes
    assert "4h" in cfg.timeframes


def test_config_risk_is_half_percent():
    cfg = Config()
    assert cfg.max_risk_per_trade == 0.005, "Riesgo debe ser exactamente 0.5%"


def test_config_demo_by_default():
    cfg = Config()
    assert cfg.binance_testnet is True, "Debe iniciar en testnet/demo"
    assert cfg.mt5_demo is True, "MT5 debe iniciar en demo"
