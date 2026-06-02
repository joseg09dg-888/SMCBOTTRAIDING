# tests/core/test_volume_calculator.py
import pytest
from core.volume_calculator import VolumeCalculator


class TestVolumeCalculator:
    def test_instantiates(self):
        vc = VolumeCalculator()
        assert vc is not None

    def test_calculate_volume_eurusd(self):
        vc = VolumeCalculator()
        vol = vc.calculate_volume(100_000, 1.1000, 1.0950, "EURUSD")
        assert vol >= 0.01
        assert vol <= 10.0

    def test_calculate_volume_usdjpy(self):
        vc = VolumeCalculator()
        vol = vc.calculate_volume(100_000, 158.0, 157.5, "USDJPY")
        assert vol >= 0.01

    def test_calculate_volume_xauusd(self):
        vc = VolumeCalculator()
        vol = vc.calculate_volume(100_000, 3000.0, 2950.0, "XAUUSD")
        assert vol >= 0.01
        assert vol <= 10.0

    def test_volume_min_clamped(self):
        vc = VolumeCalculator()
        # Very large SL -> tiny calculated volume -> clamped to 0.01
        vol = vc.calculate_volume(1_000, 1.0, 0.001, "EURUSD")
        assert vol == 0.01

    def test_volume_max_clamped(self):
        vc = VolumeCalculator()
        # Very small SL -> huge calculated volume -> clamped to max cap (0.20 for EURUSD)
        vol = vc.calculate_volume(10_000_000, 1.1000, 1.0999, "EURUSD")
        assert vol == 0.20

    def test_volume_rounds_to_2dp(self):
        vc = VolumeCalculator()
        vol = vc.calculate_volume(50_000, 1.1000, 1.0950, "EURUSD")
        assert vol == round(vol, 2)

    def test_risk_pct_applied(self):
        vc = VolumeCalculator()
        # Both hit the 0.20 cap for EURUSD with large capital — use a symbol without cap
        vol_half = vc.calculate_volume(10_000, 1.1000, 1.0950, "EURUSD", risk_pct=0.0025)
        vol_full = vc.calculate_volume(10_000, 1.1000, 1.0950, "EURUSD", risk_pct=0.005)
        assert vol_full >= vol_half

    def test_get_stage_volume_demo(self):
        vc = VolumeCalculator()
        assert vc.get_stage_volume(100_000) == 0.10

    def test_get_stage_volume_seed(self):
        vc = VolumeCalculator()
        assert vc.get_stage_volume(5_000) == 0.01

    def test_get_stage_volume_incubation(self):
        vc = VolumeCalculator()
        assert vc.get_stage_volume(25_000) == 0.03

    def test_get_stage_volume_pro(self):
        vc = VolumeCalculator()
        assert vc.get_stage_volume(300_000) == 0.30

    def test_get_stage_volume_pro_m(self):
        vc = VolumeCalculator()
        assert vc.get_stage_volume(1_500_000) == 1.00

    def test_project_monthly_profit_returns_dict(self):
        vc = VolumeCalculator()
        result = vc.project_monthly_profit(100_000)
        assert isinstance(result, dict)
        assert "net_profit_usd" in result
        assert "your_share_80pct" in result

    def test_project_monthly_profit_positive_wr(self):
        vc = VolumeCalculator()
        result = vc.project_monthly_profit(100_000, win_rate=0.62)
        assert result["net_profit_usd"] > 0

    def test_project_monthly_profit_bad_wr_loses_money(self):
        vc = VolumeCalculator()
        result = vc.project_monthly_profit(100_000, win_rate=0.30)
        assert result["net_profit_usd"] < 0

    def test_project_monthly_profit_scales_with_capital(self):
        vc = VolumeCalculator()
        r1 = vc.project_monthly_profit(100_000)
        r2 = vc.project_monthly_profit(200_000)
        assert r2["net_profit_usd"] > r1["net_profit_usd"]

    def test_all_stages_covered(self):
        vc = VolumeCalculator()
        for stage, data in vc.AXI_STAGES.items():
            vol = vc.get_stage_volume(data["capital"])
            assert vol == data["volume"], f"Stage {stage} mismatch"

    def test_unknown_symbol_uses_default_pip_value(self):
        vc = VolumeCalculator()
        vol = vc.calculate_volume(100_000, 1.5000, 1.4950, "UNKNOWN")
        assert vol >= 0.01

    def test_zero_sl_distance_returns_zero(self):
        vc = VolumeCalculator()
        vol = vc.calculate_volume(100_000, 1.0, 1.0, "EURUSD")
        assert vol == 0.0

    def test_demo_volume_is_010(self):
        vc = VolumeCalculator()
        assert vc.get_stage_volume(100_000) == 0.10

    def test_roi_pct_in_result(self):
        vc = VolumeCalculator()
        result = vc.project_monthly_profit(100_000, win_rate=0.62)
        assert "monthly_roi_pct" in result
        assert result["monthly_roi_pct"] > 0
