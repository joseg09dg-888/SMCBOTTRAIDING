import pytest
from agents.axi_capital_adjuster import AxiCapitalAdjuster, AdjustResult


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """BUG-AXI-STATE-TORN-WRITE: several tests here call check() with a
    real jump, which persists to memory/axi_select_state.json. Without
    this isolation those writes land on the real production state file
    every time the suite runs."""
    monkeypatch.chdir(tmp_path)


def make_adjuster(known: float = 500.0) -> AxiCapitalAdjuster:
    a = AxiCapitalAdjuster()
    a._known_capital = known
    return a


def test_no_change_when_balance_same():
    a = make_adjuster(10_000)
    r = a.check(10_000)
    assert r.adjusted is False

def test_no_change_on_small_increase():
    a = make_adjuster(10_000)
    r = a.check(10_050)   # +0.5% < 10%
    assert r.adjusted is False

def test_jump_detected_by_pct():
    a = make_adjuster(10_000)
    r = a.check(11_500)   # +15% > 10%
    assert r.adjusted is True

def test_jump_detected_by_usd():
    a = make_adjuster(500)
    r = a.check(1_600)    # +$1100 > $1000 threshold
    assert r.adjusted is True

def test_known_capital_updated_after_jump():
    a = make_adjuster(10_000)
    a.check(50_000)
    assert a.known_capital == pytest.approx(50_000)

def test_no_update_without_jump():
    a = make_adjuster(10_000)
    a.check(10_050)
    assert a.known_capital == pytest.approx(10_000)

def test_sizing_pre_seed():
    a = make_adjuster(500)
    r = a.check(500)
    assert r.new_risk_pct == pytest.approx(0.020)

def test_sizing_seed():
    a = make_adjuster(1)
    r = a.check(5_001)
    assert r.new_risk_pct == pytest.approx(0.015)

def test_sizing_incubation():
    a = make_adjuster(1)
    r = a.check(50_001)
    assert r.new_risk_pct == pytest.approx(0.010)

def test_sizing_pro():
    a = make_adjuster(1)
    r = a.check(500_001)
    assert r.new_risk_pct == pytest.approx(0.006)

def test_max_risk_swing_scales():
    a = make_adjuster(1)
    r1 = a.check(500)
    a2 = make_adjuster(1)
    r2 = a2.check(50_001)
    assert r2.new_max_risk_swing > r1.new_max_risk_swing

def test_jump_pct_correct():
    a = make_adjuster(10_000)
    r = a.check(50_000)
    assert r.jump_pct == pytest.approx(4.0)   # 400%

def test_jump_usd_correct():
    a = make_adjuster(10_000)
    r = a.check(50_000)
    assert r.jump_usd == pytest.approx(40_000)

def test_reason_contains_new_capital():
    a = make_adjuster(10_000)
    r = a.check(50_000)
    assert "50,000" in r.reason or "50000" in r.reason

def test_format_telegram_adjusted():
    a = make_adjuster(10_000)
    r = a.check(50_000)
    msg = a.format_telegram(r)
    assert "CAPITAL ESCALADO" in msg
    assert "50,000" in msg

def test_format_telegram_no_change():
    a = make_adjuster(10_000)
    r = a.check(10_001)
    msg = a.format_telegram(r)
    assert "sin cambio" in msg.lower()

def test_adjust_result_is_dataclass():
    a = make_adjuster(10_000)
    r = a.check(50_000)
    assert isinstance(r, AdjustResult)
    assert r.adjusted is True
    assert r.prev_capital == pytest.approx(10_000)
    assert r.new_capital  == pytest.approx(50_000)
