import pytest
import numpy as np
from agents.footprint_agent import FootprintAgent, FootprintCandle, FootprintLevel

def make_trades(n_buy=100, n_sell=80, price_range=(100.0, 105.0), seed=42):
    rng = np.random.default_rng(seed)
    n = n_buy + n_sell
    prices = rng.uniform(price_range[0], price_range[1], n)
    qtys   = rng.uniform(0.1, 2.0, n)
    trades = []
    for i in range(n_buy):
        trades.append({'price': float(prices[i]), 'qty': float(qtys[i]), 'isBuyerMaker': False})
    for i in range(n_buy, n):
        trades.append({'price': float(prices[i]), 'qty': float(qtys[i]), 'isBuyerMaker': True})
    return trades

def make_candle(total_delta=100.0, absorption=False, exhaustion_high=False,
                exhaustion_low=False, stacked=0, poc_price=100.0):
    levels = [FootprintLevel(100.0, 80.0, 100.0, 20.0, "buying")]
    c = FootprintCandle(
        open=99.0, high=105.0, low=98.0, close=103.0,
        total_volume=180.0, total_delta=total_delta,
        poc_price=poc_price, poc_volume=50.0, levels=levels,
        absorption=absorption, exhaustion_high=exhaustion_high,
        exhaustion_low=exhaustion_low, stacked_imbalances=stacked,
        imbalance_zones=[])
    return c

# FootprintLevel
def test_level_total_volume():
    l = FootprintLevel(100.0, 40.0, 60.0, 20.0, "buying")
    assert l.total_volume == pytest.approx(100.0)

def test_level_imbalance_ratio():
    l = FootprintLevel(100.0, 10.0, 40.0, 30.0, "buying")
    assert l.imbalance_ratio == pytest.approx(4.0)

def test_level_imbalance_ratio_zero_bid():
    l = FootprintLevel(100.0, 0.0, 50.0, 50.0, "buying")
    assert l.imbalance_ratio == float('inf')

def test_level_delta():
    l = FootprintLevel(100.0, 30.0, 50.0, 20.0, "buying")
    assert l.delta == pytest.approx(20.0)

# build_footprint_from_trades
def test_build_returns_candle():
    agent = FootprintAgent(tick_size=1.0)
    assert isinstance(agent.build_footprint_from_trades(make_trades(), 100.0, 105.0, 98.0, 103.0), FootprintCandle)

def test_build_levels_not_empty():
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades(make_trades(), 100.0, 105.0, 98.0, 103.0)
    assert len(c.levels) > 0

def test_build_total_volume():
    trades = [{'price':100.0,'qty':10.0,'isBuyerMaker':False},
              {'price':100.0,'qty':5.0, 'isBuyerMaker':True},
              {'price':101.0,'qty':8.0, 'isBuyerMaker':False}]
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades(trades, 99.0, 102.0, 99.0, 101.0)
    assert c.total_volume == pytest.approx(23.0)

def test_build_total_delta():
    trades = [{'price':100.0,'qty':10.0,'isBuyerMaker':False},
              {'price':100.0,'qty':5.0, 'isBuyerMaker':True}]
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades(trades, 99.0, 101.0, 99.0, 100.0)
    assert c.total_delta == pytest.approx(5.0)

def test_build_poc_max_volume():
    trades = [{'price':100.0,'qty':50.0,'isBuyerMaker':False},
              {'price':101.0,'qty':5.0, 'isBuyerMaker':False}]
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades(trades, 99.0, 102.0, 99.0, 101.0)
    assert c.poc_price == pytest.approx(100.0)
    assert c.poc_volume == pytest.approx(50.0)

def test_build_empty_trades():
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades([], 100.0, 105.0, 98.0, 103.0)
    assert isinstance(c, FootprintCandle)
    assert c.total_volume == 0.0

def test_build_imbalance_buying():
    trades = [{'price':100.0,'qty':40.0,'isBuyerMaker':False},
              {'price':100.0,'qty':5.0, 'isBuyerMaker':True}]
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades(trades, 99.0, 101.0, 99.0, 100.5)
    lvl = next(l for l in c.levels if abs(l.price - 100.0) < 0.01)
    assert lvl.imbalance == "buying"

def test_build_imbalance_selling():
    trades = [{'price':100.0,'qty':5.0, 'isBuyerMaker':False},
              {'price':100.0,'qty':40.0,'isBuyerMaker':True}]
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades(trades, 99.0, 101.0, 99.0, 100.5)
    lvl = next(l for l in c.levels if abs(l.price - 100.0) < 0.01)
    assert lvl.imbalance == "selling"

def test_build_levels_sorted():
    agent = FootprintAgent(tick_size=1.0)
    c = agent.build_footprint_from_trades(make_trades(price_range=(98.0, 103.0), seed=7), 98.0, 103.0, 98.0, 103.0)
    prices = [l.price for l in c.levels]
    assert prices == sorted(prices)

# delta_ratio
def test_delta_ratio_positive():
    c = make_candle(total_delta=50.0)
    assert c.delta_ratio == pytest.approx(50.0 / 180.0)

def test_delta_ratio_zero_volume():
    c = make_candle(total_delta=0.0)
    c.total_volume = 0.0
    assert c.delta_ratio == pytest.approx(0.0)

# absorption
def test_absorption_bullish_neg_delta():
    agent = FootprintAgent()
    c = make_candle()
    c.open=100.0; c.close=105.0; c.total_delta=-20.0; c.total_volume=100.0
    assert agent._detect_absorption(c) is True

def test_absorption_bearish_pos_delta():
    agent = FootprintAgent()
    c = make_candle()
    c.open=105.0; c.close=100.0; c.total_delta=20.0; c.total_volume=100.0
    assert agent._detect_absorption(c) is True

def test_no_absorption_confirmed():
    agent = FootprintAgent()
    c = make_candle()
    c.open=100.0; c.close=105.0; c.total_delta=30.0; c.total_volume=100.0
    assert agent._detect_absorption(c) is False

# exhaustion
def test_exhaustion_high():
    agent = FootprintAgent(tick_size=1.0)
    levels = [FootprintLevel(98.0, 5.0, 50.0, 45.0, "buying"),
              FootprintLevel(99.0, 5.0, 40.0, 35.0, "buying"),
              FootprintLevel(100.0, 60.0, 5.0, -55.0, "selling")]
    ex_h, ex_l = agent._detect_exhaustion(levels)
    assert ex_h is True

def test_exhaustion_low():
    agent = FootprintAgent(tick_size=1.0)
    levels = [FootprintLevel(98.0, 5.0, 60.0, 55.0, "buying"),
              FootprintLevel(99.0, 40.0, 5.0, -35.0, "selling"),
              FootprintLevel(100.0, 50.0, 5.0, -45.0, "selling")]
    ex_h, ex_l = agent._detect_exhaustion(levels)
    assert ex_l is True

def test_no_exhaustion_balanced():
    agent = FootprintAgent(tick_size=1.0)
    levels = [FootprintLevel(98.0,10.0,10.0,0.0,"neutral"),
              FootprintLevel(99.0,10.0,10.0,0.0,"neutral"),
              FootprintLevel(100.0,10.0,10.0,0.0,"neutral")]
    ex_h, ex_l = agent._detect_exhaustion(levels)
    assert ex_h is False and ex_l is False

# stacked imbalances
def test_stacked_buying():
    agent = FootprintAgent(tick_size=1.0)
    levels = [FootprintLevel(100.0,5.0,40.0,35.0,"buying"),
              FootprintLevel(101.0,5.0,40.0,35.0,"buying"),
              FootprintLevel(102.0,5.0,40.0,35.0,"buying"),
              FootprintLevel(103.0,30.0,5.0,-25.0,"selling")]
    count, zones = agent._count_stacked_imbalances(levels)
    assert count >= 3

def test_stacked_empty():
    agent = FootprintAgent()
    count, zones = agent._count_stacked_imbalances([])
    assert count == 0 and zones == []

def test_no_stacked_alternating():
    agent = FootprintAgent(tick_size=1.0)
    levels = [FootprintLevel(100.0,5.0,40.0,35.0,"buying"),
              FootprintLevel(101.0,40.0,5.0,-35.0,"selling"),
              FootprintLevel(102.0,5.0,40.0,35.0,"buying")]
    count, _ = agent._count_stacked_imbalances(levels)
    assert count < 3

# score_for_trade
def test_score_delta_confirms_long():
    agent = FootprintAgent()
    c = make_candle(total_delta=20.0)
    c.total_volume=100.0
    assert agent.score_for_trade(c, "long", 100.0) >= 10

def test_score_delta_diverges_long():
    agent = FootprintAgent()
    c = make_candle(total_delta=-20.0)
    c.total_volume=100.0
    assert agent.score_for_trade(c, "long", 500.0) <= -10

def test_score_absorption_bonus():
    agent = FootprintAgent()
    c = make_candle(absorption=True, total_delta=5.0)
    c.total_volume=100.0
    assert agent.score_for_trade(c, "long", 100.0) >= 20

def test_score_poc_near_entry():
    agent = FootprintAgent()
    c = make_candle(poc_price=100.2, total_delta=10.0)
    c.total_volume=100.0
    assert agent.score_for_trade(c, "long", 100.0) >= 10

def test_score_range():
    agent = FootprintAgent()
    for d in ("long","short"):
        for delta in (-50.0, 0.0, 50.0):
            c = make_candle(total_delta=delta); c.total_volume=100.0
            pts = agent.score_for_trade(c, d, 100.0)
            assert -30 <= pts <= 30

# fetch_recent_trades
def test_fetch_returns_list():
    agent = FootprintAgent(api_key="", api_secret="")
    assert isinstance(agent.fetch_recent_trades("BTCUSDT", 10), list)

def test_fetch_never_raises():
    agent = FootprintAgent(api_key="INVALID", api_secret="INVALID")
    try:
        result = agent.fetch_recent_trades("BTCUSDT", 5)
        assert isinstance(result, list)
    except Exception:
        pytest.fail("fetch_recent_trades raised")

# format_telegram
def test_format_contains_symbol():
    agent = FootprintAgent()
    assert "BTCUSDT" in agent.format_telegram(make_candle(), "BTCUSDT")

def test_format_contains_delta():
    agent = FootprintAgent()
    msg = agent.format_telegram(make_candle(total_delta=355.0), "BTC")
    assert "355" in msg

def test_format_contains_poc():
    agent = FootprintAgent()
    msg = agent.format_telegram(make_candle(poc_price=67450.0), "BTC")
    assert "POC" in msg and "67" in msg

def test_format_is_string():
    assert isinstance(FootprintAgent().format_telegram(make_candle()), str)

# to_decision_pts
def test_to_decision_pts_absorption():
    c = make_candle(absorption=True)
    assert c.to_decision_pts() >= 20

def test_to_decision_pts_stacked():
    c = make_candle(stacked=3)
    assert c.to_decision_pts() >= 15

def test_to_decision_pts_exhaustion_penalty():
    c = make_candle(exhaustion_high=True)
    assert c.to_decision_pts() <= -10


