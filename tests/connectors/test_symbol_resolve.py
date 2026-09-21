import types
import pytest

import connectors.metatrader_connector as mc


class _FakeMT5:
    def __init__(self, names):
        self._names = set(names)

    def symbol_info(self, name):
        return object() if name in self._names else None

    def symbols_get(self, pattern):
        prefix = pattern.rstrip("*")
        return [types.SimpleNamespace(name=n) for n in sorted(self._names) if n.startswith(prefix)]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(mc, "HAS_MT5", True)
    mc._SYMBOL_CACHE.clear()
    yield
    mc._SYMBOL_CACHE.clear()


def test_demo_plain_names_unchanged(monkeypatch):
    monkeypatch.setattr(mc, "mt5", _FakeMT5({"EURUSD", "USDCAD"}))
    assert mc.resolve_symbol("EURUSD") == "EURUSD"
    assert mc.plain_symbol("EURUSD") == "EURUSD"


def test_real_account_sa_suffix_resolved(monkeypatch):
    monkeypatch.setattr(mc, "mt5", _FakeMT5({"EURUSD.sa", "USDCAD.sa"}))
    assert mc.resolve_symbol("EURUSD") == "EURUSD.sa"
    assert mc.plain_symbol("EURUSD.sa") == "EURUSD"


def test_plain_symbol_strips_sa_even_when_not_cached():
    assert mc.plain_symbol("NZDUSD.sa") == "NZDUSD"


def test_unknown_symbol_returned_as_is(monkeypatch):
    monkeypatch.setattr(mc, "mt5", _FakeMT5({"EURUSD"}))
    assert mc.resolve_symbol("FOOBAR") == "FOOBAR"


def test_prefers_shortest_match(monkeypatch):
    monkeypatch.setattr(mc, "mt5", _FakeMT5({"EURUSD.sa", "EURUSD.sa.x"}))
    assert mc.resolve_symbol("EURUSD") == "EURUSD.sa"
