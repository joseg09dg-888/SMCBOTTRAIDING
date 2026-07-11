"""
TDD tests for core/atomic_json.py -- BUG-AXI-STATE-TORN-WRITE regression.

memory/axi_select_state.json is shared by AxiSelectGuard/AxiSelectTracker/
AxiCapitalAdjuster, each doing read-modify-write with no atomicity. A crash
mid-write used to leave a truncated file, silently falling back to hardcoded
defaults on the next read. write_json_atomic must never leave a partial file
on disk, even if writing raises midway.
"""
import json
import os
import pytest

from core.atomic_json import read_json, write_json_atomic


def test_read_json_missing_file_returns_default(tmp_path):
    path = str(tmp_path / "nope.json")
    assert read_json(path, {"a": 1}) == {"a": 1}


def test_read_json_corrupt_file_returns_default(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert read_json(str(path), {"fallback": True}) == {"fallback": True}


def test_write_then_read_roundtrip(tmp_path):
    path = str(tmp_path / "sub" / "state.json")
    write_json_atomic(path, {"capital": 96013.0})
    assert read_json(path, {}) == {"capital": 96013.0}


def test_write_creates_parent_dir(tmp_path):
    path = str(tmp_path / "memory" / "state.json")
    write_json_atomic(path, {"x": 1})
    assert os.path.exists(path)


def test_write_overwrites_existing_file_fully(tmp_path):
    path = str(tmp_path / "state.json")
    write_json_atomic(path, {"a": 1, "b": 2})
    write_json_atomic(path, {"a": 99})
    assert read_json(path, {}) == {"a": 99}  # "b" is gone -- caller must merge

def test_no_leftover_temp_files_after_write(tmp_path):
    path = str(tmp_path / "state.json")
    write_json_atomic(path, {"a": 1})
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]
    assert leftovers == []

def test_failed_serialization_does_not_leave_temp_file(tmp_path, monkeypatch):
    path = str(tmp_path / "state.json")

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": Unserializable()})

    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]
    assert leftovers == []
    assert not os.path.exists(path)  # original write never landed
