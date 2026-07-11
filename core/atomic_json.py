"""
Atomic JSON read/write helper.

BUG-AXI-STATE-TORN-WRITE (2026-07-10): AxiSelectGuard/AxiSelectTracker/
AxiCapitalAdjuster all share memory/axi_select_state.json, each doing a
plain read-modify-write with no atomicity. A crash or pm2 restart mid-write
left a truncated/corrupt file, which the next process's json.load() failed
to parse -- silently falling back to hardcoded defaults (capital=$500)
and re-detecting a huge fake "capital escalated" jump. Write via a temp
file + os.replace so a write is never observed half-done.
"""
from __future__ import annotations
import json
import os
import tempfile
from typing import Any


def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json_atomic(path: str, data: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
