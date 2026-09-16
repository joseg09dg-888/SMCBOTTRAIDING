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
import time
from typing import Any


def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# BUG-ROLLOVER-SILENT-ABORT (2026-09-16): os.replace() on Windows throws
# [WinError 5] Acceso denegado when the destination file is momentarily
# locked (AV scan, another thread's read) -- confirmed 163 real occurrences
# in the live log. write_json_atomic re-raised unconditionally, and its only
# caller in the hot path (_manage_open_positions -> AxiSelectTracker.record_day)
# had no inner try/except, so the exception propagated to the function's
# top-level handler and silently aborted the ENTIRE position-management pass
# for that cycle -- including the Friday-close and Rollover-close safety
# blocks that run further down in the same function. This is the real reason
# ROLLOVER-CLOSE never fired on either of its two live test nights (2026-09-08
# and 2026-09-15): the lock is far more likely to hit exactly when multiple
# positions are closing back-to-back (the rollover spread-spike window
# itself), racing this same file. Fix: retry the atomic rename a few times
# with a short backoff before giving up -- the lock is transient (a few ms
# to tens of ms), not a real error.
def write_json_atomic(path: str, data: Any, _retries: int = 5, _delay: float = 0.05) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        last_exc = None
        for attempt in range(_retries):
            try:
                os.replace(tmp_path, path)
                return
            except OSError as exc:
                last_exc = exc
                if attempt < _retries - 1:
                    time.sleep(_delay)
        raise last_exc
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
