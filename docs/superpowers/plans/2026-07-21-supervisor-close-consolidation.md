# Supervisor close_position() Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the 14 duplicated `close_position()` call sites in `core/supervisor.py` (position peak cleanup + close call + Telegram alert, repeated near-identically 14 times) into one shared helper, without changing any trading behavior.

**Architecture:** Add one private async helper `_close_guarded(loop, ticket, reason, telegram_html, extra_cleanup=None)` on `TradingSupervisor` that does: call `self.mt5.close_position(ticket, reason)` via executor, pop `self._position_peaks[ticket]` on success, send the Telegram alert, return the `ok` bool. Each of the 14 call sites is replaced with a call to this helper, preserving its exact existing `reason` tag (IDX-NO-SL, META-DIA-SCALP, META-SWING, SCALP-DAY, SCALP-TP, SCALP-SL, SWING-STOP, FRIDAY-CLOSE, ANTI-DRAG, NO-SL-CLOSE, LOSS-LIMIT, PEAK-GUARD, STRUCT-INVALID, TIME-CLOSE-36H, STAGNANT) and its exact existing Telegram message text. No new abstraction beyond this one helper — per ponytail, do not generalize further than the 14 real call sites already need.

**Tech Stack:** Python 3.12, asyncio, pytest (existing suite, no new test framework).

**Precondition (do not start Task 1 until this is true):** the currently-running `code-review` workflow (`wf_ce95ae2e-67e`) has finished and its commit(s) are on `main`. Check with `git log --oneline -5` — if supervisor.py has uncommitted changes from that workflow, wait/commit those first. Editing the same file concurrently with that workflow will produce merge conflicts or silently clobber one set of changes.

---

### Task 1: Add the `_close_guarded` helper

**Files:**
- Modify: `core/supervisor.py` (add method near `_save_open_episodes`, ~line 2850, so it's near the other position-management private helpers)
- Test: `tests/core/test_supervisor_close_guarded.py` (new file)

- [ ] **Step 1: Write the failing test**

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


def _make_supervisor():
    from core.supervisor import TradingSupervisor
    sup = TradingSupervisor.__new__(TradingSupervisor)
    sup.mt5 = MagicMock()
    sup.telegram = MagicMock()
    sup.telegram.send_glint_alert = AsyncMock()
    sup._position_peaks = {123: 45.0}
    return sup


@pytest.mark.asyncio
async def test_close_guarded_success_pops_peak_and_alerts():
    sup = _make_supervisor()
    sup.mt5.close_position.return_value = True
    loop = asyncio.get_event_loop()

    ok = await sup._close_guarded(loop, 123, "PEAK-GUARD", "<b>test</b> msg")

    assert ok is True
    sup.mt5.close_position.assert_called_once_with(123, "PEAK-GUARD")
    assert 123 not in sup._position_peaks
    sup.telegram.send_glint_alert.assert_awaited_once_with("<b>test</b> msg")


@pytest.mark.asyncio
async def test_close_guarded_failure_keeps_peak_no_alert():
    sup = _make_supervisor()
    sup.mt5.close_position.return_value = False
    loop = asyncio.get_event_loop()

    ok = await sup._close_guarded(loop, 123, "PEAK-GUARD", "<b>test</b> msg")

    assert ok is False
    assert 123 in sup._position_peaks
    sup.telegram.send_glint_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_guarded_telegram_error_does_not_raise():
    sup = _make_supervisor()
    sup.mt5.close_position.return_value = True
    sup.telegram.send_glint_alert.side_effect = Exception("network down")
    loop = asyncio.get_event_loop()

    ok = await sup._close_guarded(loop, 123, "PEAK-GUARD", "<b>test</b> msg")

    assert ok is True  # close itself still succeeded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/core/test_supervisor_close_guarded.py -v`
Expected: FAIL with `AttributeError: 'TradingSupervisor' object has no attribute '_close_guarded'`

- [ ] **Step 3: Write minimal implementation**

Add this method to `core/supervisor.py`, near `_save_open_episodes` (~line 2850):

```python
    async def _close_guarded(self, loop, ticket: int, reason: str,
                              telegram_html: str) -> bool:
        """Shared close path for every position-management guard.
        SIMPLIFY-2026-07-21: replaces 14 near-identical inline blocks
        (close call + peak cleanup + Telegram alert) scattered across
        _manage_open_positions -- see docs/superpowers/plans/
        2026-07-21-supervisor-close-consolidation.md for the audit that
        found them. Behavior is unchanged: same close_position(ticket,
        reason) call, same peak-pop-on-success, same best-effort alert.
        """
        ok = await loop.run_in_executor(
            None, lambda t=ticket, r=reason: self.mt5.close_position(t, r)
        )
        if ok:
            self._position_peaks.pop(ticket, None)
            try:
                await self.telegram.send_glint_alert(telegram_html)
            except Exception:
                pass
        return ok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/core/test_supervisor_close_guarded.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/supervisor.py tests/core/test_supervisor_close_guarded.py
git commit -m "refactor: add _close_guarded helper for the 14 duplicated close-position call sites"
```

---

### Task 2: Replace each of the 14 call sites, one at a time, verifying after each

**Files:**
- Modify: `core/supervisor.py` (14 separate edits, one per guard)

For **each** of the 14 tags below, in this exact order (lowest risk / most isolated first):

`SCALP-TP`, `SCALP-SL`, `SCALP-DAY`, `IDX-NO-SL`, `NO-SL-CLOSE`, `META-DIA-SCALP`, `META-SWING`, `FRIDAY-CLOSE`, `ANTI-DRAG`, `SWING-STOP`, `LOSS-LIMIT`, `PEAK-GUARD`, `STRUCT-INVALID`, `TIME-CLOSE-36H`

- [ ] **Step 1 (repeat per tag): Find the call site**

Run: `grep -n "close_position(t.*\"<TAG>\"" core/supervisor.py` (substitute the tag)

- [ ] **Step 2 (repeat per tag): Replace the inline block with the helper call**

Example for `PEAK-GUARD` (apply the same shape to the other 13 — the existing `telegram_html` message string for that specific guard stays exactly as it is today, just passed as the 4th argument instead of being inlined after the close):

Before:
```python
                        ok = await loop.run_in_executor(
                            None, lambda t=ticket: self.mt5.close_position(t, "PEAK-GUARD")
                        )
                        if ok:
                            self._position_peaks.pop(ticket, None)
                            try:
                                await self.telegram.send_glint_alert(
                                    f"<b>GANANCIA ASEGURADA</b>\n{sym} #{ticket}\n"
                                    f"Peak: ${peak:.2f} → Retroceso 30% → cerrado en ${pnl:.2f}"
                                )
                            except Exception:
                                pass
```

After:
```python
                        ok = await self._close_guarded(
                            loop, ticket, "PEAK-GUARD",
                            f"<b>GANANCIA ASEGURADA</b>\n{sym} #{ticket}\n"
                            f"Peak: ${peak:.2f} → Retroceso 30% → cerrado en ${pnl:.2f}"
                        )
```

- [ ] **Step 3 (repeat per tag): Run the full test suite**

Run: `.venv\Scripts\python -m pytest tests/ -q`
Expected: `1516 passed` (same count as before this task started — if it changed, stop and investigate before continuing to the next tag)

- [ ] **Step 4 (repeat per tag): Commit that one tag's change**

```bash
git add core/supervisor.py
git commit -m "refactor: use _close_guarded for <TAG> close site"
```

**After all 14 tags are done:**

- [ ] **Step 5: Confirm no inline pattern remains**

Run: `grep -n "self.mt5.close_position(t" core/supervisor.py`
Expected: zero matches (every call now goes through `self._close_guarded`)

- [ ] **Step 6: Live sanity check before considering the bot restart**

Do NOT restart the bot as part of this plan. Once Task 2 is fully committed, hand back to the user/main session to decide on restart timing — this plan's scope ends at "code compiles, tests pass, no behavior change verified by identical test count."

---

## Explicitly out of scope for this plan (do not do these here)

- The Silver Bullet ICT gate (documented as never firing in 2 years of backtest data) is **not dead code** — it still runs a real check every cycle at hour 14 UTC. Removing it changes behavior (removes a real, if rarely-triggered, gate) and needs its own separate decision, not a "simplification."
- Any change to guard *thresholds* or *logic* (STAGNANT_HOURS, PEAK_GUARD_MIN, etc.) — this plan only touches the mechanical close-call duplication, not the decision logic.
- Consolidating the 6 scan-loop score/threshold computations (KZ multiplier, LEARN-THR, ADAPT-THR) — that logic is not duplicated the way close_position is; leave it alone (YAGNI, per ponytail — don't refactor what isn't actually repeated).
