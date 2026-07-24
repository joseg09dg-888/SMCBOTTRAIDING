from datetime import datetime, timedelta, timezone
import asyncio
import json
import os

from core.supervisor_constants import (
    MT5_REAL_SCORE_THRESHOLD,
    MT5_SCORE_AUTO_REDUCE,
    MAX_OPEN_POSITIONS,
    DAILY_PROFIT_TARGET,
    INITIAL_CAPITAL,
    RECOVERY_SCALP_TP,
    RECOVERY_SCALP_SL,
    RECOVERY_TRIGGER_LOSS,
    ACCEL_TRIGGER_PROFIT,
    ACCEL_SCALP_TP,
    ACCEL_SCALP_SL,
    ACCEL_MAX_SCALPS,
)


class PositionGuardsMixin:
    """Position-lifecycle guards: close helper, orphaned-episode recovery,
    adaptive score threshold, and the full open-position management loop
    (PEAK-GUARD, STAGNANT, LOSS-LIMIT, SWING-STOP, FRIDAY-CLOSE, ANTI-DRAG,
    STRUCT-INVALID, IDX-NO-SL, NO-SL-CLOSE, META-DIA-SCALP, META-SWING,
    SCALP-TP/SCALP-SL/SCALP-DAY).

    SIMPLIFY-2026-07-23: extracted out of TradingSupervisor (was ~700
    lines of a single 4386-line file) into its own mixin so the guard
    logic is a self-contained, independently readable unit. No behavior
    change -- every method body is unchanged, methods still resolve via
    self.* exactly as before since TradingSupervisor inherits this mixin.
    """

    async def _close_guarded(self, loop, ticket: int, reason: str,
                              telegram_html: str = None) -> bool:
        """Shared close path for every position-management guard.
        SIMPLIFY-2026-07-21: replaces 14 near-identical inline blocks
        (close call + peak cleanup + Telegram alert) scattered across
        _manage_open_positions -- see docs/superpowers/plans/
        2026-07-21-supervisor-close-consolidation.md for the audit that
        found them. Behavior is unchanged: same close_position(ticket,
        reason) call, same peak-pop-on-success, same best-effort alert.
        telegram_html=None for the 2 batch-close call sites (META-DIA-SCALP,
        META-SWING) that already send one alert for the whole batch after
        the loop instead of per-position -- consolidating the close+peak-pop
        there without forcing a per-position alert that never existed.
        """
        ok = await loop.run_in_executor(
            None, lambda t=ticket, r=reason: self.mt5.close_position(t, r)
        )
        if ok:
            self._position_peaks.pop(ticket, None)
            if telegram_html:
                try:
                    await self.telegram.send_glint_alert(telegram_html)
                except Exception:
                    pass
        return ok

    def _recover_orphaned_episodes(self) -> None:
        """On startup: backfill outcomes for tickets that closed during a prior restart.

        BUG-ORPHAN-SILENT-DROP (2026-07-20): both branches below used to
        unconditionally add the ticket to `removed` -- even when the closing
        deal wasn't found, and even when update_episode_result raised (the
        exception was swallowed by a bare `except: pass` with no log line).
        Either path stopped tracking the ticket forever, leaving its
        episodes.db row with result=NULL permanently and silently -- no
        warning, no retry on the next restart. Found via 2 real orphaned
        rows (EURAUD #76484092, GBPCAD #76484444, both closed 2026-07-17
        with deals that WERE present in MT5 history when queried manually
        3 days later) that episodes.db never recorded. Fix: only drop a
        ticket from tracking once its result is actually persisted; keep
        retrying indefinitely otherwise, with a visible warning each time
        so a repeat failure is never silent again.
        """
        try:
            import MetaTrader5 as _mt5_oe
            from datetime import timedelta
            from memory.episodic_db import update_episode_result
            open_pos = _mt5_oe.positions_get() or []
            open_tickets = {p.ticket for p in open_pos}
            orphaned = {t: eid for t, eid in self._open_episodes.items()
                        if t not in open_tickets}
            if not orphaned:
                return
            now = datetime.now(timezone.utc)
            deals = _mt5_oe.history_deals_get(now - timedelta(days=90), now) or []
            closing = {d.position_id: d for d in deals if d.entry == 1}
            removed = []
            for ticket, episode_id in orphaned.items():
                d = closing.get(ticket)
                if not d:
                    print(f"[LEARN] orphan ticket={ticket}: sin deal de cierre en 90 dias -- se reintenta en el proximo restart", flush=True)
                    continue
                pnl = round(d.profit + d.swap + d.commission, 2)
                result = "WIN" if pnl > 0 else "LOSS"
                try:
                    update_episode_result(
                        episode_id,
                        exit_price=d.price, pnl=pnl, result=result,
                        lesson=f"Backfill: {result} PnL={pnl:+.2f}",
                        conn=self._episodic_conn,
                    )
                    print(f"[LEARN] backfill ticket={ticket} -> {result} pnl={pnl:+.2f}", flush=True)
                    removed.append(ticket)
                except Exception as _upd_exc:
                    print(f"[LEARN] orphan ticket={ticket}: update_episode_result fallo ({_upd_exc}) -- se reintenta en el proximo restart", flush=True)
            for t in removed:
                self._open_episodes.pop(t, None)
            if removed:
                self._save_open_episodes()
        except Exception as _e:
            print(f"[LEARN] orphan recovery error: {_e}", flush=True)

    def _adaptive_threshold(self) -> int:
        """
        Calcula threshold dinamico basado en win rate de ultimos 10 trades reales.
        Cuanto peor el rendimiento reciente, mas selectivo se vuelve el bot.
        """
        try:
            from datetime import timedelta
            desde = datetime.now(timezone.utc) - timedelta(days=14)
            hasta = datetime.now(timezone.utc)
            import MetaTrader5 as _mt5
            deals = _mt5.history_deals_get(desde, hasta)
            # Solo contar SWINGS (vol > 0.1L) — los micro-scalps sesgan el WR
            closed = [d for d in (deals or []) if d.type in (0, 1) and d.entry == 1 and d.volume > 0.10]
            recent = sorted(closed, key=lambda d: d.time)[-10:]
            if len(recent) < 3:
                # Pocos datos → threshold moderado pero no paralizar
                thr = MT5_SCORE_AUTO_REDUCE
                print(f"[ADAPT-THR] datos insuficientes ({len(recent)} trades) → threshold={thr}", flush=True)
                return thr
            wins = sum(1 for d in recent if d.profit > 0)
            wr   = wins / len(recent)
            # Recalibrado 2026-07-01 tras barrido thr x RR sobre 2 anos reales (EURUSD+USDCAD):
            # thr=80/RR=3.0 -> WR=41.7%, P(pasar 5% mensual)=28.4% (vs 8.5% con 90-95)
            # thr=95 NO demostro mejor calidad que 80 en los datos -- solo menos volumen.
            # Igual se mantiene selectividad dinamica: peor WR reciente -> mas estricto.
            if wr >= 0.65:
                thr = MT5_SCORE_AUTO_REDUCE - 2       # 78 — buen WR reciente, el mas permisivo
            elif wr >= 0.55:
                thr = MT5_SCORE_AUTO_REDUCE           # 80 — punto optimo del backtest
            elif wr >= 0.40:
                thr = MT5_SCORE_AUTO_REDUCE + 5        # 85
            else:
                thr = MT5_SCORE_AUTO_REDUCE + 10       # 90 — WR<40%, maxima selectividad (sin llegar al 95 que no aporta segun el backtest)
            print(f"[ADAPT-THR] WR={wr*100:.0f}% → threshold={thr} (optimo backtest=80)", flush=True)
            return thr
        except Exception as _e:
            return MT5_REAL_SCORE_THRESHOLD  # fallback al default

    async def _manage_open_positions(self):
        """
        Active position management:
        0a. Friday pre-close: close ALL open positions (winners + losers) by 19:30 UTC Friday (before 21:00 close) — avoids weekend gap risk on both sides
        0b. Anti-drag: close worst loser when net P&L is negative and loser > winner
        1. Auto-close on loss > 0.8% balance
        1b. Peak-profit retracement: close when profit falls 25% from peak (peak >= $20)
        2. Move SL to breakeven when profit >= 1R (SL distance)
        3. Trail SL at 1R below/above price when profit >= 2R
        4. Hard-close LOSING positions stuck > 36h (prevents directionless drains)
        """
        MAX_HOLD_HOURS = 36  # only close positions that are losing after 36h
        FRIDAY_CLOSE_HOUR = 19   # UTC — close losers by 19:30 UTC to avoid weekend gap risk
        FRIDAY_CLOSE_MIN  = 30
        try:
            loop = asyncio.get_running_loop()
            positions = await loop.run_in_executor(None, self.mt5.get_positions)
            import MetaTrader5 as _mt5
            from datetime import timezone as _tz

            bal       = self._risk_gate_state.current_balance or self.capital
            limit_usd = bal * 0.008  # 0.8% = emergency stop per position

            # ── 0. Daily profit target ────────────────────────────────────────
            # $245 = meta mínima diaria → notifica → bot sigue para más ganancia
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_pnl_date != today_utc:
                self._daily_pnl_date    = today_utc
                self._daily_target_hit  = False
                self._daily_protect_hit = False

            # Sync realized PnL from MT5 — acumula todos los trades cerrados hoy
            mt5_daily = await loop.run_in_executor(None, self.mt5.get_daily_pnl)
            if mt5_daily is not None:
                self._daily_realized_pnl = float(mt5_daily)
                self._axi_tracker.record_day(self._daily_realized_pnl, capital=bal)
                # Si el PnL REAL está bajo el target (ej: ganó $277 luego NAS100 -$212 → $65)
                # el flag se resetea para que el bot pueda seguir operando y llegar a $250
                if self._daily_target_hit and self._daily_realized_pnl < DAILY_PROFIT_TARGET:
                    self._daily_target_hit = False
                    print(f"[META-RESET] PnL=${self._daily_realized_pnl:.2f} bajo meta $250 — bot puede operar swings", flush=True)

            # ── Swing profit target: cierra SWINGS cuando llegan a $245 juntos ──
            # Los scalps siguen corriendo para acumular más ganancia.
            # Clasificar por VOLUMEN — más confiable que distancia de TP
            # Scalp: vol <= 0.1L (todos los trades pequeños, cerrar en +$10/-$4)
            # Swing: vol > 0.1L (trades grandes, cerrar cuando swing_float >= $245)
            scalp_positions = [p for p in (positions or []) if p.get("volume", 0) <= 0.10]
            swing_positions = [p for p in (positions or []) if p.get("volume", 0) > 0.10]
            swing_float = sum(p.get("profit", 0.0) for p in swing_positions)
            float_pnl   = sum(p.get("profit", 0.0) for p in (positions or []))
            total_today = self._daily_realized_pnl + float_pnl

            # META = realizados + flotantes >= $250 — el bot incluye lo ya ganado hoy
            _total_combined = self._daily_realized_pnl + swing_float
            if not self._daily_target_hit and _total_combined >= DAILY_PROFIT_TARGET:
                self._daily_target_hit = True
                print(
                    f"[META-SWING] realizado=${self._daily_realized_pnl:.2f} + float=${swing_float:.2f}"
                    f" = ${_total_combined:.2f} >= ${DAILY_PROFIT_TARGET:.0f}"
                    f" — META CUMPLIDA, cerrando SWINGS",
                    flush=True,
                )
                for sp in list(swing_positions):
                    t_ticket = sp["ticket"]
                    t_sym    = sp.get("symbol", "?")
                    t_pnl    = sp.get("profit", 0.0)
                    ok = await self._close_guarded(loop, t_ticket, "META-SWING")
                    if ok:
                        print(f"[META-CLOSE-SWING] {t_sym} #{t_ticket} cerrado ${t_pnl:+.2f}", flush=True)
                try:
                    await self.telegram.send_glint_alert(
                        f"<b>META DIARIA CUMPLIDA — SWINGS CERRADOS</b>\n"
                        f"Swing profit: <b>${swing_float:.2f}</b>\n"
                        f"Scalps siguen operando para mas ganancia."
                    )
                except Exception:
                    pass

            if not positions:
                return  # nada que gestionar

            # ── 0. Scalp gestión de P&L ───────────────────────────────────────
            # Modo Recuperación — dos condiciones:
            #   1. Dia en rojo > $50
            #   2. Balance debajo del capital base $100K (recuperar lo perdido)
            _current_bal   = self._risk_gate_state.current_balance or self.capital
            # Actualizar high-water mark (balance máximo histórico)
            if _current_bal > self._balance_peak:
                self._balance_peak = _current_bal
                print(f"[PEAK] Nuevo máximo histórico: ${self._balance_peak:,.2f}", flush=True)

            # Recovery solo por pérdida real del día — _below_peak eliminado porque
            # risk-gate inicia current_balance=$100K causando falso drawdown desde día 1
            # La protección multi-día ya la cubre: RiskGovernor + risk-gate drawdown limits
            _day_in_loss      = self._daily_realized_pnl <= RECOVERY_TRIGGER_LOSS
            _in_recovery      = _day_in_loss and not self._scalp_daily_hit

            # Estrategia 5: Modo Aceleración — dia muy bueno → maximizar
            _in_accel = (
                self._daily_realized_pnl >= ACCEL_TRIGGER_PROFIT and
                _current_bal >= self.capital * 0.98 and
                not _in_recovery and
                not self._scalp_daily_hit
            )

            if _in_recovery:
                SCALP_MIN_PROFIT = RECOVERY_SCALP_TP
                SCALP_MAX_LOSS   = RECOVERY_SCALP_SL
                # Log RECOVERY solo cada 300 ciclos (~30s) para no spam con monitor 100ms
                if getattr(self, "_recovery_log_count", 0) % 300 == 0:
                    if self._balance_peak > INITIAL_CAPITAL and _current_bal < self._balance_peak:
                        _gap = self._balance_peak - _current_bal
                        print(f"[RECOVERY] Cayó ${_gap:.0f} del pico ${self._balance_peak:,.0f} — recuperando", flush=True)
                    elif _current_bal < INITIAL_CAPITAL:
                        print(f"[RECOVERY] Balance ${_current_bal:,.0f} bajo $100K — recuperando capital base", flush=True)
                    else:
                        print(f"[RECOVERY] Dia ${self._daily_realized_pnl:.2f} — recuperando el dia", flush=True)
                self._recovery_log_count = getattr(self, "_recovery_log_count", 0) + 1
            elif _in_accel:
                SCALP_MIN_PROFIT = ACCEL_SCALP_TP   # +$15
                SCALP_MAX_LOSS   = ACCEL_SCALP_SL   # -$4
                print(f"[ACCEL] Dia +${self._daily_realized_pnl:.2f} — modo aceleracion activo (TP=$15 max={ACCEL_MAX_SCALPS})", flush=True)
            else:
                SCALP_MIN_PROFIT =  10.0
                SCALP_MAX_LOSS   =  -4.0
            SCALP_DAILY_TARGET =  60.0   # cerrar TODOS scalps cuando acumula $60 hoy

            # Sincronizar scalp P&L desde MT5 real cada ciclo — no confiar en contador en memoria
            _today_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._scalp_pnl_date != _today_s:
                self._scalp_pnl_date   = _today_s
                self._scalp_daily_hit  = False
                self._scalp_peak_today = 0.0
            _scalp_mt5 = await loop.run_in_executor(None, self.mt5.get_scalp_daily_pnl)
            if _scalp_mt5 is not None:
                self._scalp_realized_today = float(_scalp_mt5)

            # Actualizar peak diario de scalps
            if self._scalp_realized_today > self._scalp_peak_today:
                self._scalp_peak_today = self._scalp_realized_today

            # Trailing lock por milestones:
            # Peak $65 → lock=$60. Peak $110 → lock=$100. Peak $250 → lock=$200.
            # Si current cae AL lock → cerrar todo (asegurar ese nivel)
            _SCALP_MILESTONES = [60, 100, 200, 300, 500, 750, 1000]
            _scalp_lock = 0
            for _m in _SCALP_MILESTONES:
                if self._scalp_peak_today >= _m:
                    _scalp_lock = _m
            if (_scalp_lock > 0 and
                    self._scalp_realized_today <= _scalp_lock and
                    self._scalp_peak_today > _scalp_lock and
                    not self._scalp_daily_hit):
                self._scalp_daily_hit = True
                print(f"[SCALP-LOCK] Peak=${self._scalp_peak_today:.2f} cayó a ${self._scalp_realized_today:.2f} <= lock=${_scalp_lock} — cerrando todo", flush=True)

            # Meta diaria scalp $60 alcanzada → cerrar todos los scalps abiertos
            if self._scalp_daily_hit and scalp_positions:
                for sp in list(scalp_positions):
                    ok = await self._close_guarded(loop, sp["ticket"], "SCALP-DAY")
                    if ok:
                        print(f"[SCALP-DAY] {sp.get('symbol','?')} cerrado ${sp.get('profit',0):+.2f} (meta $60 scalp cumplida)", flush=True)
            else:
                # Gestión individual de cada scalp
                for sp in list(scalp_positions):
                    sp_pnl    = sp.get("profit", 0.0)
                    sp_ticket = sp["ticket"]
                    sp_sym    = sp.get("symbol", "?")
                    if sp_pnl >= SCALP_MIN_PROFIT:
                        ok = await self._close_guarded(loop, sp_ticket, "SCALP-TP")
                        if ok:
                            self._scalp_realized_today += sp_pnl
                            print(f"[SCALP-TP] {sp_sym} #{sp_ticket} ${sp_pnl:+.2f} | total scalp hoy=${self._scalp_realized_today:.2f}", flush=True)
                            if not self._scalp_daily_hit and self._scalp_realized_today >= SCALP_DAILY_TARGET:
                                self._scalp_daily_hit = True
                                print(f"[SCALP-META] ${self._scalp_realized_today:.2f} >= $60 — META SCALP CUMPLIDA", flush=True)
                                try:
                                    await self.telegram.send_glint_alert(
                                        f"<b>META SCALP DIARIA $60 CUMPLIDA</b>\n"
                                        f"Total scalps hoy: <b>${self._scalp_realized_today:.2f}</b>\n"
                                        f"Cerrando scalps restantes. Swing sigue."
                                    )
                                except Exception:
                                    pass
                    elif sp_pnl <= SCALP_MAX_LOSS:
                        ok = await self._close_guarded(loop, sp_ticket, "SCALP-SL")
                        if ok:
                            self._scalp_realized_today += sp_pnl
                            print(f"[SCALP-SL] {sp_sym} #{sp_ticket} ${sp_pnl:+.2f} | total scalp hoy=${self._scalp_realized_today:.2f}", flush=True)

            # ── 0a. Friday pre-close: dump ALL losers before weekend ──────────
            # ── 0b-TRAIL: Swing trailing — SL físico en MT5 al superar peak proporcional al riesgo ─
            # Problema: bot cicla cada 30s, precio puede caer de +$18 a +$1 en segundos
            # Solución: cuando peak >= umbral → mover SL MT5 a entry (breakeven) inmediato
            # Esto protege a nivel broker, independiente del ciclo del bot
            # BUG-BE-TOO-EARLY (2026-07-03): umbral fijo de $10 disparaba en cuanto el
            # precio hacia ruido normal, mucho antes del 1:1 RR real (~$275-390 en estas
            # posiciones) donde el mecanismo de partial-close (linea ~2438) esta pensado
            # para actuar. Resultado: dos trades reales (USDCAD, NZDUSD) que llegaron a
            # peak $30-90 volvieron a breakeven exacto ($0.00 neto) sin nunca tener
            # oportunidad de alcanzar el partial ni el TP. Fix original: umbral = 50% del
            # riesgo real de la posicion (via SL distance), con piso de $10.
            #
            # BUG-BE-STILL-TOO-EARLY (2026-07-09): auditoria de episodes.db mostro avg
            # WIN=$27.72 vs avg LOSS=$22.02 (ratio real 1.26x) pese a que el sistema esta
            # calculado para RR=3.0 -- decenas de "wins" de $0.24-$8 en el historial son
            # exactamente el patron de romper a breakeven apenas a mitad de camino del
            # riesgo real. El umbral de 50% de 1R seguia siendo demasiado temprano -- el
            # propio comentario original de esta funcion (arriba) dice "mover a breakeven
            # cuando profit >= 1R", pero el codigo aplicaba 0.5R. Subido a 1.0R real para
            # que las ganadoras tengan el espacio que el docstring siempre dijo que debian
            # tener antes de asegurar breakeven.
            #
            # BUG-BE-KILLS-RR (2026-07-17): auditoria completa de episodes.db (593 trades
            # cerrados) confirmo que el problema persistia incluso con el umbral en 1.0R:
            # avg WIN=$27.72 vs avg LOSS=$20.15 -- RR real 1.38, contra el RR=3.0 de diseno.
            # Causa mecanica: al llegar a 1R el SL saltaba a entry EXACTO (0R). Cualquier
            # ruido normal de precio que retrocede hasta entry -- algo que pasa en casi
            # toda posicion que ya estuvo en +1R -- saca el trade en ~$0 antes de que
            # tenga oportunidad de acercarse al TP real (3R). Fix: subir el umbral a 2.0R
            # (mas espacio antes de tocar el stop) y, al dispararse, asegurar 0.5R de
            # ganancia en vez de breakeven exacto -- protege capital sin borrar la ganancia
            # entera en el primer retroceso.
            _be_moved = self.__dict__.setdefault("_breakeven_set", set())
            for sw in list(swing_positions):
                sw_pnl    = sw.get("profit", 0.0)
                sw_ticket = sw["ticket"]
                sw_sym    = sw.get("symbol", "?")
                sw_entry  = sw.get("price_open", 0.0)
                sw_type   = sw.get("type", "BUY")
                sw_tp     = sw.get("tp", 0.0)
                sw_sl     = sw.get("sl", 0.0)
                # Actualizar peak
                _peak = self._position_peaks.get(sw_ticket, 0.0)
                if sw_pnl > _peak:
                    self._position_peaks[sw_ticket] = sw_pnl
                    _peak = sw_pnl
                # Umbral proporcional al riesgo real (2.0R -- ver BUG-BE-KILLS-RR
                # arriba), piso $10
                _be_trigger = 10.0
                _be_sl_dist = 0.0
                if sw_sl and sw_entry:
                    try:
                        from core.volume_calculator import VolumeCalculator as _VC
                        _be_base      = _VC._norm(sw_sym)
                        _be_pip_size  = _VC._PIP_SIZE.get(_be_base, 0.0001)
                        _be_pip_value = _VC._PIP_VALUE.get(_be_base, 10.0)
                        _be_sl_dist   = abs(sw_entry - sw_sl)
                        _be_sl_pips   = _be_sl_dist / _be_pip_size if _be_pip_size else 0.0
                        _be_risk_usd  = sw.get("volume", 0.0) * _be_sl_pips * _be_pip_value
                        if _be_risk_usd > 0:
                            _be_trigger = max(10.0, _be_risk_usd * 2.0)
                    except Exception:
                        pass
                # Cuando peak >= umbral → asegurar 0.5R de ganancia en MT5 (no breakeven
                # exacto -- ver BUG-BE-KILLS-RR), una sola vez
                if _peak >= _be_trigger and sw_ticket not in _be_moved and sw_entry > 0:
                    _be_lock_price = sw_entry
                    if _be_sl_dist > 0:
                        _be_offset = _be_sl_dist * 0.5
                        _be_lock_price = sw_entry + _be_offset if sw_type == "BUY" else sw_entry - _be_offset
                    _be_ok = await loop.run_in_executor(
                        None, lambda t=sw_ticket, e=_be_lock_price, tp=sw_tp: self.mt5.modify_position_sl_tp(t, e, tp)
                    )
                    if _be_ok:
                        _be_moved.add(sw_ticket)
                        print(f"[BE-SET] {sw_sym} #{sw_ticket} peak=${_peak:.2f} → SL MT5 movido a +0.5R {_be_lock_price:.5f}", flush=True)
                # BUG-DOUBLE-PEAK-GUARD (2026-07-13): este cierre por software
                # (peak>=$100, retrocede 50%) duplicaba al guardia PEAK-GUARD de
                # mas abajo (linea ~3376: peak>=$200, retrocede 30%), pero con un
                # umbral MUCHO mas bajo -- por eso este disparaba primero y
                # aseguraba apenas $50 de una posicion con peak $100, sin dejarle
                # nunca la oportunidad de llegar al TP real (a veces +$200-400).
                # Auditoria de episodes.db: max win historico jamas alcanzado
                # fue $279.04 pese a TPs disenados para RR=3.0 -- este guardia
                # duplicado y prematuro es la causa mecanica. Eliminado; el
                # breakeven-set de arriba (linea 3181) ya protege el capital sin
                # cortar la ganancia, y PEAK-GUARD mas abajo protege peaks
                # grandes con un umbral realista.

            # ── 0b. Swing dollar-stop: solo cierre de emergencia — dejar que SL del broker actúe
            # No cerrar antes del SL: los swings USDCAD necesitan tiempo para llegar al TP (+$247)
            # El broker SL ya protege el capital. Auto-close solo si falla el broker SL (emergencia).
            SWING_MAX_LOSS = -150.0  # emergencia si broker SL falla — no interferir antes
            for sw in list(swing_positions):
                sw_pnl    = sw.get("profit", 0.0)
                sw_ticket = sw["ticket"]
                sw_sym    = sw.get("symbol", "?")
                sw_auto_close = sw_pnl <= SWING_MAX_LOSS  # solo emergencia — NO cerrar por tiempo
                if sw_auto_close:
                    ok = await self._close_guarded(
                        loop, sw_ticket, "SWING-STOP",
                        f"<b>SWING STOP -$50</b>\n{sw_sym} #{sw_ticket}\nCerrado en ${sw_pnl:.2f}"
                    )
                    if ok:
                        # Registrar cooldown: no reabrir este par/dirección por 4 horas
                        import time as _t; _sw_dir = "BUY" if sw.get("type") == "BUY" else "SELL"
                        self._symbol_sl_time[f"{sw_sym}_{_sw_dir}"] = _t.time()
                        try:
                            json.dump(self._symbol_sl_time, open(os.path.join("memory", "sl_cooldown_state.json"), "w"))
                        except Exception:
                            pass
                        print(f"[SWING-STOP] {sw_sym} #{sw_ticket} cerrado ${sw_pnl:+.2f} → COOLDOWN 4h (persistido)", flush=True)

            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() == 4:  # Friday
                past_cutoff = (now_utc.hour > FRIDAY_CLOSE_HOUR or
                               (now_utc.hour == FRIDAY_CLOSE_HOUR and now_utc.minute >= FRIDAY_CLOSE_MIN))
                if past_cutoff:
                    # Cierra TODO antes del fin de semana — ganadoras y perdedoras.
                    # Antes solo cerraba perdedoras; una ganadora abierta el viernes
                    # queda expuesta al mismo gap risk que una perdedora el lunes.
                    for lp in list(positions):
                        sym    = lp.get("symbol", "?")
                        ticket = lp["ticket"]
                        pnl    = lp.get("profit", 0.0)
                        estado = "perdiendo" if pnl < 0 else "ganando"
                        print(
                            f"[FRIDAY-CLOSE] {sym} #{ticket} {estado} ${pnl:.2f} "
                            f"— cerrando antes del fin de semana (19:30 UTC)",
                            flush=True,
                        )
                        ok = await self._close_guarded(
                            loop, ticket, "FRIDAY-CLOSE",
                            f"<b>CIERRE VIERNES</b> {sym} #{ticket}\n"
                            f"Cerrado antes del fin de semana.\n"
                            f"P&amp;L: ${pnl:.2f}"
                        )
                        if not ok:
                            print(f"[FRIDAY-CLOSE] ERROR cerrando {sym} #{ticket}", flush=True)
                    # Reload positions after Friday cleanup
                    positions = await loop.run_in_executor(None, self.mt5.get_positions)
                    if not positions:
                        return

            # ── 0. Anti-drag: ONLY for positions WITHOUT a proper SL ────────
            # Positions WITH a SL are already protected — don't override MT5's SL/TP.
            # Anti-drag only fires for positions missing SL (shouldn't happen, but safety net).
            NET_DRAG_THRESHOLD = -20.0
            MIN_DRAG_LOSS_USD  = -35.0
            all_pnls   = [p.get("profit", 0.0) for p in positions]
            net_pnl    = sum(all_pnls)
            total_wins = sum(x for x in all_pnls if x > 0)
            worst_loss = min(all_pnls) if all_pnls else 0.0

            # Only fire anti-drag for positions WITHOUT a SL — ones WITH SL are safe
            positions_no_sl = [p for p in positions if p.get("sl", 0.0) == 0.0]
            if (positions_no_sl
                    and net_pnl < NET_DRAG_THRESHOLD
                    and worst_loss < MIN_DRAG_LOSS_USD
                    and abs(worst_loss) > total_wins):
                # Find the position with the worst loss (among those without SL)
                drag_pos = min(positions_no_sl, key=lambda p: p.get("profit", 0.0))
                drag_ticket = drag_pos["ticket"]
                drag_sym    = drag_pos.get("symbol", "?")
                drag_pnl    = drag_pos.get("profit", 0.0)
                print(
                    f"[ANTI-DRAG] Neto abierto=${net_pnl:+.2f} | "
                    f"{drag_sym} #{drag_ticket} perdiendo ${drag_pnl:.2f} > ganadores ${total_wins:.2f} "
                    f"→ cerrando perdedora para proteger ganancias",
                    flush=True,
                )
                ok = await self._close_guarded(
                    loop, drag_ticket, "ANTI-DRAG",
                    f"<b>ANTI-DRAG CLOSE</b>\n{drag_sym} #{drag_ticket}\n"
                    f"Perdida ${abs(drag_pnl):.2f} cancelaba ganancias (neto ${net_pnl:+.2f})\n"
                    f"→ Perdedora cerrada. Ganadoras protegidas."
                )
                if ok:
                    # Reload positions after close
                    positions = await loop.run_in_executor(None, self.mt5.get_positions)
                    if not positions:
                        return

            for p in positions:
                pnl    = p.get("profit", 0.0)
                ticket = p["ticket"]
                sym    = p.get("symbol", "?")
                entry  = p.get("price_open", 0.0)
                sl_cur = p.get("sl", 0.0)
                tp_cur = p.get("tp", 0.0)
                pos_type = p.get("type", "BUY")
                is_buy = pos_type == "BUY"
                open_time = p.get("time", 0)  # Unix timestamp

                # ── 0b. No SL protection: retry setting SL if it's missing ───
                if sl_cur == 0.0 and tp_cur > 0 and entry > 0:
                    await asyncio.sleep(1.0)
                    ok = await loop.run_in_executor(
                        None,
                        lambda t=ticket, e=entry, tp=tp_cur, ib=is_buy:
                            self.mt5.modify_position_sl_tp(
                                t,
                                round(e * (0.995 if ib else 1.005), 5),
                                tp,
                            )
                    )
                    if ok:
                        print(
                            f"[SL-RETRY] {sym} #{ticket} SL aplicado en diferido",
                            flush=True,
                        )
                    else:
                        # If SL still can't be set, close the position to protect capital
                        if pnl < -limit_usd * 0.3:  # at 30% of normal limit
                            print(
                                f"[NO-SL CLOSE] {sym} #{ticket} sin SL y perdiendo ${abs(pnl):.2f} → cerrando",
                                flush=True,
                            )
                            await self._close_guarded(loop, ticket, "NO-SL-CLOSE")
                            continue

                # ── 1. Loss protection ─────────────────────────────────────
                # BUG-LOSS-LIMIT-FLAT (2026-07-21): used to be one flat
                # 0.8%-of-balance threshold (limit_usd) for every trade
                # regardless of intended risk_pct.
                # BUG-INTENDED-RISK-LOST-ON-RESTART (2026-07-23): the fix above
                # stored intended risk in `self._position_intended_risk`, an
                # in-memory dict populated only at order-open time. ANY pm2
                # restart wipes it -- every position that was already open
                # silently falls back to the flat 0.8%-of-balance limit again
                # (~$760 on a $95K account), regardless of how small its real
                # intended risk was. Confirmed live: EURAUD #78900206 lost its
                # entry from before today's restart and bled from -$49 to
                # -$85+ with no scaled backstop active. Fixed by computing
                # intended risk directly from the position's own live
                # volume/entry/SL (broker truth, survives every restart)
                # instead of trusting fragile session memory. Falls back to
                # the in-memory value, then the flat limit, only if the
                # position has no SL set yet (can't compute risk).
                _intended = None
                if sl_cur and sl_cur > 0 and entry > 0:
                    try:
                        from core.volume_calculator import VolumeCalculator as _VC
                        _base_sym    = _VC._norm(sym)
                        _pip_size    = _VC._PIP_SIZE.get(_base_sym, 0.0001)
                        _pip_value   = _VC._PIP_VALUE.get(_base_sym, 10.0)
                        _sl_dist_cur = abs(entry - sl_cur)
                        _intended    = p.get("volume", 0.0) * (_sl_dist_cur / _pip_size) * _pip_value
                    except Exception:
                        _intended = None
                if not _intended:
                    _intended = self._position_intended_risk.get(ticket)
                _ticket_limit_usd = max(10.0, _intended * 2.0) if _intended else limit_usd
                if pnl < -_ticket_limit_usd:
                    print(
                        f"[AUTO-CLOSE] {sym} #{ticket} perdiendo ${abs(pnl):.2f}"
                        f" > limite ${_ticket_limit_usd:.0f} → cerrando",
                        flush=True,
                    )
                    await self._close_guarded(
                        loop, ticket, "LOSS-LIMIT",
                        f"<b>AUTO-CIERRE PERDIDA</b>\n{sym} #{ticket}\n"
                        f"Perdida ${abs(pnl):.2f} > limite → cerrado"
                    )
                    continue

                # ── 1b. Peak-profit retracement guard ─────────────────────
                # Closes if profit retreats 30% from peak, once peak clears
                # PEAK_MIN_USD (floor exists so market noise on a $1-2
                # winner doesn't trigger a close).
                # BUG-PEAK-GUARD-TOO-HIGH (2026-07-22): $200 meant a position
                # had to be a genuinely large winner before ANY profit
                # protection applied -- found live when a real EURUSD swing
                # peaked at $108 and gave it all back with zero protection.
                # Lowered to $50, then to $15 after a second incident
                # (GBPCAD peaked $25.34 -> -$159.76) -- both were real
                # anecdotes, but neither was validated against the full
                # 2-year backtest before deploying.
                # BUG-PEAK-GUARD-15-BACKTESTED-WORSE (2026-07-24): ran all
                # three values through scripts/backtest_multiyear.py (2 years
                # H1, real data) instead of trusting the anecdotes: $15 ->
                # P(pass Axi 5%)=28%, E[month]=$1,762, Sharpe=0.31. $200 ->
                # P(pass)=44%, E[month]=$4,139, Sharpe=0.53 (best of the
                # three; PEAK-GUARD disabled entirely scored almost
                # identically to $200). Cutting winners as early as $15
                # kills far more real winning trades across 2 years of data
                # than the 2 anecdotes it was meant to protect. Reverted to
                # $200 -- the two live incidents were real, but the fix for
                # them needs to target *those specific conditions*, not a
                # blanket floor lowered for the whole system without
                # measuring the tradeoff first.
                PEAK_MIN_USD      = 200.0
                PEAK_RETRACE_PCT  = 0.30    # close if profit drops 30% from peak
                if pnl > 0:
                    peak = self._position_peaks.get(ticket, 0.0)
                    if pnl > peak:
                        self._position_peaks[ticket] = pnl
                        peak = pnl
                    if (peak >= PEAK_MIN_USD
                            and pnl < peak * (1.0 - PEAK_RETRACE_PCT)):
                        print(
                            f"[PEAK-GUARD] {sym} #{ticket} peak=${peak:.2f} → "
                            f"actual=${pnl:.2f} (retroceso {(1-pnl/peak)*100:.0f}%) → cerrando para asegurar ganancia",
                            flush=True,
                        )
                        await self._close_guarded(
                            loop, ticket, "PEAK-GUARD",
                            f"<b>GANANCIA ASEGURADA</b>\n{sym} #{ticket}\n"
                            f"Peak: ${peak:.2f} → Retroceso 30% → cerrado en ${pnl:.2f}"
                        )
                        continue
                else:
                    self._position_peaks.pop(ticket, None)

                # ── 1c. Stagnation guard ────────────────────────────────────
                # BUG-STAGNANT-NO-GUARD (2026-07-20): a position that never
                # loses enough to trip LOSS-LIMIT and never wins enough to
                # trip PEAK-GUARD ($200) had NO guard at all -- it could sit
                # open indefinitely, drifting near breakeven for a full day
                # or more, without ever reaching its real SL or TP. That's
                # capital tied up producing neither a result nor a learning
                # signal, and it occupies one of only MAX_OPEN_POSITIONS
                # slots that a fresh, better setup could use instead. Close
                # it once it's been open a long time and never showed real
                # movement in either direction.
                # 2026-07-20 (pedido usuario): bajado de 12h a 4h -- no quiere
                # posiciones ocupando un cupo todo el dia sin moverse. Y no
                # debe forzar el cierre a la primera perdida que encuentre:
                # una vez marcada estancada, prefiere cerrar en beneficio/
                # breakeven; si esta perdiendo, da un margen corto
                # (STAGNANT_GRACE_HOURS) esperando que suba a >=0 antes de
                # forzar el cierre en la menor perdida disponible en ese
                # momento -- nunca espera indefinidamente.
                STAGNANT_HOURS       = 4.0
                STAGNANT_PEAK_MAX    = 15.0   # never even reached this much peak profit
                STAGNANT_GRACE_HOURS = 2.0    # margen extra esperando pnl >= 0 antes de forzar
                if open_time > 0:
                    import time as _stagn_time
                    _now_stagn = _stagn_time.time()
                    age_h = (_now_stagn - open_time) / 3600.0
                    # Dedicated, never-reset lifetime peak (see
                    # BUG-STAGNANT-PEAK-SHARED-STATE at __init__) -- unlike
                    # _position_peaks, this only ever grows for the life of
                    # the ticket, independent of PEAK-GUARD's own resets.
                    _lifetime_peak = max(self._position_lifetime_peak.get(ticket, 0.0), pnl)
                    self._position_lifetime_peak[ticket] = _lifetime_peak
                    peak_seen = _lifetime_peak
                    _stagn_flags = self.__dict__.setdefault("_stagnant_flagged", {})
                    if age_h >= STAGNANT_HOURS and peak_seen < STAGNANT_PEAK_MAX:
                        flagged_at = _stagn_flags.get(ticket)
                        if flagged_at is None:
                            _stagn_flags[ticket] = _now_stagn
                            flagged_at = _now_stagn
                        grace_elapsed_h = (_now_stagn - flagged_at) / 3600.0
                        should_close = pnl >= 0 or grace_elapsed_h >= STAGNANT_GRACE_HOURS
                        if should_close:
                            _motivo = "en breakeven/beneficio" if pnl >= 0 else f"forzado tras {grace_elapsed_h:.1f}h de margen"
                            print(
                                f"[STAGNANT] {sym} #{ticket} abierta {age_h:.1f}h, peak nunca superó "
                                f"${STAGNANT_PEAK_MAX:.0f} (max visto ${peak_seen:.2f}) → cerrando {_motivo}, "
                                f"actual ${pnl:+.2f} -- no llega a TP ni a SL",
                                flush=True,
                            )
                            ok = await self._close_guarded(
                                loop, ticket, "STAGNANT",
                                f"<b>CIERRE POR ESTANCAMIENTO</b>\n{sym} #{ticket}\n"
                                f"Abierta {age_h:.1f}h sin movimiento real → cerrada {_motivo} en ${pnl:+.2f}"
                            )
                            if ok:
                                _stagn_flags.pop(ticket, None)
                            else:
                                # BUG-STAGNANT-SILENT-RETRY-STALL (2026-07-22):
                                # a failed close used to leave no trace beyond
                                # this print never repeating -- found live when
                                # a real position's stagnant-close attempt
                                # failed once and then went unexplained-silent
                                # for 4+ hours. Now explicit every time it fails.
                                print(f"[STAGNANT] {sym} #{ticket} cierre fallo, reintenta el proximo ciclo", flush=True)
                        # Flagged as stagnant (closing now or still in the grace
                        # window) -- either way, skip trailing-stop logic below,
                        # it doesn't apply to a position that never developed.
                        continue
                    else:
                        _stagn_flags.pop(ticket, None)

                # ── 2-3. Trailing stop (only for winning positions) ────────
                if entry > 0 and sl_cur > 0:
                    sl_dist = abs(entry - sl_cur)  # 1R distance
                    if sl_dist < 0.0001:           # SL ya en breakeven o muy cerca → skip trailing
                        sl_dist = 0.0              # fuerza skip del bloque abajo
                    if sl_dist > 0:
                        tick = _mt5.symbol_info_tick(sym)
                        if tick is not None:
                            cur_price = tick.ask if is_buy else tick.bid
                            profit_r  = (cur_price - entry) / sl_dist if is_buy else (entry - cur_price) / sl_dist

                            new_sl = None
                            if profit_r >= 3.0:
                                # At 3R+: tight trail at 0.5R below/above price (lock in more)
                                trail_sl = (cur_price - sl_dist * 0.5) if is_buy else (cur_price + sl_dist * 0.5)
                                trail_sl = round(trail_sl, 5)
                                if (is_buy and trail_sl > sl_cur) or (not is_buy and trail_sl < sl_cur):
                                    new_sl = trail_sl
                                    print(
                                        f"[TRAIL] {sym} #{ticket} profit_R={profit_r:.1f} "
                                        f"tight trail SL {sl_cur:.5f}→{new_sl:.5f}",
                                        flush=True,
                                    )
                            elif profit_r >= 2.0:
                                # At 2R+: trail SL at 1R below/above current price
                                trail_sl = (cur_price - sl_dist) if is_buy else (cur_price + sl_dist)
                                trail_sl = round(trail_sl, 5)
                                if (is_buy and trail_sl > sl_cur) or (not is_buy and trail_sl < sl_cur):
                                    new_sl = trail_sl
                                    print(
                                        f"[TRAIL] {sym} #{ticket} profit_R={profit_r:.1f} "
                                        f"trail SL {sl_cur:.5f}→{new_sl:.5f}",
                                        flush=True,
                                    )
                            elif profit_r >= 1.5 and sl_cur != entry:
                                # At 1.5R+: move SL to breakeven (raised from 1R — more room for winners)
                                new_sl = round(entry, 5)
                                if (is_buy and new_sl > sl_cur) or (not is_buy and new_sl < sl_cur):
                                    print(
                                        f"[TRAIL] {sym} #{ticket} profit_R={profit_r:.1f} "
                                        f"→ breakeven SL {sl_cur:.5f}→{new_sl:.5f}",
                                        flush=True,
                                    )
                                else:
                                    new_sl = None

                            if new_sl is not None:
                                ok = await loop.run_in_executor(
                                    None,
                                    lambda t=ticket, s=new_sl, tp=tp_cur:
                                        self.mt5.modify_position_sl_tp(t, s, tp)
                                )
                                if ok:
                                    try:
                                        await self.telegram.send_glint_alert(
                                            f"<b>TRAILING STOP</b>\n{sym} #{ticket}\n"
                                            f"SL movido a {new_sl:.5f} | P&L: ${pnl:+.2f}"
                                        )
                                    except Exception:
                                        pass

                # ── 3b. Structure invalidation: LOSING position whose H4 bias
                # reversed against it. NOTE: H4=WAIT is deliberately NOT treated
                # as invalidation here -- the scan loop already preserves the
                # prior LONG/SHORT through WAIT reads to avoid closing on normal
                # pullback noise (see BUG #H4-WAIT-PERMANENT). Only a genuine
                # flip to the opposite direction counts -- that means the setup
                # that justified the trade is gone, not just quiet.
                # Added 2026-07-08: positions were holding to the full SL/36h
                # timer even after their own structural justification reversed,
                # with nothing acting on it until a human checked manually.
                if pnl < 0:
                    import time as _stime
                    _pos_dir = "LONG" if is_buy else "SHORT"
                    _opposite_dir = "SHORT" if is_buy else "LONG"
                    _cur_h4_dir = self._mt5_h4_direction.get(sym)
                    if _cur_h4_dir == _opposite_dir:
                        _last_try = self._close_attempted.get(ticket, 0.0)
                        if _stime.time() - _last_try >= 300:
                            self._close_attempted[ticket] = _stime.time()
                            print(
                                f"[STRUCTURE-INVALID] {sym} #{ticket} era {_pos_dir}, "
                                f"H4 ahora {_cur_h4_dir} -- perdiendo ${pnl:.2f} → cerrando",
                                flush=True,
                            )
                            ok = await self._close_guarded(
                                loop, ticket, "STRUCT-INVALID",
                                f"<b>CIERRE POR REVERSION DE ESTRUCTURA</b>\n{sym} #{ticket}\n"
                                f"Era {_pos_dir}, H4 ahora {_cur_h4_dir} → cerrada en ${pnl:+.2f}"
                            )
                            if ok:
                                self._close_attempted.pop(ticket, None)
                            continue

                # ── 4. Hard close LOSING position stuck > MAX_HOLD_HOURS ────
                # Winners are handled by trail SL / breakeven — don't kill them early.
                if pnl <= 0 and open_time > 0:
                    import time as _time
                    age_h = (_time.time() - open_time) / 3600
                    if age_h >= MAX_HOLD_HOURS:
                        # Cooldown: don't spam close attempts — retry at most every 5 min
                        _last_try = self._close_attempted.get(ticket, 0.0)
                        if _time.time() - _last_try < 300:
                            continue  # skip until cooldown expires
                        self._close_attempted[ticket] = _time.time()
                        print(
                            f"[TIME-CLOSE] {sym} #{ticket} abierta {age_h:.1f}h "
                            f"perdiendo ${pnl:.2f} → cerrando (limite {MAX_HOLD_HOURS}h)",
                            flush=True,
                        )
                        ok = await self._close_guarded(
                            loop, ticket, "TIME-CLOSE-36H",
                            f"<b>CIERRE POR TIEMPO</b>\n{sym} #{ticket}\n"
                            f"Abierta {age_h:.1f}h perdiendo → cerrada en ${pnl:+.2f}"
                        )
                        if ok:
                            self._close_attempted.pop(ticket, None)
                        else:
                            print(f"[TIME-CLOSE] {sym} #{ticket} close FALLO — reintento en 5min", flush=True)

        except Exception as _me:
            import traceback
            print(f"[AUTO-CLOSE] error monitor: {_me}\n{traceback.format_exc()[:300]}", flush=True)
