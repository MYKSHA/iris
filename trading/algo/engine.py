"""Event-driven backtest engine.

One global timeline across every feed (symbol x market). At most ONE open
position across all modules at any time. Entries fill at the signal bar's
close (plus slippage); exits are evaluated on each later base bar in this
order: gap through stop/target at the open, stop, target (when one bar
touches both, engine.intrabar decides: stop_first = conservative default,
open_proximity = whichever level is nearer the open), then break-even / time stop / exit signal / end of day
at the close.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Market
from .strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class Feed:
    symbol: str
    market: Market
    base: pd.DataFrame
    signals: dict[str, pd.DataFrame] = field(default_factory=dict)  # strategy name -> signals

    def arrays(self):
        b = self.base
        self.o = b["open"].to_numpy(float)
        self.h = b["high"].to_numpy(float)
        self.l = b["low"].to_numpy(float)
        self.c = b["close"].to_numpy(float)
        self.ts = b.index.tz_convert("UTC")
        self.bar_end = pd.DatetimeIndex(b["bar_end"]).tz_convert("UTC")
        self.is_last = b["is_last_bar"].to_numpy(bool)
        self.date = b["session_date"].to_numpy()
        self.sig_np = {
            k: (s["entry"].to_numpy(int), s["stop"].to_numpy(float), s["target"].to_numpy(float),
                s["exit_signal"].to_numpy(bool))
            for k, s in self.signals.items()
        }


@dataclass
class Position:
    strategy: Strategy
    feed_id: int
    side: int
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_pos: int
    initial_stop: float
    stop: float
    target: float
    risk_amount: float
    entry_commission: float
    equity_before: float
    be_done: bool = False
    time_deadline: pd.Timestamp | None = None
    mfe: float = 0.0
    mae: float = 0.0


class Backtester:
    def __init__(self, strategies: list[Strategy], feeds: list[Feed], initial_equity: float,
                 one_position: bool = True, intrabar: str = "stop_first"):
        self.strategies = strategies
        self.feeds = feeds
        self.initial_equity = float(initial_equity)
        self.one_position = one_position
        if intrabar not in ("stop_first", "open_proximity"):
            raise ValueError("engine.intrabar must be 'stop_first' or 'open_proximity'")
        self.intrabar = intrabar  # which fills first when one bar touches both stop and target
        if not one_position:
            log.warning("engine.one_position=false is not supported; enforcing a single position")

    # ---- costs / sizing ----------------------------------------------
    @staticmethod
    def _commission(m: Market, qty: float, price: float) -> float:
        return qty * m.commission_per_unit * m.fx_to_base + abs(qty * price * m.point_value * m.fx_to_base) * m.commission_bps / 1e4

    @staticmethod
    def size(strat: Strategy, m: Market, equity: float, entry: float, stop: float) -> float:
        per_unit = abs(entry - stop) * m.point_value * m.fx_to_base
        if per_unit <= 0 or equity <= 0:
            return 0.0
        sz = strat.sizing
        qty = math.floor(equity * sz.risk_pct / 100.0 / per_unit)
        if m.asset_class == "future":
            if sz.max_contracts:
                qty = min(qty, int(sz.max_contracts))
            return float(max(qty, 0))
        lot = max(m.lot_size, 1)
        qty = (qty // lot) * lot
        unit_notional = entry * m.point_value * m.fx_to_base
        if sz.max_position_pct:
            cap = math.floor(equity * sz.max_position_pct / 100.0 / unit_notional)
            qty = min(qty, (cap // lot) * lot)
        if sz.min_position_pct and qty * unit_notional < equity * sz.min_position_pct / 100.0:
            return 0.0
        return float(max(qty, 0))

    # ---- main loop ----------------------------------------------------
    def run(self) -> dict[str, Any]:
        for f in self.feeds:
            f.arrays()
        pairs = [(s, fi) for s in self.strategies for fi, f in enumerate(self.feeds) if s.name in f.signals]

        ts_parts, feed_parts, pos_parts = [], [], []
        for fi, f in enumerate(self.feeds):
            n = len(f.base)
            ts_parts.append(f.ts.as_unit("ns").asi8)
            feed_parts.append(np.full(n, fi))
            pos_parts.append(np.arange(n))
        if not ts_parts:
            return self._result([], pd.Series(dtype=float))
        all_ts = np.concatenate(ts_parts)
        all_feed = np.concatenate(feed_parts)
        all_pos = np.concatenate(pos_parts)
        order = np.lexsort((all_feed, all_ts))
        all_ts, all_feed, all_pos = all_ts[order], all_feed[order], all_pos[order]
        bounds = np.flatnonzero(np.diff(all_ts)) + 1
        starts = np.r_[0, bounds]
        ends = np.r_[bounds, len(all_ts)]

        cash = self.initial_equity
        peak = cash
        pos: Position | None = None
        last_px: float | None = None
        trades: list[dict[str, Any]] = []
        eq_ts = np.empty(len(starts), dtype=np.int64)
        eq_val = np.empty(len(starts))
        day_count: dict[tuple[str, Any], int] = {}

        for gi, (a, b) in enumerate(zip(starts, ends)):
            group = dict(zip(all_feed[a:b].tolist(), all_pos[a:b].tolist()))
            exited = False
            if pos is not None and pos.feed_id in group:
                i = group[pos.feed_id]
                f = self.feeds[pos.feed_id]
                last_px = f.c[i]
                res = self._manage(pos, f, i)
                if res is not None:
                    px, reason = res
                    tr, cash = self._close(pos, f, i, px, reason, cash, peak)
                    trades.append(tr)
                    pos, last_px, exited = None, None, True
            if pos is None and not exited:
                for strat, fi in pairs:
                    i = group.get(fi)
                    if i is None:
                        continue
                    f = self.feeds[fi]
                    ent, stp, tgt, _ = f.sig_np[strat.name]
                    side = int(ent[i])
                    if side == 0:
                        continue
                    key = (strat.name, f.date[i])
                    if strat.max_trades_per_day and day_count.get(key, 0) >= strat.max_trades_per_day:
                        continue
                    opened = self._open(strat, fi, f, i, side, stp[i], tgt[i], cash)
                    if opened is None:
                        continue
                    pos, cash = opened
                    last_px = f.c[i]
                    day_count[key] = day_count.get(key, 0) + 1
                    break
            mtm = cash
            if pos is not None and last_px is not None:
                m = self.feeds[pos.feed_id].market
                mtm += pos.side * (last_px - pos.entry_price) * pos.qty * m.point_value * m.fx_to_base
            peak = max(peak, mtm)
            eq_ts[gi] = all_ts[a]
            eq_val[gi] = mtm

        if pos is not None:  # close anything still open at the final bar
            f = self.feeds[pos.feed_id]
            i = len(f.base) - 1
            tr, cash = self._close(pos, f, i, f.c[i] - pos.side * f.market.slippage_ticks * f.market.tick_size,
                                   "end_of_data", cash, peak)
            trades.append(tr)
            eq_val[-1] = cash
        equity = pd.Series(eq_val, index=pd.DatetimeIndex(eq_ts, tz="UTC"), name="equity")
        return self._result(trades, equity)

    def _result(self, trades, equity):
        tdf = pd.DataFrame(trades)
        return {"trades": tdf, "equity": equity, "initial_equity": self.initial_equity}

    def _open(self, strat: Strategy, fi: int, f: Feed, i: int, side: int, stop: float, target: float, cash: float):
        m = f.market
        slip = m.slippage_ticks * m.tick_size
        fill = f.c[i] + side * slip
        if not np.isfinite(stop) or side * (fill - stop) <= 0:
            return None
        qty = self.size(strat, m, cash, fill, stop)
        if qty <= 0:
            log.debug("%s %s %s: size 0, skipped", strat.name, f.symbol, f.ts[i])
            return None
        comm = self._commission(m, qty, fill)
        risk_amount = qty * abs(fill - stop) * m.point_value * m.fx_to_base
        entry_time = f.bar_end[i]
        er = strat.exit_rules
        deadline = entry_time + pd.Timedelta(minutes=er.time_stop_minutes) if er.time_stop_minutes else None
        pos = Position(
            strategy=strat, feed_id=fi, side=side, qty=qty, entry_price=fill, entry_time=entry_time,
            entry_pos=i, initial_stop=float(stop), stop=float(stop),
            target=float(target) if np.isfinite(target) else float("nan"),
            risk_amount=risk_amount, entry_commission=comm, equity_before=cash, time_deadline=deadline,
        )
        log.info("ENTRY %-3s %-12s %s %s qty=%g @ %.4f stop=%.4f target=%.4f", strat.name, f.symbol,
                 entry_time.tz_convert(m.timezone), "LONG" if side > 0 else "SHORT", qty, fill, stop, pos.target)
        return pos, cash - comm

    def _manage(self, p: Position, f: Feed, i: int):
        """Return (exit_price, reason) or None."""
        m = f.market
        s = p.side
        slip = m.slippage_ticks * m.tick_size
        o, h, l, c = f.o[i], f.h[i], f.l[i], f.c[i]
        adverse, favor = (l, h) if s > 0 else (h, l)
        has_tgt = np.isfinite(p.target)
        stop_reason = "breakeven" if p.be_done else "stop"
        p.mae = max(p.mae, s * (p.entry_price - adverse))
        if s * (o - p.stop) <= 0:
            return o - s * slip, stop_reason
        if has_tgt and s * (o - p.target) >= 0:
            return o, "target"
        hit_stop = s * (adverse - p.stop) <= 0
        hit_tgt = has_tgt and s * (favor - p.target) >= 0
        if hit_stop and hit_tgt and self.intrabar == "open_proximity" and abs(o - p.target) < abs(o - p.stop):
            return p.target, "target"
        if hit_stop:
            return p.stop - s * slip, stop_reason
        if hit_tgt:
            return p.target, "target"
        p.mfe = max(p.mfe, s * (favor - p.entry_price))
        er = p.strategy.exit_rules
        r_unit = abs(p.entry_price - p.initial_stop)
        if er.break_even_r is not None and not p.be_done and p.mfe >= er.break_even_r * r_unit:
            p.stop = max(p.stop, p.entry_price) if s > 0 else min(p.stop, p.entry_price)
            p.be_done = True
        if p.time_deadline is not None:
            if p.mfe >= er.time_stop_min_ticks * m.tick_size:
                p.time_deadline = None
            elif f.bar_end[i] >= p.time_deadline:
                return c - s * slip, "time_stop"
        if er.use_exit_signal and f.sig_np[p.strategy.name][3][i]:
            return c - s * slip, "exit_signal"
        if er.eod_flatten and f.is_last[i]:
            return c - s * slip, "eod"
        return None

    def _close(self, p: Position, f: Feed, i: int, px: float, reason: str, cash: float, peak: float):
        m = f.market
        gross = p.side * (px - p.entry_price) * p.qty * m.point_value * m.fx_to_base
        comm = self._commission(m, p.qty, px)
        pnl = gross - p.entry_commission - comm
        cash = cash + gross - comm
        peak = max(peak, cash)
        r_unit = abs(p.entry_price - p.initial_stop)
        tr = {
            "strategy": p.strategy.name,
            "symbol": f.symbol,
            "side": "LONG" if p.side > 0 else "SHORT",
            "entry_time": p.entry_time.tz_convert(m.timezone),
            "exit_time": f.bar_end[i].tz_convert(m.timezone),
            "entry_price": p.entry_price,
            "exit_price": px,
            "qty": p.qty,
            "initial_stop": p.initial_stop,
            "final_stop": p.stop,
            "target": p.target,
            "exit_reason": reason,
            "pnl": pnl,
            "r_multiple": pnl / p.risk_amount if p.risk_amount > 0 else 0.0,
            "return_pct": pnl / p.equity_before * 100.0,
            "result": "WIN" if pnl > 0 else "LOSS",
            "mfe_r": p.mfe / r_unit if r_unit else 0.0,
            "mae_r": p.mae / r_unit if r_unit else 0.0,
            "bars_held": i - p.entry_pos,
            "equity_before": p.equity_before,
            "equity_after": cash,
            "drawdown_pct": (peak - cash) / peak * 100.0 if peak > 0 else 0.0,
        }
        log.info("EXIT  %-3s %-12s %s @ %.4f reason=%s pnl=%.2f R=%.2f dd=%.2f%%", tr["strategy"], f.symbol,
                 tr["exit_time"], px, reason, pnl, tr["r_multiple"], tr["drawdown_pct"])
        return tr, cash
