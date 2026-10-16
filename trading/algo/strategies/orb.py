"""MODULE 2 — Fabio Valentini ORB + Order Flow (Nasdaq futures, 5-min)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import _parse_time
from ..orderflow import bar_delta, poc, volume_profile_from_bars, volume_profile_from_ticks
from .base import ExitRules, Strategy


class ORBStrategy(Strategy):
    name = "orb"

    def __init__(self, cfg: dict[str, Any], market):
        super().__init__(cfg, market)
        self.exit_rules = ExitRules(
            time_stop_minutes=cfg.get("time_stop_minutes", 3.0),
            time_stop_min_ticks=float(cfg.get("time_stop_min_ticks", 2.0)),
            eod_flatten=cfg.get("eod_flatten", True),
            use_exit_signal=False,
        )
        self.or_start = _parse_time(cfg.get("or_start", "09:30"))
        self.or_end = _parse_time(cfg.get("or_end", "10:00"))
        self.entry_cutoff = _parse_time(cfg.get("entry_cutoff", market.session_end.strftime("%H:%M")))

    def opening_range(self, base: pd.DataFrame, ticks: pd.DataFrame | None) -> pd.DataFrame:
        """Per session date: OR high/low and the opening-range volume-profile POC."""
        tick = self.market.tick_size
        bin_size = float(self.cfg.get("profile_bin_ticks", 1)) * tick
        t = base.index.time
        or_bars = base[(t >= self.or_start) & (t < self.or_end)]
        rows = {}
        for d, g in or_bars.groupby("session_date"):
            if ticks is not None:
                tt = ticks[(ticks.index >= g.index[0]) & (ticks.index < g["bar_end"].iloc[-1])]
                prof = volume_profile_from_ticks(tt, bin_size) if len(tt) else volume_profile_from_bars(g, bin_size)
            else:
                prof = volume_profile_from_bars(g, bin_size)
            rows[d] = (g["high"].max(), g["low"].min(), poc(prof, bin_size), g["bar_end"].iloc[-1])
        return pd.DataFrame.from_dict(rows, orient="index", columns=["or_high", "or_low", "poc", "or_complete"])

    def generate(self, symbol, base, daily, meta):
        out = self.empty_signals(base)
        sig, last_pos = self.to_signal_tf(base)
        if sig.empty:
            return out
        tick = self.market.tick_size
        orr = self.opening_range(base, meta.get("ticks"))
        if orr.empty:
            return out
        sig = sig.copy()
        sig["delta"] = bar_delta(sig).to_numpy()
        sig["cum_delta"] = sig.groupby("session_date")["delta"].cumsum()
        sig_end = base["bar_end"].to_numpy()[last_pos]
        tz = self.market.timezone

        stop_ticks = float(self.cfg.get("stop_ticks", 6))
        target_ticks = float(self.cfg.get("target_ticks", 12))
        allow_short = bool(self.cfg.get("allow_short", True))
        delay = pd.Timedelta(seconds=self.market.entry_delay_seconds)

        close = sig["close"].to_numpy(dtype=float)
        cd = sig["cum_delta"].to_numpy(dtype=float)
        dates = sig["session_date"].to_numpy()
        prev_close = np.r_[np.nan, close[:-1]]
        prev_date = np.r_[[None], dates[:-1]]

        for k in range(len(sig)):
            d = dates[k]
            if d not in orr.index:
                continue
            orh, orl, p, or_done = orr.loc[d]
            end_k = sig_end[k]
            if end_k <= or_done:
                continue  # OR not complete yet
            sess_open = pd.Timestamp(datetime.combine(d, self.market.session_start), tz=tz)
            if end_k < sess_open + delay:
                continue
            cutoff = pd.Timestamp(datetime.combine(d, self.entry_cutoff), tz=tz)
            if end_k >= cutoff:
                continue
            pc = prev_close[k] if prev_date[k] == d else np.nan
            if close[k] > orh and (np.isnan(pc) or pc <= orh) and cd[k] > 0 and p < close[k]:
                self.place(out, last_pos, k, entry=1, stop=orh - stop_ticks * tick,
                           target=close[k] + target_ticks * tick)
            elif allow_short and close[k] < orl and (np.isnan(pc) or pc >= orl) and cd[k] < 0 and p > close[k]:
                self.place(out, last_pos, k, entry=-1, stop=orl + stop_ticks * tick,
                           target=close[k] - target_ticks * tick)
        return out
