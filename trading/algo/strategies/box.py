"""MODULE 3 — Gajjala Horizontal Fade / Box Breakout (stocks, 5/15-min)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..indicators import ema, prior_sma
from .base import ExitRules, Strategy


def find_boxes(high: pd.Series, low: pd.Series, volume: pd.Series, min_bars: int, max_bars: int,
               min_range_pct: float, max_range_pct: float, vol_sma_bars: int) -> pd.DataFrame:
    """For each bar t, the longest box of N in [min_bars, max_bars] bars ending at t-1.

    Box qualifies when (box_high - box_low) / box_low is within [min_range_pct, max_range_pct]
    and mean box volume < the 20-bar SMA volume measured just before the box started.
    Returns columns box_high, box_low, box_bars (NaN where no box).
    """
    size = len(high)
    bh = np.full(size, np.nan)
    bl = np.full(size, np.nan)
    bn = np.full(size, np.nan)
    vol_sma = volume.rolling(vol_sma_bars, min_periods=vol_sma_bars).mean()
    for n in range(max_bars, min_bars - 1, -1):
        hh = high.rolling(n).max().shift(1).to_numpy()
        ll = low.rolling(n).min().shift(1).to_numpy()
        vm = volume.rolling(n).mean().shift(1).to_numpy()
        ref = vol_sma.shift(n + 1).to_numpy()  # SMA of the bars preceding the box
        with np.errstate(divide="ignore", invalid="ignore"):
            rng = (hh - ll) / ll * 100.0
        ok = (rng >= min_range_pct) & (rng <= max_range_pct) & (vm < ref) & np.isnan(bn)
        bh[ok], bl[ok], bn[ok] = hh[ok], ll[ok], n
    return pd.DataFrame({"box_high": bh, "box_low": bl, "box_bars": bn}, index=high.index)


class BoxStrategy(Strategy):
    name = "box"

    def __init__(self, cfg: dict[str, Any], market):
        super().__init__(cfg, market)
        self.exit_rules = ExitRules(
            break_even_r=cfg.get("break_even_r"),
            eod_flatten=cfg.get("eod_flatten", True),
            use_exit_signal=cfg.get("exit_on_close_below_ema", False),
        )

    def generate(self, symbol, base, daily, meta):
        out = self.empty_signals(base)
        sig, last_pos = self.to_signal_tf(base)
        c = self.cfg
        vsb = int(c.get("volume_sma_bars", 20))
        if len(sig) < int(c.get("max_bars", 30)) + vsb + 2:
            return out
        boxes = find_boxes(
            sig["high"], sig["low"], sig["volume"],
            int(c.get("min_bars", 10)), int(c.get("max_bars", 30)),
            float(c.get("min_range_pct", 1.0)), float(c.get("max_range_pct", 3.0)), vsb,
        )
        close = sig["close"].to_numpy(dtype=float)
        vol = sig["volume"].to_numpy(dtype=float)
        e21 = ema(sig["close"], int(c.get("ema_slow", 21))).to_numpy()
        e9 = ema(sig["close"], int(c.get("ema_fast", 9))).to_numpy()
        vs = prior_sma(sig["volume"], vsb).to_numpy()
        vmult = float(c.get("breakout_volume_mult", 1.5))
        mode = c.get("stop_mode", "box_low")
        tp_r = c.get("take_profit_r", 2.0)
        require_trend = bool(c.get("require_trend", False))
        bh, bl = boxes["box_high"].to_numpy(), boxes["box_low"].to_numpy()

        for k in range(len(sig)):
            if close[k] < e9[k]:
                self.place(out, last_pos, k, exit_signal=True)
            if np.isnan(bh[k]) or not np.isfinite(vs[k]):
                continue
            if not (close[k] > bh[k] and vol[k] > vmult * vs[k]):
                continue
            if require_trend and not (close[k] > e9[k] > e21[k]):
                continue
            box_stop = bl[k] - self.stop_buffer(bl[k])
            ema_stop = e21[k] - self.stop_buffer(e21[k])
            cands = {"box_low": box_stop, "ema21": ema_stop}
            below = [s for s in (box_stop, ema_stop) if s < close[k]]
            if mode in cands:
                stop = cands[mode]
            elif not below:
                continue
            else:
                stop = max(below) if mode == "tighter" else min(below)
            if not stop < close[k]:
                continue
            target = close[k] + tp_r * (close[k] - stop) if tp_r else np.nan
            self.place(out, last_pos, k, entry=1, stop=stop, target=target)
        return out
