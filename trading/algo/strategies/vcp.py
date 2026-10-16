"""MODULE 1 — Govardhan Gajjala VCP + Momentum (stocks, 5/15-min)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..indicators import ema, pivots, prior_sma
from .base import ExitRules, Strategy


@dataclass
class Pivot:
    kind: str  # "H" | "L"
    idx: int
    price: float


@dataclass
class VCP:
    pivots: list[Pivot]
    depths_pct: list[float]
    volumes: list[float]
    last_low: float
    tl_now: float
    tl_prev: float


def _push_pivot(seq: list[Pivot], p: Pivot) -> None:
    """Keep an alternating H/L sequence; consecutive same-type pivots keep the extreme."""
    if seq and seq[-1].kind == p.kind:
        if (p.kind == "H" and p.price >= seq[-1].price) or (p.kind == "L" and p.price <= seq[-1].price):
            seq[-1] = p
        return
    seq.append(p)


def match_vcp(seq: list[Pivot], t: int, high: np.ndarray, low: np.ndarray, close: np.ndarray,
              vol: np.ndarray, p: dict[str, Any]) -> VCP | None:
    """Check whether the confirmed pivots ending at bar t form a VCP that breaks out at t."""
    if len(seq) < 2 or seq[-1].kind != "L" or t < 1:
        return None
    rmin, rmax = p["ratio_min"], p["ratio_max"]
    for k in range(p["max_contractions"], p["min_contractions"] - 1, -1):
        pts = seq[-2 * k:]
        if len(pts) < 2 * k or pts[0].kind != "H":
            continue
        if t - pts[0].idx > p["max_base_bars"]:
            continue
        highs = pts[0::2]
        lows = pts[1::2]
        depths = [(hh.price - ll.price) / hh.price * 100.0 for hh, ll in zip(highs, lows)]
        if depths[0] <= 0:
            continue
        if p["max_first_depth_pct"] is not None and depths[0] > p["max_first_depth_pct"]:
            continue
        if not all(depths[i - 1] > 0 and rmin <= depths[i] / depths[i - 1] <= rmax for i in range(1, k)):
            continue
        vols = [float(vol[hh.idx: ll.idx + 1].mean()) for hh, ll in zip(highs, lows)]
        if not all(vols[i] < vols[i - 1] for i in range(1, k)):
            continue
        h1, hk = highs[0], highs[-1]
        if hk.price >= h1.price or hk.idx == h1.idx:
            continue  # trendline must descend
        slope = (hk.price - h1.price) / (hk.idx - h1.idx)
        tl = lambda x: h1.price + slope * (x - h1.idx)  # noqa: E731
        tol = p["trendline_tolerance_pct"] / 100.0
        if any(hh.price > tl(hh.idx) * (1 + tol) for hh in highs[1:-1]):
            continue
        last_low = lows[-1]
        if low[last_low.idx + 1: t + 1].min(initial=np.inf) < last_low.price:
            continue  # last swing low undercut — base broken
        tl_now = tl(t)
        if not close[t] > tl_now:
            continue
        # first close above the trendline since the last contraction high must be recent
        xs = np.arange(hk.idx + 1, t + 1)
        first = xs[np.argmax(close[xs] > h1.price + slope * (xs - h1.idx))]
        if t - first >= p["breakout_window_bars"]:
            continue
        return VCP(pts, depths, vols, last_low.price, tl_now, tl(t - 1))
    return None


class VCPStrategy(Strategy):
    name = "vcp"

    def __init__(self, cfg: dict[str, Any], market):
        super().__init__(cfg, market)
        self.exit_rules = ExitRules(
            break_even_r=cfg.get("break_even_r", 1.0),
            eod_flatten=cfg.get("eod_flatten", True),
            use_exit_signal=cfg.get("exit_on_close_below_ema", True),
        )
        self.params = {
            "pivot_bars": int(cfg.get("pivot_bars", 3)),
            "min_contractions": int(cfg.get("min_contractions", 3)),
            "max_contractions": int(cfg.get("max_contractions", 4)),
            "ratio_min": float(cfg.get("contraction_ratio_min", 0.50)),
            "ratio_max": float(cfg.get("contraction_ratio_max", 0.70)),
            "max_base_bars": int(cfg.get("max_base_bars", 200)),
            "max_first_depth_pct": cfg.get("max_first_depth_pct"),
            "trendline_tolerance_pct": float(cfg.get("trendline_tolerance_pct", 0.5)),
            "breakout_window_bars": int(cfg.get("breakout_window_bars", 3)),
        }

    # ---- daily universe + momentum filter -----------------------------
    def daily_filter(self, daily: pd.DataFrame | None, meta: dict[str, Any]) -> pd.Series:
        c = self.cfg
        if daily is None or daily.empty:
            return pd.Series(dtype=bool)
        ok = pd.Series(True, index=daily.index)
        if not c.get("use_universe_filter", True) and not c.get("use_momentum_filter", True):
            return ok
        mc = meta.get("market_cap_cr")
        lo, hi = c.get("market_cap_min_cr", 1000.0), c.get("market_cap_max_cr", 5000.0)
        if c.get("use_universe_filter", True):
            if mc is None:
                if c.get("require_market_cap", True):
                    ok &= False
            elif not (lo <= mc <= hi):
                ok &= False
            avg_days = int(c.get("avg_volume_days", 20))
            avg_vol = daily["volume"].rolling(avg_days, min_periods=avg_days).mean().shift(1)
            ok &= avg_vol > float(c.get("min_avg_volume", 500_000))
        if not c.get("use_momentum_filter", True):
            return ok.fillna(False)
        lb0, lb1 = int(c.get("momentum_lookback_min_days", 20)), int(c.get("momentum_lookback_max_days", 25))
        mmin, mmax = c.get("momentum_min_pct", 40.0), c.get("momentum_max_pct", 50.0)
        prev_close = daily["close"].shift(1)  # use info as of the prior session close only
        mom = pd.Series(False, index=daily.index)
        for n in range(lb0, lb1 + 1):
            ret = (prev_close / daily["close"].shift(1 + n) - 1.0) * 100.0
            cond = ret >= mmin
            if mmax is not None:
                cond &= ret <= mmax
            mom |= cond.fillna(False)
        return (ok & mom).fillna(False)

    def generate(self, symbol, base, daily, meta):
        out = self.empty_signals(base)
        sig, last_pos = self.to_signal_tf(base)
        if len(sig) < 30:
            return out
        o, h, l, c, v = (sig[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
        e9 = ema(sig["close"], int(self.cfg.get("ema_fast", 9))).to_numpy()
        e21 = ema(sig["close"], int(self.cfg.get("ema_slow", 21))).to_numpy()
        vs = prior_sma(sig["volume"], int(self.cfg.get("volume_sma_bars", 20))).to_numpy()
        vmult = float(self.cfg.get("breakout_volume_mult", 1.5))

        dfilt = self.daily_filter(daily, meta)
        dates = pd.DatetimeIndex(pd.to_datetime(sig["session_date"]))
        filt_ok = dfilt.reindex(dates).fillna(False).to_numpy(dtype=bool) if not dfilt.empty else np.zeros(len(sig), bool)

        n = self.params["pivot_bars"]
        ph, pl = pivots(h, l, n)
        tp_pct = float(self.cfg.get("take_profit_pct", 10.0))
        exit_sig = c < e9
        for k in range(len(sig)):
            if exit_sig[k]:
                self.place(out, last_pos, k, exit_signal=True)

        seq: list[Pivot] = []
        used: set[tuple[int, int]] = set()
        for t in range(len(sig)):
            j = t - n
            if j >= 0:
                if ph[j] and pl[j]:
                    first, second = (("L", l[j]), ("H", h[j])) if c[j] >= o[j] else (("H", h[j]), ("L", l[j]))
                    _push_pivot(seq, Pivot(first[0], j, first[1]))
                    _push_pivot(seq, Pivot(second[0], j, second[1]))
                elif ph[j]:
                    _push_pivot(seq, Pivot("H", j, h[j]))
                elif pl[j]:
                    _push_pivot(seq, Pivot("L", j, l[j]))
            if not filt_ok[t]:
                continue
            if not (c[t] > e9[t] > e21[t]):
                continue
            if not (np.isfinite(vs[t]) and v[t] > vmult * vs[t]):
                continue
            m = match_vcp(seq, t, h, l, c, v, self.params)
            if m is None:
                continue
            key = (m.pivots[0].idx, m.pivots[-1].idx)
            if key in used:
                continue  # one entry per VCP base
            used.add(key)
            stop = m.last_low - self.stop_buffer(m.last_low)
            if stop >= c[t]:
                continue
            self.place(out, last_pos, t, entry=1, stop=stop, target=c[t] * (1 + tp_pct / 100.0))
        return out
