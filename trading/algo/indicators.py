"""Technical indicators. All functions are causal (no lookahead) unless noted."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def prior_sma(s: pd.Series, n: int) -> pd.Series:
    """SMA of the n bars *before* the current bar (excludes the current bar)."""
    return sma(s, n).shift(1)


def pivots(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractal pivots: bar j is a pivot high if high[j] is the max of high[j-n : j+n+1].

    A pivot at j is only *known* at bar j+n — callers must respect that delay.
    """
    size = len(high)
    ph = np.zeros(size, dtype=bool)
    pl = np.zeros(size, dtype=bool)
    if size < 2 * n + 1:
        return ph, pl
    hs = pd.Series(high)
    ls = pd.Series(low)
    win = 2 * n + 1
    hmax = hs.rolling(win, center=True).max().to_numpy()
    lmin = ls.rolling(win, center=True).min().to_numpy()
    ph = (high == hmax) & ~np.isnan(hmax)
    pl = (low == lmin) & ~np.isnan(lmin)
    return ph, pl
