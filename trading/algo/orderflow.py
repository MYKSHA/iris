"""Order-flow helpers: delta (real or OHLCV-estimated), tick aggregation, volume profile / POC."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_delta(df: pd.DataFrame) -> pd.Series:
    """Estimated delta = ((close - open) / range) * volume. Zero-range bars get 0."""
    rng = (df["high"] - df["low"]).to_numpy(dtype=float)
    body = (df["close"] - df["open"]).to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(rng > 0, body / rng * vol, 0.0)
    return pd.Series(d, index=df.index, name="delta")


def bar_delta(df: pd.DataFrame) -> pd.Series:
    """Real delta if the data carries it (from tick/footprint aggregation), else the estimate."""
    if "delta" in df.columns and df["delta"].notna().all():
        return df["delta"].astype(float)
    est = estimate_delta(df)
    if "delta" in df.columns:
        return df["delta"].astype(float).fillna(est)
    return est


def ticks_to_bars(ticks: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate time & sales into OHLCV bars with real delta.

    ticks: DatetimeIndex, columns `price`, `size`, optional `side` (aggressor:
    'B'/'S', 'buy'/'sell', +1/-1). Without `side`, the tick rule is used
    (uptick = buy, downtick = sell, unchanged = previous side).
    """
    t = ticks.sort_index()
    price = t["price"].astype(float)
    size = t["size"].astype(float)
    if "side" in t.columns:
        raw = t["side"]
        if not pd.api.types.is_numeric_dtype(raw):
            side = raw.astype(str).str.upper().str[0].map({"B": 1.0, "S": -1.0}).fillna(0.0)
        else:
            side = np.sign(raw.astype(float))
    else:
        side = np.sign(price.diff()).replace(0.0, np.nan).ffill().fillna(0.0)
    signed = size * side
    g = pd.DataFrame({"price": price, "size": size, "signed": signed}).resample(
        interval, label="left", closed="left"
    )
    bars = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["size"].sum(),
            "delta": g["signed"].sum(),
        }
    ).dropna(subset=["open"])
    return bars


def volume_profile_from_bars(df: pd.DataFrame, bin_size: float) -> pd.Series:
    """Spread each bar's volume uniformly over the price bins its high-low range covers."""
    acc: dict[int, float] = {}
    for lo, hi, v in zip(df["low"].to_numpy(), df["high"].to_numpy(), df["volume"].to_numpy()):
        b0 = int(np.floor(lo / bin_size + 1e-9))
        b1 = int(np.floor(hi / bin_size + 1e-9))
        n = b1 - b0 + 1
        share = float(v) / n
        for b in range(b0, b1 + 1):
            acc[b] = acc.get(b, 0.0) + share
    if not acc:
        return pd.Series(dtype=float)
    keys = sorted(acc)
    return pd.Series([acc[k] for k in keys], index=[k * bin_size for k in keys], name="volume")


def volume_profile_from_ticks(ticks: pd.DataFrame, bin_size: float) -> pd.Series:
    bins = np.floor(ticks["price"].astype(float) / bin_size + 1e-9) * bin_size
    return ticks["size"].astype(float).groupby(bins).sum().sort_index().rename("volume")


def poc(profile: pd.Series, bin_size: float) -> float:
    """Point of control = centre of the highest-volume bin; ties go to the bin nearest the profile midpoint."""
    if profile.empty:
        return float("nan")
    mx = profile.max()
    cands = profile.index[profile.to_numpy() >= mx - 1e-12].to_numpy(dtype=float)
    mid = (profile.index.min() + profile.index.max()) / 2.0
    best = cands[np.argmin(np.abs(cands - mid))]
    return float(best + bin_size / 2.0)
