"""Data loading: CSV (TradingView export / generic), Yahoo Finance, tick CSV, synthetic.

Every loader returns a DataFrame indexed by tz-aware *bar-start* timestamps in the
market's timezone with lowercase columns open/high/low/close/volume (+ optional delta).
`prepare_session` then trims to the trading session and adds session bookkeeping.
"""

from __future__ import annotations

import logging
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Market
from .orderflow import ticks_to_bars

log = logging.getLogger(__name__)

_TIME_COLS = ("time", "datetime", "date", "timestamp", "ts")


def _parse_times(col: pd.Series) -> pd.DatetimeIndex:
    """Unix seconds (TradingView export), offset-aware ISO strings, or naive local strings."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.DatetimeIndex(pd.to_datetime(col, unit="s", utc=True))
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.DatetimeIndex(col)
    has_offset = col.astype(str).str.contains(r"(?:[+-]\d\d:?\d\d|Z)$", regex=True).any()
    return pd.DatetimeIndex(pd.to_datetime(col, utc=bool(has_offset)))


def _normalise(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in _TIME_COLS if c in df.columns), None)
    if tcol is not None:
        idx = _parse_times(df[tcol])
        df = df.drop(columns=[tcol])
        df.index = pd.DatetimeIndex(idx)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("data has no datetime column/index")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    df = df.rename(columns={"vol": "volume"})
    keep = [c for c in ("open", "high", "low", "close", "volume", "delta") if c in df.columns]
    missing = {"open", "high", "low", "close", "volume"} - set(keep)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    df = df[keep].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def load_csv(path: str | Path, tz: str) -> pd.DataFrame:
    return _normalise(pd.read_csv(path), tz)


def load_tick_csv(path: str | Path, tz: str, interval: str) -> pd.DataFrame:
    """Tick CSV (time, price, size[, side]) -> bars with real delta."""
    t = pd.read_csv(path)
    t.columns = [str(c).strip().lower() for c in t.columns]
    tcol = next(c for c in _TIME_COLS if c in t.columns)
    t.index = _parse_times(t[tcol])
    t.index = t.index.tz_localize(tz) if t.index.tz is None else t.index.tz_convert(tz)
    return ticks_to_bars(t.drop(columns=[tcol]), interval)


def load_yfinance(symbol: str, interval: str, start: str | None, end: str | None, tz: str) -> pd.DataFrame:
    import yfinance as yf

    yf_interval = interval.replace("min", "m")
    df = yf.download(
        symbol, start=start, end=end, interval=yf_interval, auto_adjust=False,
        progress=False, multi_level_index=False,
    )
    if df is None or df.empty:
        raise ValueError(
            f"yfinance returned no data for {symbol} {interval} (Yahoo serves intraday <1h only for the last 60 days)"
        )
    return _normalise(df, tz)


def load_yfinance_daily(symbol: str, start: str | None, end: str | None, tz: str) -> pd.DataFrame:
    import yfinance as yf

    s = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d") if start else None
    df = yf.download(symbol, start=s, end=end, interval="1d", auto_adjust=False,
                     progress=False, multi_level_index=False)
    df = _normalise(df, tz)
    df.index = pd.DatetimeIndex(df.index.date, name="session_date")
    return df


def yfinance_market_cap(symbol: str) -> float | None:
    """Market cap in the listing currency (None if unavailable)."""
    try:
        import yfinance as yf

        mc = yf.Ticker(symbol).fast_info.get("marketCap")
        return float(mc) if mc else None
    except Exception as e:  # network / symbol errors
        log.warning("market cap lookup failed for %s: %s", symbol, e)
        return None


def synthetic(
    symbol: str, market: Market, start: str, end: str, interval: str = "5min", seed: int | None = None,
    start_price: float = 500.0, daily_vol: float = 0.02, base_volume: float = 50_000.0,
) -> pd.DataFrame:
    """Random-walk intraday bars with a U-shaped volume curve, inside the market session."""
    rng = np.random.default_rng([0 if seed is None else int(seed), zlib.crc32(symbol.encode())])
    step = pd.Timedelta(interval)
    days = pd.bdate_range(start, end)
    s0 = datetime.combine(datetime(2000, 1, 1), market.session_start)
    s1 = datetime.combine(datetime(2000, 1, 1), market.session_end)
    n_per_day = int((s1 - s0) / step)
    sigma = daily_vol / np.sqrt(max(n_per_day, 1))
    rows = []
    price = start_price
    for d in days:
        price *= float(np.exp(rng.normal(0, daily_vol)))  # overnight gap
        t0 = pd.Timestamp(datetime.combine(d.date(), market.session_start), tz=market.timezone)
        u = np.linspace(-1, 1, n_per_day)
        vol_curve = base_volume * (0.6 + 1.4 * u**2)
        for k in range(n_per_day):
            o = price
            r = rng.normal(0, sigma, 4)
            path = o * np.exp(np.cumsum(r))
            c = float(path[-1])
            h = float(max(o, path.max()))
            lo = float(min(o, path.min()))
            v = float(max(1.0, vol_curve[k] * rng.lognormal(0, 0.4)))
            rows.append((t0 + k * step, o, h, lo, c, v))
            price = c
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
    tick = market.tick_size
    for c in ("open", "high", "low", "close"):
        df[c] = (df[c] / tick).round() * tick
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def infer_interval(df: pd.DataFrame) -> pd.Timedelta:
    d = df.index.to_series().diff().dropna()
    d = d[d > pd.Timedelta(0)]
    return d.mode().iloc[0] if not d.empty else pd.Timedelta("5min")


def prepare_session(df: pd.DataFrame, market: Market, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Trim to [session_start, session_end) and add session bookkeeping columns.

    Added columns: session_date (date), bar_end (Timestamp), is_last_bar (bool).
    """
    df = df.tz_convert(market.timezone) if df.index.tz is not None else df.tz_localize(market.timezone)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz=market.timezone)]
    if end:
        df = df[df.index < pd.Timestamp(end, tz=market.timezone) + pd.Timedelta(days=1)]
    t = df.index.time
    df = df[(t >= market.session_start) & (t < market.session_end)].copy()
    if df.empty:
        return df
    step = infer_interval(df)
    df["session_date"] = df.index.date
    df["bar_end"] = df.index + step
    nxt = pd.Series(df["session_date"].to_numpy()).shift(-1).to_numpy()
    last_by_date = pd.Series(df["session_date"].to_numpy()) != pd.Series(nxt)
    end_clock = pd.Series(
        [datetime.combine(d, market.session_end) for d in df["session_date"]]
    ).dt.tz_localize(market.timezone)
    hits_end = (df["bar_end"].reset_index(drop=True) >= end_clock).to_numpy()
    df["is_last_bar"] = last_by_date.to_numpy() | hits_end
    return df


def daily_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("session_date")
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }
    )
    out.index = pd.DatetimeIndex(out.index, name="session_date")
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "delta" in df.columns:
        agg["delta"] = "sum"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])


def load_symbol(cfg, market: Market, symbol: str, module_cfg: dict, start: str | None, end: str | None) -> pd.DataFrame:
    """Load base bars for one symbol according to [data] config (module may override source)."""
    dc = cfg.data
    source = module_cfg.get("data_source", dc.get("source", "csv"))
    interval = module_cfg.get("base_interval", dc.get("interval", "5min"))
    if source == "csv":
        csv_dir = cfg.resolve(dc.get("csv_dir", "data"))
        fname = symbol.replace("=", "_").replace("^", "") + ".csv"
        df = load_csv(csv_dir / fname, market.timezone)
    elif source == "ticks":
        tick_dir = cfg.resolve(dc.get("tick_dir", "data/ticks"))
        fname = symbol.replace("=", "_").replace("^", "") + ".csv"
        df = load_tick_csv(tick_dir / fname, market.timezone, interval)
    elif source == "yfinance":
        df = load_yfinance(symbol, interval, start, end, market.timezone)
    elif source == "synthetic":
        sd = dc.get("synthetic", {})
        price = sd.get("start_price_future", 20000.0) if market.asset_class == "future" else sd.get("start_price_stock", 500.0)
        bvol = sd.get("base_volume_future", 2000.0) if market.asset_class == "future" else sd.get("base_volume_stock", 50000.0)
        df = synthetic(symbol, market, start or "2025-01-01", end or "2025-06-30", interval,
                       seed=sd.get("seed"), start_price=price, base_volume=bvol)
    else:
        raise ValueError(f"unknown data source '{source}'")
    return df
