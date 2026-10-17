"""Strategy interface shared by all modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import Market
from ..data import resample_ohlcv


@dataclass
class ExitRules:
    """Exit rules the engine applies to an open position, bar by bar on base data."""

    break_even_r: float | None = None       # move stop to entry once trade reaches +N R
    time_stop_minutes: float | None = None  # exit if no +min_ticks move within N minutes
    time_stop_min_ticks: float = 0.0
    eod_flatten: bool = True                # flat on the last bar of each session
    use_exit_signal: bool = False           # honour strategy exit_signal column (e.g. close < 9 EMA)


@dataclass
class Sizing:
    risk_pct: float = 2.0                    # max % of equity at risk per trade
    max_position_pct: float | None = None    # stocks: cap notional at % of equity
    min_position_pct: float | None = None    # stocks: skip trade if notional would be below this
    max_contracts: int | None = None         # futures: hard contract cap


SIGNAL_COLUMNS = ("entry", "stop", "target", "exit_signal")


class Strategy:
    """Generates entry/stop/target/exit signals aligned to the base bars of one symbol.

    Contract for `generate`: returns a DataFrame with the same index as `base` and
    columns entry (+1 long, -1 short, 0 none), stop, target (NaN = none) and
    exit_signal (bool). An entry at base bar i fills at that bar's close; signals
    must use only information available at that close.
    """

    name: str = "base"

    def __init__(self, cfg: dict[str, Any], market: Market):
        self.cfg = cfg
        self.market = market
        self.signal_timeframe: str | None = cfg.get("signal_timeframe")
        self.max_trades_per_day: int | None = cfg.get("max_trades_per_day")
        self.exit_rules = ExitRules()
        self.sizing = Sizing(
            risk_pct=float(cfg.get("risk_pct", 2.0)),
            max_position_pct=cfg.get("max_position_pct"),
            min_position_pct=cfg.get("min_position_pct"),
            max_contracts=cfg.get("max_contracts"),
        )

    # ---- helpers -------------------------------------------------------
    def to_signal_tf(self, base: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Resample base bars to the signal timeframe.

        Returns (sig_df, last_pos) where last_pos[k] is the positional index of the
        last base bar inside signal bar k — the bar at whose close signal k is known.
        """
        tf = self.signal_timeframe
        if tf is None or pd.Timedelta(tf) <= (base["bar_end"].iloc[0] - base.index[0]):
            sig = base[[c for c in ("open", "high", "low", "close", "volume", "delta") if c in base.columns]].copy()
            sig["session_date"] = base["session_date"].to_numpy()
            return sig, np.arange(len(base))
        cols = base[[c for c in ("open", "high", "low", "close", "volume", "delta") if c in base.columns]]
        sig = resample_ohlcv(cols, tf)
        keys = base.index.floor(tf)
        pos = pd.Series(np.arange(len(base)), index=base.index).groupby(keys).max()
        sig = sig.loc[sig.index.isin(pos.index)]
        last_pos = pos.reindex(sig.index).to_numpy()
        sig["session_date"] = base["session_date"].to_numpy()[last_pos]
        return sig, last_pos

    @staticmethod
    def empty_signals(base: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=base.index)
        out["entry"] = 0
        out["stop"] = np.nan
        out["target"] = np.nan
        out["exit_signal"] = False
        return out

    @staticmethod
    def place(out: pd.DataFrame, last_pos: np.ndarray, k: int, **vals) -> None:
        i = int(last_pos[k])
        for col, v in vals.items():
            out.iat[i, out.columns.get_loc(col)] = v

    def stop_buffer(self, price: float) -> float:
        pct = float(self.cfg.get("stop_buffer_pct", 0.0))
        ticks = float(self.cfg.get("stop_buffer_ticks", 0.0))
        return price * pct / 100.0 + ticks * self.market.tick_size

    # ---- override -----------------------------------------------------
    def generate(self, symbol: str, base: pd.DataFrame, daily: pd.DataFrame | None, meta: dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError
