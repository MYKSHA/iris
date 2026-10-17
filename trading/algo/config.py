"""Config loading (TOML) and validation against the strategy spec ranges."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODULE_NAMES = ("vcp", "orb", "box")


def _parse_time(s: str) -> time:
    hh, mm = s.split(":")[:2]
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class Market:
    name: str
    timezone: str
    session_start: time
    session_end: time
    entry_delay_seconds: int
    currency: str
    fx_to_base: float
    tick_size: float
    point_value: float
    asset_class: str  # "stock" | "future"
    lot_size: int
    commission_per_unit: float
    commission_bps: float
    slippage_ticks: float

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "Market":
        return cls(
            name=name,
            timezone=d["timezone"],
            session_start=_parse_time(d["session_start"]),
            session_end=_parse_time(d["session_end"]),
            entry_delay_seconds=int(d.get("entry_delay_seconds", 0)),
            currency=d.get("currency", "USD"),
            fx_to_base=float(d.get("fx_to_base", 1.0)),
            tick_size=float(d["tick_size"]),
            point_value=float(d.get("point_value", 1.0)),
            asset_class=d.get("asset_class", "stock"),
            lot_size=int(d.get("lot_size", 1)),
            commission_per_unit=float(d.get("commission_per_unit", 0.0)),
            commission_bps=float(d.get("commission_bps", 0.0)),
            slippage_ticks=float(d.get("slippage_ticks", 0.0)),
        )


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    markets: dict[str, Market] = field(default_factory=dict)

    @property
    def account(self) -> dict[str, Any]:
        return self.raw.get("account", {})

    @property
    def engine(self) -> dict[str, Any]:
        return self.raw.get("engine", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw.get("backtest", {})

    @property
    def montecarlo(self) -> dict[str, Any]:
        return self.raw.get("montecarlo", {})

    def module(self, name: str) -> dict[str, Any]:
        return self.raw.get("modules", {}).get(name, {})

    def enabled_modules(self) -> list[str]:
        return [m for m in MODULE_NAMES if self.module(m).get("enabled", False)]

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.path.parent / p)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = Config(raw=raw, path=path)
    cfg.markets = {k: Market.from_dict(k, v) for k, v in raw.get("markets", {}).items()}
    validate(cfg)
    return cfg


class ConfigError(ValueError):
    pass


def _check_range(errors: list[str], warns: list[str], label: str, val, lo, hi, hard: bool = False):
    if val is None:
        return
    if not (lo <= val <= hi):
        msg = f"{label}={val} outside spec range [{lo}, {hi}]"
        (errors if hard else warns).append(msg)


def validate(cfg: Config) -> None:
    errors: list[str] = []
    warns: list[str] = []
    for m in MODULE_NAMES:
        mc = cfg.module(m)
        if not mc:
            continue
        mk = mc.get("market")
        if mk not in cfg.markets:
            errors.append(f"modules.{m}.market '{mk}' is not defined under [markets]")
        risk = mc.get("risk_pct")
        if risk is not None and risk > 2.0:
            errors.append(f"modules.{m}.risk_pct={risk} exceeds the 2% max risk per trade")

    v = cfg.module("vcp")
    if v:
        _check_range(errors, warns, "vcp.take_profit_pct", v.get("take_profit_pct"), 8.0, 13.0)
        _check_range(errors, warns, "vcp.max_position_pct", v.get("max_position_pct"), 10.0, 20.0)
        _check_range(errors, warns, "vcp.min_position_pct", v.get("min_position_pct"), 10.0, 20.0)
        _check_range(errors, warns, "vcp.min_contractions", v.get("min_contractions"), 3, 4, hard=True)
        _check_range(errors, warns, "vcp.max_contractions", v.get("max_contractions"), 3, 4, hard=True)
        if v.get("min_contractions", 3) > v.get("max_contractions", 4):
            errors.append("vcp.min_contractions > vcp.max_contractions")
    o = cfg.module("orb")
    if o:
        _check_range(errors, warns, "orb.stop_ticks", o.get("stop_ticks"), 5, 8)
        _check_range(errors, warns, "orb.target_ticks", o.get("target_ticks"), 10, 15)
    b = cfg.module("box")
    if b:
        if b.get("stop_mode", "box_low") not in ("box_low", "ema21", "tighter", "wider"):
            errors.append("box.stop_mode must be one of box_low|ema21|tighter|wider")
        if b.get("min_bars", 10) > b.get("max_bars", 30):
            errors.append("box.min_bars > box.max_bars")

    for w in warns:
        log.warning("config: %s", w)
    if errors:
        raise ConfigError("; ".join(errors))
