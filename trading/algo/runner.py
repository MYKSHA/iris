"""Wires config -> data -> strategies -> engine -> metrics / Monte Carlo -> files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data as D
from .config import Config
from .engine import Backtester, Feed
from .metrics import compute_metrics, monte_carlo
from .strategies import REGISTRY, Strategy

log = logging.getLogger(__name__)


def build_strategies(cfg: Config, modules: list[str]) -> list[Strategy]:
    priority = cfg.engine.get("priority", ["orb", "vcp", "box"])
    ordered = [m for m in priority if m in modules] + [m for m in modules if m not in priority]
    return [REGISTRY[m](cfg.module(m), cfg.markets[cfg.module(m)["market"]]) for m in ordered]


def _market_cap_cr(cfg: Config, mod: dict[str, Any], symbol: str) -> float | None:
    caps = mod.get("market_caps_cr", {})
    if symbol in caps:
        return float(caps[symbol])
    if mod.get("market_cap_source", "config") == "yfinance":
        mc = D.yfinance_market_cap(symbol)
        return mc / 1e7 if mc else None  # INR -> crore
    return None


def _daily(cfg: Config, mod: dict[str, Any], symbol: str, base: pd.DataFrame, market) -> pd.DataFrame:
    src = mod.get("daily_source", "intraday")
    if src == "yfinance":
        return D.load_yfinance_daily(symbol, cfg.backtest.get("start"), cfg.backtest.get("end"), market.timezone)
    if src == "csv":
        p = cfg.resolve(cfg.data.get("daily_dir", "data/daily")) / (symbol.replace("=", "_").replace("^", "") + ".csv")
        df = D.load_csv(p, market.timezone)
        df.index = pd.DatetimeIndex(df.index.date, name="session_date")
        return df
    return D.daily_from_intraday(base)


def load_feeds(cfg: Config, strategies: list[Strategy]) -> list[Feed]:
    bt = cfg.backtest
    start, end = bt.get("start"), bt.get("end")
    warm = int(bt.get("warmup_days", 45))
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=warm)).strftime("%Y-%m-%d") if start else None
    feeds: dict[tuple[str, str, str], Feed] = {}
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    for strat in strategies:
        mod = strat.cfg
        m = strat.market
        interval = mod.get("base_interval", cfg.data.get("interval", "5min"))
        source = mod.get("data_source", cfg.data.get("source", "csv"))
        for sym in mod.get("symbols", []):
            key = (m.name, sym, interval)
            if key not in cache:
                raw = D.load_symbol(cfg, m, sym, mod, load_start, end)
                cache[key] = D.prepare_session(raw, m, load_start, end)
            base = cache[key]
            if base.empty:
                log.warning("%s: no bars for %s inside session/date range", strat.name, sym)
                continue
            meta: dict[str, Any] = {"market_cap_cr": _market_cap_cr(cfg, mod, sym) if strat.name == "vcp" else None}
            if strat.name == "orb" and source == "ticks":
                p = cfg.resolve(cfg.data.get("tick_dir", "data/ticks")) / (sym.replace("=", "_").replace("^", "") + ".csv")
                t = pd.read_csv(p)
                t.columns = [c.lower() for c in t.columns]
                tcol = next(c for c in ("time", "datetime", "timestamp", "ts") if c in t.columns)
                idx = D._parse_times(t.pop(tcol))
                t.index = idx.tz_localize(m.timezone) if idx.tz is None else idx.tz_convert(m.timezone)
                meta["ticks"] = t
            daily = _daily(cfg, mod, sym, base, m) if strat.name == "vcp" else None
            sig = strat.generate(sym, base, daily, meta)
            if key not in feeds:
                feeds[key] = Feed(symbol=sym, market=m, base=base)
            feeds[key].signals[strat.name] = sig
            log.info("%s %s: %d bars, %d entry signals", strat.name, sym, len(base), int((sig["entry"] != 0).sum()))

    out = []
    for f in feeds.values():  # trade only inside [start, end]; warmup bars only fed indicators
        mask = np.ones(len(f.base), bool)
        if start:
            mask &= f.base.index >= pd.Timestamp(start, tz=f.market.timezone)
        f.base = f.base[mask]
        f.signals = {k: v[mask] for k, v in f.signals.items()}
        if len(f.base):
            out.append(f)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run_once(cfg: Config, modules: list[str], out_dir: Path, mc: bool = True) -> dict[str, Any]:
    strategies = build_strategies(cfg, modules)
    feeds = load_feeds(cfg, strategies)
    initial = float(cfg.account.get("initial_equity", 1_000_000))
    res = Backtester(strategies, feeds, initial, cfg.engine.get("one_position", True),
                     cfg.engine.get("intrabar", "stop_first")).run()
    trades, equity = res["trades"], res["equity"]
    rf = float(cfg.backtest.get("risk_free_annual", 0.0))
    report: dict[str, Any] = {"modules": modules, "overall": compute_metrics(trades, equity, initial, rf)}
    if len(modules) > 1 and not trades.empty:
        report["by_module"] = {m: compute_metrics(trades[trades["strategy"] == m], None, initial)
                               for m in modules}
    if mc and not trades.empty:
        mcc = cfg.montecarlo
        report["montecarlo"] = monte_carlo(
            trades["return_pct"].to_numpy(), int(mcc.get("n_sims", 1000)), mcc.get("method", "bootstrap"),
            mcc.get("seed", 42), float(mcc.get("ruin_drawdown_pct", 30.0)),
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity.rename_axis("time_utc").to_csv(out_dir / "equity.csv")
    (out_dir / "report.json").write_text(json.dumps(_jsonable(report), indent=2, default=str))
    return report


def format_report(name: str, rep: dict[str, Any]) -> str:
    m = rep["overall"]
    f = lambda x, d=2: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"  # noqa: E731
    lines = [
        f"== {name} ({', '.join(rep['modules'])}) ==",
        f"  trades            {m['trades']}",
    ]
    if m["trades"]:
        lines += [
            f"  win rate          {f(m['win_rate_pct'])}%",
            f"  profit factor     {f(m['profit_factor'])}",
            f"  avg R-multiple    {f(m['avg_r'])}   (win {f(m['avg_win_r'])}R / loss {f(m['avg_loss_r'])}R, R:R {f(m['reward_to_risk'])})",
            f"  total return      {f(m['total_return_pct'])}%",
            f"  max drawdown      {f(m['max_drawdown_pct'])}%",
            f"  sharpe (daily)    {f(m['sharpe'])}",
            f"  max losing streak {m['max_consecutive_losses']}",
            f"  exits             {m['exit_reasons']}",
            f"  baseline          {m['baseline_check']}",
        ]
    for mod, mm in rep.get("by_module", {}).items():
        if mm["trades"]:
            lines.append(f"  [{mod}] trades={mm['trades']} win={f(mm['win_rate_pct'])}% PF={f(mm['profit_factor'])} "
                         f"avgR={f(mm['avg_r'])} maxDD={f(mm['max_drawdown_pct'])}%")
        else:
            lines.append(f"  [{mod}] trades=0")
    mc = rep.get("montecarlo")
    if mc and mc.get("n_sims"):
        fr, dd = mc["final_return_pct"], mc["max_drawdown_pct"]
        ruin_key = next(k for k in mc if k.startswith("prob_drawdown_over_"))
        lines += [
            f"  monte carlo ({mc['method']}, {mc['n_sims']} sims x {mc['trades_per_sim']} trades)",
            f"    final return  p5 {f(fr['p5'])}%  p50 {f(fr['p50'])}%  p95 {f(fr['p95'])}%",
            f"    max drawdown  p50 {f(dd['p50'])}%  p95 {f(dd['p95'])}%  worst {f(dd['worst'])}%",
            f"    P(loss) {f(mc['prob_loss_pct'])}%   P({ruin_key.removeprefix('prob_')}) {f(mc[ruin_key])}%",
        ]
    return "\n".join(lines)
