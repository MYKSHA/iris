"""Download Yahoo Finance bars into the CSV layout the backtester reads.

python -m algo.fetch --config config.toml --market nse --interval 5min CDSL.NS KPITTECH.NS
python -m algo.fetch --config config.toml --market cme_nq --interval 1min NQ=F

Yahoo limits: 1m = last 7 days, 2-30m = last 60 days, 1h = last 730 days. Run it
regularly and it appends/merges, so history accumulates past those limits.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .config import load_config
from .data import load_csv, load_yfinance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="algo.fetch")
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--market", required=True)
    ap.add_argument("--interval", default="5min")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    m = cfg.markets[a.market]
    out_dir = cfg.resolve(cfg.data.get("csv_dir", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym in a.symbols:
        df = load_yfinance(sym, a.interval, None, None, m.timezone) if a.interval != "1min" else \
            load_yfinance(sym, a.interval, (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), None, m.timezone)
        path = out_dir / (sym.replace("=", "_").replace("^", "") + ".csv")
        if path.exists():
            old = load_csv(path, m.timezone)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.rename_axis("time").to_csv(path)
        print(f"{sym}: {len(df)} bars {df.index[0]} -> {df.index[-1]} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
