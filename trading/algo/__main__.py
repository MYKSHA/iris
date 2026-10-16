"""CLI: python -m algo --config config.toml [--modules vcp,orb,box] [--separate]"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import MODULE_NAMES, load_config
from .runner import format_report, run_once


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="algo", description="Multi-strategy intraday backtester (VCP / ORB / Box)")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--modules", help="comma list overriding [modules.*].enabled, e.g. vcp,box")
    ap.add_argument("--separate", action="store_true", help="also backtest every module standalone")
    ap.add_argument("--separate-only", action="store_true", help="only standalone runs, skip the combined run")
    ap.add_argument("--out", help="output directory (default: [backtest].output_dir/<timestamp>)")
    ap.add_argument("--no-montecarlo", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every entry/exit to the console")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    modules = [m.strip() for m in a.modules.split(",")] if a.modules else cfg.enabled_modules()
    bad = [m for m in modules if m not in MODULE_NAMES]
    if bad:
        ap.error(f"unknown modules: {bad}")
    if not modules:
        ap.error("no modules enabled — set [modules.<name>].enabled = true or pass --modules")

    out = Path(a.out) if a.out else cfg.resolve(cfg.backtest.get("output_dir", "results")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(out / "backtest.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO if a.verbose else logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(fh)
    root.addHandler(sh)

    runs: list[tuple[str, list[str]]] = []
    if not a.separate_only:
        runs.append(("combined" if len(modules) > 1 else modules[0], modules))
    if a.separate or a.separate_only:
        done = {name for name, _ in runs}
        runs += [(m, [m]) for m in modules if m not in done]

    for name, mods in runs:
        rep = run_once(cfg, mods, out / name, mc=not a.no_montecarlo)
        print(format_report(name, rep))
        print()
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
