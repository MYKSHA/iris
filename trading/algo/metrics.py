"""Performance metrics and Monte Carlo robustness testing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def max_drawdown_pct(equity: pd.Series | np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.max((peak - e) / peak) * 100.0)


def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252, risk_free_annual: float = 0.0) -> float:
    if equity.empty:
        return float("nan")
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return float("nan")
    excess = rets - risk_free_annual / periods_per_year
    return float(excess.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year))


def longest_streak(results: list[bool], value: bool) -> int:
    best = cur = 0
    for r in results:
        cur = cur + 1 if r == value else 0
        best = max(best, cur)
    return best


def compute_metrics(trades: pd.DataFrame, equity: pd.Series | None, initial_equity: float,
                    risk_free_annual: float = 0.0) -> dict[str, Any]:
    n = len(trades)
    out: dict[str, Any] = {"trades": n}
    if n == 0:
        out.update(win_rate_pct=0.0, profit_factor=float("nan"), avg_r=float("nan"), max_drawdown_pct=0.0,
                   sharpe=float("nan"), total_return_pct=0.0)
        return out
    pnl = trades["pnl"].to_numpy(float)
    r = trades["r_multiple"].to_numpy(float)
    wins, losses = pnl > 0, pnl <= 0
    gp, gl = pnl[wins].sum(), -pnl[losses].sum()
    avg_win_r = float(r[wins].mean()) if wins.any() else 0.0
    avg_loss_r = float(r[losses].mean()) if losses.any() else 0.0
    if equity is not None and not equity.empty:
        mdd = max_drawdown_pct(np.r_[initial_equity, equity.to_numpy()])
        sharpe = sharpe_ratio(equity, risk_free_annual=risk_free_annual)
        final = float(equity.iloc[-1])
    else:  # trade-sequence equity (used for per-module slices of a combined run)
        curve = initial_equity + np.cumsum(pnl)
        mdd = max_drawdown_pct(np.r_[initial_equity, curve])
        sharpe = float("nan")
        final = float(curve[-1])
    out.update(
        wins=int(wins.sum()),
        losses=int(losses.sum()),
        win_rate_pct=float(wins.mean() * 100.0),
        profit_factor=float(gp / gl) if gl > 0 else float("inf"),
        avg_r=float(r.mean()),
        median_r=float(np.median(r)),
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        reward_to_risk=float(avg_win_r / abs(avg_loss_r)) if avg_loss_r < 0 else float("inf"),
        expectancy_r=float(r.mean()),
        total_pnl=float(pnl.sum()),
        total_return_pct=float((final / initial_equity - 1.0) * 100.0),
        max_drawdown_pct=mdd,
        sharpe=sharpe,
        max_consecutive_losses=longest_streak(list(wins), False),
        avg_bars_held=float(trades["bars_held"].mean()),
        exit_reasons=trades["exit_reason"].value_counts().to_dict(),
    )
    out["baseline_check"] = baseline_check(out)
    return out


def baseline_check(m: dict[str, Any], min_win_rate: float = 30.0, min_rr: float = 2.0, min_trades: int = 30) -> str:
    """Spec baseline: ~30% win rate with a 2:1+ reward-to-risk ratio."""
    if m.get("trades", 0) < min_trades:
        return f"INSUFFICIENT SAMPLE ({m.get('trades', 0)} trades < {min_trades})"
    wr_ok = m.get("win_rate_pct", 0) >= min_win_rate * 0.9  # "~30%" -> allow 27%+
    rr_ok = m.get("reward_to_risk", 0) >= min_rr
    pf_ok = m.get("profit_factor", 0) > 1.0
    flags = [f"win_rate {'OK' if wr_ok else 'LOW'}", f"R:R {'OK' if rr_ok else 'LOW'}",
             f"PF {'OK' if pf_ok else '<=1'}"]
    return ("PASS" if (wr_ok and rr_ok and pf_ok) else "FAIL") + " (" + ", ".join(flags) + ")"


def monte_carlo(trade_returns_pct: np.ndarray, n_sims: int = 1000, method: str = "bootstrap",
                seed: int | None = 42, ruin_dd_pct: float = 30.0, initial_equity: float = 1.0) -> dict[str, Any]:
    """Resample per-trade % returns (compounded) to test robustness of the trade sequence.

    method="shuffle": permute the order (same trades, different sequence — tests path/drawdown risk).
    method="bootstrap": sample with replacement (tests sensitivity to the trade mix).
    """
    r = np.asarray(trade_returns_pct, dtype=float) / 100.0
    if r.size == 0:
        return {"n_sims": 0}
    rng = np.random.default_rng(seed)
    n = r.size
    if method == "shuffle":
        samples = np.array([rng.permutation(r) for _ in range(n_sims)])
    elif method == "bootstrap":
        samples = rng.choice(r, size=(n_sims, n), replace=True)
    else:
        raise ValueError("montecarlo.method must be 'shuffle' or 'bootstrap'")
    curves = initial_equity * np.cumprod(1.0 + samples, axis=1)
    curves = np.hstack([np.full((n_sims, 1), initial_equity), curves])
    peaks = np.maximum.accumulate(curves, axis=1)
    dds = ((peaks - curves) / peaks).max(axis=1) * 100.0
    finals = (curves[:, -1] / initial_equity - 1.0) * 100.0
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    return {
        "n_sims": n_sims,
        "method": method,
        "trades_per_sim": n,
        "final_return_pct": {"p5": pct(finals, 5), "p25": pct(finals, 25), "p50": pct(finals, 50),
                             "p75": pct(finals, 75), "p95": pct(finals, 95)},
        "max_drawdown_pct": {"p5": pct(dds, 5), "p50": pct(dds, 50), "p95": pct(dds, 95), "worst": float(dds.max())},
        "prob_loss_pct": float((finals < 0).mean() * 100.0),
        f"prob_drawdown_over_{ruin_dd_pct:g}pct": float((dds > ruin_dd_pct).mean() * 100.0),
    }
