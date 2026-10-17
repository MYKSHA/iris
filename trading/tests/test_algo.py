from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo import data as D  # noqa: E402
from algo.config import Market, load_config  # noqa: E402
from algo.engine import Backtester, Feed  # noqa: E402
from algo.metrics import compute_metrics, max_drawdown_pct, monte_carlo  # noqa: E402
from algo.orderflow import estimate_delta, poc, ticks_to_bars, volume_profile_from_bars  # noqa: E402
from algo.strategies import BoxStrategy, ORBStrategy, VCPStrategy  # noqa: E402
from algo.strategies.box import find_boxes  # noqa: E402


def mk_market(**kw) -> Market:
    d = dict(name="nse", timezone="Asia/Kolkata", session_start=time(9, 15), session_end=time(15, 30),
             entry_delay_seconds=0, currency="INR", fx_to_base=1.0, tick_size=0.05, point_value=1.0,
             asset_class="stock", lot_size=1, commission_per_unit=0.0, commission_bps=0.0, slippage_ticks=0.0)
    d.update(kw)
    return Market(**d)


NQ = mk_market(name="cme_nq", timezone="America/New_York", session_start=time(9, 30), session_end=time(15, 30),
               entry_delay_seconds=60, tick_size=0.25, point_value=20.0, asset_class="future", currency="USD")


def bars_from_closes(closes, vols, market: Market, day="2025-03-03", step="5min", wick=0.05):
    """Sequential bars across sessions: open = previous close, wick added beyond body."""
    step = pd.Timedelta(step)
    rows, ts = [], []
    d = pd.Timestamp(day)
    t = pd.Timestamp(datetime.combine(d.date(), market.session_start), tz=market.timezone)
    prev = closes[0]
    for c, v in zip(closes, vols):
        o = prev
        rows.append((o, max(o, c) + wick, min(o, c) - wick, c, v))
        ts.append(t)
        prev = c
        t = t + step
        if t.time() >= market.session_end:
            d = d + pd.offsets.BDay(1)
            t = pd.Timestamp(datetime.combine(d.date(), market.session_start), tz=market.timezone)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(ts))
    return D.prepare_session(df, market)


def seg(a, b, n):
    return list(np.linspace(a, b, n + 1)[1:])


# ----------------------------------------------------------------- orderflow
def test_estimate_delta():
    df = pd.DataFrame({"open": [10, 10, 10], "high": [12, 10, 11], "low": [9, 10, 9],
                       "close": [11.5, 10, 9], "volume": [300, 100, 200]})
    assert estimate_delta(df).tolist() == pytest.approx([150.0, 0.0, -100.0])


def test_poc_and_profile():
    df = pd.DataFrame({"low": [100.0, 101.0], "high": [101.75, 101.75], "volume": [800, 900]})
    prof = volume_profile_from_bars(df, 0.25)
    assert prof.sum() == pytest.approx(1700)
    assert 101.0 <= poc(prof, 0.25) <= 102.0


def test_ticks_to_bars_real_delta():
    idx = pd.date_range("2025-03-03 09:30", periods=6, freq="20s", tz="America/New_York")
    t = pd.DataFrame({"price": [100, 100.25, 100.5, 100.25, 100, 100.25], "size": [1, 2, 3, 1, 1, 5],
                      "side": ["B", "B", "B", "S", "S", "B"]}, index=idx)
    b = ticks_to_bars(t, "1min")
    assert b["delta"].tolist() == [6.0, 3.0]
    assert b["volume"].tolist() == [6.0, 7.0]


# ----------------------------------------------------------------- VCP
VCP_CFG = dict(use_universe_filter=False, use_momentum_filter=False, signal_timeframe=None, pivot_bars=3, stop_buffer_pct=0.0, take_profit_pct=10.0)


def vcp_path(p3_low=101.57):
    closes, vols = [], []
    def add(a, b, n, v):
        closes.extend(seg(a, b, n)); vols.extend([v] * n)
    closes.append(90.0); vols.append(1000)
    add(90, 95, 30, 1000)          # warmup / uptrend
    add(95, 110, 10, 1200)          # rally into H1
    add(110, 100, 6, 1500)          # pullback 1: 9.09%
    add(100, 107, 6, 900)
    add(107, 101.17, 6, 1000)       # pullback 2: 5.45% (0.60x)
    add(101.17, 105, 6, 800)
    add(105, p3_low, 6, 600)        # pullback 3: 3.27% (0.60x) by default
    add(p3_low, 102.3, 7, 700)      # drift up under trendline
    breakout_at = len(closes)
    add(102.3, 105.5, 1, 6000)      # breakout bar, big volume
    add(105.5, 106, 5, 900)
    return closes, vols, breakout_at


def test_vcp_detects_breakout():
    closes, vols, bo = vcp_path()
    base = bars_from_closes(closes, vols, mk_market())
    s = VCPStrategy(VCP_CFG, mk_market())
    daily = D.daily_from_intraday(base)
    sig = s.generate("X", base, daily, {"market_cap_cr": None})
    entries = np.flatnonzero(sig["entry"].to_numpy() != 0)
    assert entries.tolist() == [bo]
    assert sig["stop"].iloc[bo] == pytest.approx(101.57 - 0.05, abs=1e-6)
    assert sig["target"].iloc[bo] == pytest.approx(105.5 * 1.10)


def test_vcp_rejects_expanding_contractions():
    # pullback 3 of 4.8% vs pullback 2 of 5.45% -> ratio 0.88 > 0.70, not a VCP
    closes, vols, _ = vcp_path(p3_low=99.96)
    base = bars_from_closes(closes, vols, mk_market())
    sig = VCPStrategy(VCP_CFG, mk_market()).generate("X", base, D.daily_from_intraday(base), {})
    assert (sig["entry"] != 0).sum() == 0


def test_vcp_universe_filter_blocks():
    closes, vols, _ = vcp_path()
    base = bars_from_closes(closes, vols, mk_market())
    cfg = dict(VCP_CFG, use_universe_filter=True, require_market_cap=True, min_avg_volume=0)
    sig = VCPStrategy(cfg, mk_market()).generate("X", base, D.daily_from_intraday(base), {"market_cap_cr": None})
    assert (sig["entry"] != 0).sum() == 0
    sig = VCPStrategy(cfg, mk_market()).generate("X", base, D.daily_from_intraday(base), {"market_cap_cr": 9000})
    assert (sig["entry"] != 0).sum() == 0


def test_momentum_filter_window():
    idx = pd.DatetimeIndex(pd.bdate_range("2025-01-01", periods=40))
    close = pd.Series(100.0, index=idx)
    close.iloc[-1:] = 145.0  # prior-day close below is 100 -> not yet visible
    daily = pd.DataFrame({"close": close, "volume": 1e6})
    s = VCPStrategy(dict(require_market_cap=False), mk_market())
    f = s.daily_filter(daily, {})
    assert not f.iloc[-1]
    daily.loc[idx[-2], "close"] = 145.0
    f = s.daily_filter(daily, {})
    assert f.iloc[-1]          # +45% over 20-25 days as of prior close
    daily.loc[idx[-2], "close"] = 160.0
    assert not s.daily_filter(daily, {}).iloc[-1]  # +60% exceeds 50% cap


# ----------------------------------------------------------------- Box
def test_find_boxes_and_box_breakout():
    rng = np.random.default_rng(0)
    pre = list(100 + np.cumsum(rng.normal(0, 0.4, 40)))
    p0 = pre[-1]
    box = [p0 + (0.8 if i % 2 else -0.8) * p0 / 100 for i in range(20)]  # ~1.6% range
    closes = pre + box + [p0 * 1.03, p0 * 1.031]
    vols = [2000] * 40 + [800] * 20 + [8000, 1000]
    base = bars_from_closes(closes, vols, mk_market(), wick=0.0)
    s = BoxStrategy(dict(signal_timeframe=None, stop_buffer_pct=0.0, take_profit_r=2.0), mk_market())
    sig = s.generate("X", base, None, {})
    hits = np.flatnonzero(sig["entry"].to_numpy())
    assert 60 in hits
    stop = sig["stop"].iloc[60]
    assert stop == pytest.approx(min(base["low"].iloc[40:60].min(), base["low"].iloc[39:60].min()), rel=5e-3)
    entry = base["close"].iloc[60]
    assert sig["target"].iloc[60] == pytest.approx(entry + 2 * (entry - stop))


# ----------------------------------------------------------------- ORB
def test_orb_long_signal_and_levels():
    m = NQ
    # 09:30-10:00 OR between 20000-20010, then breakout above with buying
    closes = seg(20000, 20010, 3) + seg(20010, 20002, 3) + [20012, 20016, 20020]
    vols = [1000] * 9
    base = bars_from_closes([20000] + closes, [1000] + vols, m, step="5min", wick=0.25)
    s = ORBStrategy(dict(signal_timeframe=None, stop_ticks=6, target_ticks=12), m)
    sig = s.generate("NQ", base, None, {})
    k = np.flatnonzero(sig["entry"].to_numpy())
    assert len(k) == 1
    k = int(k[0])
    assert base.index[k].time() >= time(10, 0)
    orh = base[base.index.time < time(10, 0)]["high"].max()
    assert sig["entry"].iloc[k] == 1
    assert sig["stop"].iloc[k] == pytest.approx(orh - 6 * 0.25)
    assert sig["target"].iloc[k] == pytest.approx(base["close"].iloc[k] + 12 * 0.25)


# ----------------------------------------------------------------- no lookahead
@pytest.mark.parametrize("cls,cfg", [
    (VCPStrategy, dict(VCP_CFG)),
    (BoxStrategy, dict(signal_timeframe="15min")),
    (ORBStrategy, dict(signal_timeframe="5min")),
])
def test_signals_do_not_use_future_bars(cls, cfg):
    m = NQ if cls is ORBStrategy else mk_market()
    raw = D.synthetic("SYM", m, "2025-01-01", "2025-02-15", "5min", seed=3,
                      start_price=20000 if cls is ORBStrategy else 500)
    if cls is VCPStrategy:
        closes, vols, _ = vcp_path()
        full = bars_from_closes(closes, vols, m)
    else:
        full = D.prepare_session(raw, m)
    s = cls(cfg, m)
    sig_full = s.generate("S", full, D.daily_from_intraday(full), {})
    cut = int(len(full) * 0.8)
    # cut on a signal-bar boundary for the resampled strategy
    while cfg.get("signal_timeframe") == "15min" and full.index[cut].minute % 15 != 0:
        cut += 1
    part = D.prepare_session(full[["open", "high", "low", "close", "volume"]].iloc[:cut], m)
    sig_part = s.generate("S", part, D.daily_from_intraday(part), {})
    a = sig_full.iloc[:cut]
    pd.testing.assert_series_equal(a["entry"], sig_part["entry"], check_names=False)


# ----------------------------------------------------------------- engine
class _Fixed:
    """Strategy stub with precomputed signals."""

    def __init__(self, name, market, **exit_kw):
        from algo.strategies.base import ExitRules, Sizing, Strategy
        self.__class__ = type("Fixed", (Strategy,), {"name": name})
        Strategy.__init__(self, {"risk_pct": 1.0}, market)
        self.exit_rules = ExitRules(**exit_kw)


def _feed(market, rows, step="5min", day="2025-03-03"):
    t0 = pd.Timestamp(datetime.combine(pd.Timestamp(day).date(), market.session_start), tz=market.timezone)
    idx = pd.date_range(t0, periods=len(rows), freq=step)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    return D.prepare_session(df, market)


def _sig(base, entries: dict[int, tuple[int, float, float]], exit_at=()):
    s = pd.DataFrame(index=base.index)
    s["entry"] = 0
    s["stop"] = np.nan
    s["target"] = np.nan
    s["exit_signal"] = False
    for i, (side, stop, tgt) in entries.items():
        s.iloc[i, 0], s.iloc[i, 1], s.iloc[i, 2] = side, stop, tgt
    for i in exit_at:
        s.iloc[i, 3] = True
    return s


def test_engine_breakeven_stop():
    m = mk_market()
    base = _feed(m, [(100, 100, 100, 100), (100, 101.5, 99.5, 101), (101, 101, 99.9, 100.2), (100, 100, 100, 100)])
    st = _Fixed("vcp", m, break_even_r=1.0, eod_flatten=True)
    f = Feed("A", m, base, {"vcp": _sig(base, {0: (1, 99.0, np.nan)})})
    res = Backtester([st], [f], 100_000).run()
    tr = res["trades"]
    assert len(tr) == 1
    assert tr.iloc[0]["exit_reason"] == "breakeven"
    assert tr.iloc[0]["exit_price"] == pytest.approx(100.0)
    assert tr.iloc[0]["r_multiple"] == pytest.approx(0.0)


def test_engine_stop_checked_before_target_and_sizing():
    m = mk_market()
    base = _feed(m, [(100, 100, 100, 100), (100, 103, 98, 101)])
    st = _Fixed("box", m)
    st.sizing.risk_pct, st.sizing.max_position_pct = 2.0, 20.0
    f = Feed("A", m, base, {"box": _sig(base, {0: (1, 99.0, 102.0)})})
    tr = Backtester([st], [f], 100_000).run()["trades"]
    assert tr.iloc[0]["exit_reason"] == "stop"
    # 2% risk = 2000 / 1.0 per share = 2000 sh, but 20% cap = 20000/100 = 200 sh
    assert tr.iloc[0]["qty"] == 200
    assert tr.iloc[0]["r_multiple"] == pytest.approx(-1.0)


def test_engine_intrabar_open_proximity():
    m = mk_market()
    base = _feed(m, [(100, 100, 100, 100), (101.8, 103, 98, 101)])
    st = _Fixed("box", m)
    f = Feed("A", m, base, {"box": _sig(base, {0: (1, 99.0, 102.0)})})
    tr = Backtester([st], [f], 100_000, intrabar="open_proximity").run()["trades"]
    assert tr.iloc[0]["exit_reason"] == "target"


def test_engine_time_stop_futures():
    m = NQ
    rows = [(20000, 20000, 20000, 20000)] + [(20000, 20000.25, 19999.75, 20000)] * 5
    base = _feed(m, rows, step="1min")
    st = _Fixed("orb", m, time_stop_minutes=3.0, time_stop_min_ticks=2.0)
    f = Feed("NQ", m, base, {"orb": _sig(base, {0: (1, 19998.0, 20003.0)})})
    tr = Backtester([st], [f], 1_000_000).run()["trades"]
    assert tr.iloc[0]["exit_reason"] == "time_stop"
    assert tr.iloc[0]["exit_time"] - tr.iloc[0]["entry_time"] == pd.Timedelta(minutes=3)


def test_engine_single_position_across_modules():
    m = mk_market()
    rows = [(100, 100, 100, 100)] * 6
    b1, b2 = _feed(m, rows), _feed(m, rows)
    s1, s2 = _Fixed("vcp", m), _Fixed("box", m)
    f1 = Feed("A", m, b1, {"vcp": _sig(b1, {0: (1, 99.0, np.nan)})})
    f2 = Feed("B", m, b2, {"box": _sig(b2, {0: (1, 99.0, np.nan), 2: (1, 99.0, np.nan)})})
    tr = Backtester([s1, s2], [f1, f2], 100_000).run()["trades"]
    assert len(tr) == 1 and tr.iloc[0]["strategy"] == "vcp" and tr.iloc[0]["exit_reason"] == "eod"


# ----------------------------------------------------------------- metrics / MC
def test_metrics_and_montecarlo():
    trades = pd.DataFrame({"pnl": [200.0, -100, -100, 300], "r_multiple": [2.0, -1, -1, 3],
                           "return_pct": [2.0, -1, -1, 3], "bars_held": [3, 2, 2, 5],
                           "exit_reason": ["target", "stop", "stop", "target"]})
    m = compute_metrics(trades, None, 10_000)
    assert m["win_rate_pct"] == 50.0
    assert m["profit_factor"] == pytest.approx(2.5)
    assert m["avg_r"] == pytest.approx(0.75)
    assert max_drawdown_pct([100, 120, 90, 130]) == pytest.approx(25.0)
    mc = monte_carlo(trades["return_pct"].to_numpy(), 200, "shuffle", seed=1)
    p = mc["final_return_pct"]
    assert p["p5"] == pytest.approx(p["p95"])  # permutation leaves compounded final return unchanged
    mc = monte_carlo(trades["return_pct"].to_numpy(), 200, "bootstrap", seed=1)
    assert mc["final_return_pct"]["p5"] < mc["final_return_pct"]["p95"]


def test_config_loads_and_validates(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    assert set(cfg.enabled_modules()) == {"vcp", "orb", "box"}
    bad = (Path(__file__).resolve().parents[1] / "config.toml").read_text().replace(
        "risk_pct                = 2.0", "risk_pct                = 3.0", 1)
    p = tmp_path / "c.toml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_config(p)


def test_cli_saves_config_copy(tmp_path):
    from algo.__main__ import main
    root = Path(__file__).resolve().parents[1]
    src = (root / "config.toml").read_text()
    src = src.replace('source   = "csv"', 'source   = "synthetic"', 1).replace(
        'end              = "2025-12-31"', 'end              = "2025-01-31"', 1)
    cfg = tmp_path / "run.toml"
    cfg.write_text(src)
    out = tmp_path / "out"
    assert main(["--config", str(cfg), "--modules", "box", "--out", str(out), "--no-montecarlo"]) == 0
    assert (out / "config.toml").read_text() == src
    assert (out / "box" / "report.json").exists()
